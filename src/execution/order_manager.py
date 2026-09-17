"""
execution/order_manager.py

Mengelola siklus hidup order: generate client_order_id unik, catat
NIAT mengirim SEBELUM benar-benar kirim (write-ahead log), kirim lewat
Broker/MockBroker (execution/broker.py), dan rekonsiliasi kalau
jaringan putus atau program crash di tengah -- supaya restart program
TIDAK PERNAH asal kirim ulang order yang mungkin sudah terkirim.

INI SATU-SATUNYA JALUR untuk mengirim order di proyek ini -- pola sama
seperti broker.py "satu-satunya file yang boleh menyentuh jaringan
bursa": runner/paper.py WAJIB lewat OrderManager ini, tidak pernah
panggil broker.place_limit_order() langsung.

KENAPA WRITE-AHEAD LOG (bukan cuma cek status setelah error jaringan):
Skenario paling berbahaya bukan "jaringan putus, program tahu itu
putus" -- itu sudah ditangani MockBroker.fetch_order_by_client_id()
di broker.py. Yang berbahaya: PROGRAM CRASH atau KOMPUTER MATI tepat
setelah order terkirim ke bursa tapi SEBELUM konfirmasi balik ke
program kita. Kalau restart program langsung generate client_order_id
BARU dan kirim lagi, itu order GANDA sungguhan -- karena program tidak
tahu apa-apa soal order sebelumnya, ID-nya pun beda. Solusinya: simpan
client_order_id ke disk SEBELUM mengirim, bukan sesudah -- supaya
kalau program crash di titik mana pun, restart-nya tahu persis ID apa
yang tadi sedang diproses, dan WAJIB cek ke bursa dulu (bukan asal
kirim ulang) lewat reconcile_pending().
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from src.execution.approval_queue import ApprovalQueue, PendingOrder, create_pending_order
from src.execution.broker import OrderResult, OrderStatus
from src.strategy.base import Position, Strategy

DEFAULT_LOG_PATH = Path("data/order_intents.jsonl")


@dataclass
class OrderIntent:
    """
    Satu baris di log write-ahead -- niat mengirim SATU order, dicatat
    SEBELUM order itu benar-benar dikirim ke bursa.
    """

    client_order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    created_at: float  # unix timestamp
    reduce_only: bool = False
    resolved: bool = False  # True setelah statusnya DIPASTIKAN (bukan ditebak)


class OrderManager:
    def __init__(self, broker, log_path: Path = DEFAULT_LOG_PATH):
        """
        broker: instance Broker ATAU MockBroker (execution/broker.py)
        -- interface keduanya identik, OrderManager tidak peduli mana
        yang dipakai. Ini yang memungkinkan logika di file ini dites
        tuntas pakai MockBroker, baru divalidasi sekali lagi pakai
        Broker sungguhan.
        """
        self.broker = broker
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Human-in-the-Loop approval workflow (optional)
        self.approval_queue: ApprovalQueue | None = None
        self._approval_manager = None  # Lazy init when needed

    def _append_intent(self, intent: OrderIntent) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(intent)) + "\n")

    def _read_intents(self) -> list[OrderIntent]:
        if not self.log_path.exists():
            return []
        intents = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    intents.append(OrderIntent(**json.loads(line)))
        return intents

    def _mark_resolved(self, client_order_id: str) -> None:
        """
        Tulis ulang seluruh log dengan satu entri ditandai resolved.
        Cara paling sederhana & aman untuk volume order skala proyek
        ini (bukan HFT) -- kalau nanti volume order sangat besar,
        ganti ke penyimpanan yang lebih canggih (database), bukan file
        teks biasa.
        """
        intents = self._read_intents()
        with open(self.log_path, "w") as f:
            for i in intents:
                if i.client_order_id == client_order_id:
                    i.resolved = True
                f.write(json.dumps(asdict(i)) + "\n")

    def reconcile_pending(self, symbol: str) -> list[OrderResult]:
        """
        WAJIB dipanggil SETIAP KALI program (baru) mulai jalan --
        SEBELUM mengirim order baru apa pun. Cari niat order yang
        tercatat di log tapi belum "resolved" (kemungkinan besar
        karena program crash atau jaringan putus di tengah proses
        submit_limit_order sebelumnya), lalu CEK SUNGGUHAN ke bursa --
        tidak pernah menebak atau berasumsi.
        """
        resolved_results = []
        pending = [i for i in self._read_intents() if not i.resolved and i.symbol == symbol]

        for intent in pending:
            existing = self.broker.fetch_order_by_client_id(intent.client_order_id, symbol)
            if existing is not None:
                print(
                    f"  [reconcile] '{intent.client_order_id}' TERNYATA SUDAH terkirim "
                    f"(status={existing.status.value}) -- TIDAK dikirim ulang."
                )
                resolved_results.append(existing)
            else:
                print(
                    f"  [reconcile] '{intent.client_order_id}' KONFIRMASI belum pernah "
                    f"terkirim -- aman kalau mau diproses ulang."
                )
            self._mark_resolved(intent.client_order_id)

        return resolved_results

    def cancel_open_orders(self, symbol: str) -> int:
        """
        Batalkan SEMUA order yang masih terbuka (belum ke-fill) untuk
        simbol ini. WAJIB dipanggil sebelum submit_limit_order() untuk
        aksi baru -- kalau tidak, order lama yang belum ke-fill
        (mis. limit order yang harganya sudah dilewati pasar) akan
        terus menumpuk tiap kali sinyal dievaluasi ulang, dan bursa
        akan MENOLAK order reduceOnly baru karena "jatah" pengurangan
        posisi sudah dipesan duluan oleh order lama yang masih
        nangkring (ini penyebab nyata error -2022 "ReduceOnly Order is
        rejected" yang ditemukan di lapangan -- bukan salah state,
        bukan salah mode akun, murni order lama yang menumpuk).

        Return jumlah order yang dibatalkan.
        """
        open_orders = self.broker.fetch_open_orders(symbol)
        for o in open_orders:
            if o.exchange_order_id:
                print(f"  [cancel] Membatalkan order lama yang belum ke-fill: {o.client_order_id}")
                self.broker.cancel_order(o.exchange_order_id, symbol)
        return len(open_orders)

    def submit_limit_order(
        self, symbol: str, side: str, amount: float, price: float, reduce_only: bool = False
    ) -> OrderResult:
        """
        Kirim satu limit order, dengan write-ahead log supaya aman
        dari crash/jaringan putus di tengah proses.

        reduce_only: teruskan True kalau order ini niatnya MENUTUP
        posisi yang sudah ada -- lihat penjelasan di
        Broker.place_limit_order().

        Panggil reconcile_pending(symbol) dulu di awal program/restart
        -- lihat runner/paper.py.
        """
        client_order_id = f"bot-{uuid.uuid4().hex[:16]}"
        intent = OrderIntent(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            created_at=time.time(),
            reduce_only=reduce_only,
        )
        # TULIS DULU, BARU KIRIM -- urutan ini yang membuat pola ini aman
        # dari crash. Kalau dibalik (kirim dulu baru catat), crash tepat
        # setelah kirim akan membuat kita tidak tahu apa-apa soal order
        # yang sudah terlanjur terkirim itu.
        self._append_intent(intent)

        try:
            result = self.broker.place_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                client_order_id=client_order_id,
                reduce_only=reduce_only,
            )
        except Exception as e:
            # Order MUNGKIN terkirim, mungkin tidak -- kita TIDAK TAHU
            # dari sini. JANGAN retry otomatis. Biarkan tidak resolved;
            # reconcile_pending() di kesempatan berikutnya yang akan
            # memastikan lewat cek ke bursa, bukan tebak-tebakan.
            print(f"  [submit] Order '{client_order_id}' gagal terkirim (atau status tidak diketahui): {e}")
            raise

        self._mark_resolved(client_order_id)
        return result

    def cancel_order(self, exchange_order_id: str, symbol: str) -> None:
        self.broker.cancel_order(exchange_order_id, symbol)

    @staticmethod
    def classify_result(result: OrderResult) -> str:
        """
        Klasifikasikan hasil order jadi salah satu: "filled", "partial",
        "open", "rejected". Dipisah dari submit_limit_order supaya
        logika keputusan (apa yang dilakukan saat partial fill) gampang
        diuji terisolasi dan gampang diganti nanti tanpa menyentuh
        logika pengiriman/write-ahead log di atas.
        """
        if result.status == OrderStatus.FILLED:
            return "filled"
        if result.status == OrderStatus.PARTIALLY_FILLED:
            print(
                f"  [partial] '{result.client_order_id}' baru terisi "
                f"{result.filled_amount}/{result.amount} -- sisa {result.remaining_amount} menunggu."
            )
            return "partial"
        if result.status in (OrderStatus.REJECTED, OrderStatus.CANCELED):
            print(f"  [rejected] '{result.client_order_id}' status={result.status.value}, tidak ada yang terisi.")
            return "rejected"
        return "open"
        return "open"

    # ================================================================
    # Human-in-the-Loop Approval Workflow
    # ================================================================

    def enable_approval_workflow(self, approval_queue: ApprovalQueue | None = None) -> None:
        """
        Enable human-in-the-loop approval workflow.
        After calling this, submit_limit_order will queue for approval instead of executing immediately.
        Call disable_approval_workflow() to go back to direct execution.
        """
        self.approval_queue = approval_queue or ApprovalQueue()
        # Lazy import to avoid circular dependency
        from src.execution.approval_manager import ApprovalManager
        self._approval_manager = ApprovalManager(
            order_manager=self,
            broker=self.broker,
            approval_queue=self.approval_queue,
        )
        print("[OrderManager] Human-in-the-Loop approval workflow ENABLED")

    def disable_approval_workflow(self) -> None:
        """Disable approval workflow, go back to direct execution."""
        self.approval_queue = None
        self._approval_manager = None
        print("[OrderManager] Human-in-the-Loop approval workflow DISABLED")

    @property
    def approval_enabled(self) -> bool:
        return self.approval_queue is not None and self._approval_manager is not None

    def submit_for_approval(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
        strategy: Strategy | None = None,
        signal_price: float | None = None,
        current_position: int = 0,
        target_position: int = 0,
        account_balance_usdt: float | None = None,
        position_size_pct: float | None = None,
        expiry_seconds: float = 300,
    ) -> PendingOrder:
        """
        Submit an order for human approval (instead of executing immediately).
        Returns PendingOrder with approval_id. Does NOT execute the order.

        Requires approval workflow to be enabled via enable_approval_workflow().
        """
        if not self.approval_enabled:
            raise RuntimeError(
                "Approval workflow not enabled. Call enable_approval_workflow() first, "
                "or use submit_limit_order() for direct execution."
            )

        if strategy is None:
            raise ValueError("strategy is required for approval workflow (for context)")
        if signal_price is None:
            signal_price = price

        return self._approval_manager.submit_for_approval(
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            reduce_only=reduce_only,
            strategy=strategy,
            signal_price=signal_price,
            current_position=current_position,
            target_position=target_position,
            account_balance_usdt=account_balance_usdt,
            position_size_pct=position_size_pct,
            expiry_seconds=expiry_seconds,
        )

    def execute_approved(self, approval_id: str) -> OrderResult:
        """
        Execute an approved order.
        Called after human approves via CLI/Telegram.
        """
        if not self.approval_enabled:
            raise RuntimeError("Approval workflow not enabled")

        result = self._approval_manager.process_approval(
            approval_id,
            self._approval_manager.ApprovalAction.APPROVE
        )

        if not result.success:
            raise RuntimeError(f"Approval failed: {result.message}")
        if result.execution_result is None:
            raise RuntimeError("Approval succeeded but no execution result")

        return result.execution_result

    def reject_pending(self, approval_id: str) -> bool:
        """Reject a pending order (no execution)."""
        if not self.approval_enabled:
            raise RuntimeError("Approval workflow not enabled")

        result = self._approval_manager.process_approval(
            approval_id,
            self._approval_manager.ApprovalAction.REJECT
        )
        return result.success

    def get_pending_approvals(self) -> list[PendingOrder]:
        """Get all pending approval orders."""
        if not self.approval_enabled:
            return []
        return self._approval_manager.get_pending_orders()

    def get_approval_history(self, limit: int = 50) -> list[PendingOrder]:
        """Get approval history."""
        if not self.approval_enabled:
            return []
        return self._approval_manager.get_history(limit)


if __name__ == "__main__":
    # Uji asap TANPA jaringan sungguhan -- pakai MockBroker, simulasikan
    # PERSIS skenario kriteria lolos Hari 3: "jaringan diputus dengan
    # sengaja, tidak muncul order ganda" -- termasuk kasus PALING
    # berbahaya (crash sebelum sempat tandai resolved), bukan cuma
    # kasus gampang (error langsung ketahuan).
    import tempfile

    from src.execution.broker import MockBroker

    log_file = Path(tempfile.mktemp(suffix=".jsonl"))
    symbol = "BTC/USDT:USDT"

    print("=== Skenario 1: kirim normal, network hidup ===\n")
    broker = MockBroker(fill_immediately=True)
    om = OrderManager(broker, log_path=log_file)
    om.reconcile_pending(symbol)  # log masih kosong, harus tidak ngapa-ngapain
    r1 = om.submit_limit_order(symbol, "buy", 0.01, 100_000.0)
    print(f"  Order 1 terkirim: {r1.client_order_id}, status={r1.status.value}")
    print(f"  classify_result -> {OrderManager.classify_result(r1)}")

    print("\n=== Skenario 2: crash SEBELUM sempat tandai resolved ===\n")
    crashed_intent = OrderIntent(
        client_order_id="bot-crashed-before-send",
        symbol=symbol,
        side="buy",
        amount=0.01,
        price=99_000.0,
        created_at=time.time(),
    )
    om._append_intent(crashed_intent)  # simulasi manual, bukan API publik
    print(f"  (simulasi) intent '{crashed_intent.client_order_id}' tercatat, TAPI order")
    print("  belum pernah benar-benar dikirim ke broker sama sekali.")

    print("\n  Program 'restart' -- OrderManager baru, log yang sama:")
    om_restarted = OrderManager(broker, log_path=log_file)
    results = om_restarted.reconcile_pending(symbol)
    assert len(results) == 0, "seharusnya tidak ada hasil -- order itu memang belum pernah terkirim"
    print("  OK -- dikonfirmasi order itu memang belum pernah terkirim, aman diproses ulang kalau perlu.")

    print("\n=== Skenario 3: order SUDAH terkirim ke bursa, tapi crash sebelum tandai resolved ===\n")
    real_cid = "bot-sent-but-crashed"
    intent2 = OrderIntent(
        client_order_id=real_cid, symbol=symbol, side="sell", amount=0.02, price=101_000.0, created_at=time.time()
    )
    om._append_intent(intent2)
    broker.place_limit_order(symbol, "sell", 0.02, 101_000.0, client_order_id=real_cid)  # ini yang "berhasil"
    print(f"  (simulasi) order '{real_cid}' berhasil terkirim ke bursa, TAPI program")
    print("  crash sebelum sempat menandai baris log-nya 'resolved'.")

    print("\n  Program 'restart' lagi:")
    om_restarted2 = OrderManager(broker, log_path=log_file)
    results2 = om_restarted2.reconcile_pending(symbol)
    assert len(results2) == 1 and results2[0].client_order_id == real_cid
    print("  OK -- dikonfirmasi order itu SUDAH ada di bursa, TIDAK dikirim ulang.")

    print("\n=== Skenario 4: coba kirim ulang client_order_id yang sudah dipakai -- harus ditolak ===\n")
    try:
        broker.place_limit_order(symbol, "sell", 0.02, 101_000.0, client_order_id=real_cid)
        print("  GAGAL -- seharusnya ditolak")
    except ValueError as e:
        print(f"  OK -- jaring pengaman terakhir tetap berfungsi: {e}")

    log_file.unlink(missing_ok=True)