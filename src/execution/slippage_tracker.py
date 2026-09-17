"""
Ukur selisih harga sinyal vs harga fill. Bahan koreksi cost_model.

Sprint 1.
"""

# TODO Sprint 1
"""
execution/slippage_tracker.py

Catat setiap fill: waktu sinyal muncul, harga saat sinyal muncul,
waktu fill sungguhan, harga fill sungguhan, fee -- dan hitung slippage
SUNGGUHAN (bukan tebakan) dari selisihnya.

INI YANG MENGISI ANGKA slippage_bps ASLI ke src/core/cost_model.py.
Ingat catatan di cost_model.py sejak Hari 1:

    slippage_bps: float = 2.0
    slippage_is_measured: bool = False  # jadi True setelah diukur dari fill nyata

File ini adalah "fill nyata" yang dimaksud catatan itu. Setelah cukup
banyak data terkumpul lewat runner/paper.py, hitung mean_slippage_bps
dari summarize() di bawah, GANTI angka 2.0 bps yang masih tebakan itu
dengan angka terukur ini, lalu set slippage_is_measured=True.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.execution.broker import OrderResult

DEFAULT_LOG_PATH = Path("data/fills.csv")

_FIELDNAMES = [
    "signal_time",
    "signal_price",
    "fill_time",
    "fill_price",
    "symbol",
    "side",
    "amount_filled",
    "fee",
    "slippage_bps",
    "client_order_id",
]


@dataclass
class SignalContext:
    """
    Info yang diketahui strategi SAAT sinyal muncul -- direkam SEBELUM
    order dikirim, supaya bisa dibandingkan dengan harga fill
    sungguhan setelah order selesai. runner/paper.py yang bertanggung
    jawab membuat objek ini tepat saat generate_signals() menghasilkan
    sinyal baru, sebelum diteruskan ke order_manager.py.
    """

    signal_time: datetime
    signal_price: float


def compute_slippage_bps(signal_price: float, fill_price: float, side: str) -> float:
    """
    Slippage dalam bps. TANDA-nya penting dan disengaja:
        positif = harga fill LEBIH BURUK dari harga sinyal (rugi ekstra)
        negatif = harga fill LEBIH BAIK dari harga sinyal (untung ekstra,
                  jarang tapi bisa terjadi kalau limit order keisi di
                  harga lebih menguntungkan dari yang diharapkan)

    Untuk BUY: fill LEBIH MAHAL dari sinyal = buruk -> positif.
    Untuk SELL: fill LEBIH MURAH dari sinyal = buruk -> positif.
    """
    if signal_price <= 0:
        return 0.0
    raw_bps = (fill_price - signal_price) / signal_price * 10_000
    return raw_bps if side == "buy" else -raw_bps


class SlippageTracker:
    def __init__(self, log_path: Path = DEFAULT_LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()

    def record_fill(self, signal: SignalContext, result: OrderResult) -> float:
        """
        Catat satu fill, kembalikan slippage_bps yang dihitung.

        Dipanggil SETELAH order_manager.py memberi OrderResult yang
        sudah terisi (status filled ATAU partially_filled). Kalau
        order terisi bertahap (beberapa kali partial fill), panggil
        ini SETIAP kali status berubah -- bukan cuma sekali di akhir --
        supaya harga rata-rata tiap tahap fill tetap tercatat presisi.

        Pakai result.average_price (harga eksekusi SUNGGUHAN), BUKAN
        result.price (harga limit yang DIMINTA) -- keduanya bisa beda,
        dan selisih itu sendiri bagian dari slippage yang mau diukur.
        """
        fill_price = result.average_price if result.average_price is not None else result.price
        slippage_bps = compute_slippage_bps(signal.signal_price, fill_price, result.side)

        row = {
            "signal_time": signal.signal_time.isoformat(),
            "signal_price": signal.signal_price,
            "fill_time": datetime.now(timezone.utc).isoformat(),
            "fill_price": fill_price,
            "symbol": result.symbol,
            "side": result.side,
            "amount_filled": result.filled_amount,
            "fee": result.fee,
            "slippage_bps": round(slippage_bps, 4),
            "client_order_id": result.client_order_id,
        }
        with open(self.log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writerow(row)

        return slippage_bps

    def summarize(self) -> dict:
        """
        Ringkas semua fill yang tercatat. mean_slippage_bps di sini
        yang dipakai untuk mengisi CostConfig.slippage_bps di
        src/core/cost_model.py, menggantikan asumsi 2.0 bps Hari 1.
        """
        if not self.log_path.exists():
            return {"n_fills": 0}

        slippages, fees = [], []
        with open(self.log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                slippages.append(float(row["slippage_bps"]))
                fees.append(float(row["fee"]))

        if not slippages:
            return {"n_fills": 0}

        return {
            "n_fills": len(slippages),
            "mean_slippage_bps": sum(slippages) / len(slippages),
            "max_slippage_bps": max(slippages),
            "min_slippage_bps": min(slippages),
            "total_fee": sum(fees),
        }


if __name__ == "__main__":
    # Uji asap TANPA jaringan -- pakai MockBroker, simulasikan beberapa
    # fill dengan slippage berbeda-beda dan pastikan tanda +/- serta
    # perhitungannya benar untuk buy maupun sell.
    import tempfile
    from datetime import timedelta

    from src.execution.broker import MockBroker

    log_file = Path(tempfile.mktemp(suffix=".csv"))
    tracker = SlippageTracker(log_path=log_file)

    print("=== Uji asap SlippageTracker (tanpa jaringan) ===\n")

    broker = MockBroker(fill_immediately=True)
    scenarios = [
        ("buy", 100_000.0, 100_015.0),  # fill lebih mahal dari sinyal -- buruk untuk buy
        ("sell", 100_000.0, 99_990.0),  # fill lebih murah dari sinyal -- buruk untuk sell
        ("buy", 100_000.0, 99_998.0),  # fill lebih murah dari sinyal -- bagus untuk buy
    ]

    for side, signal_price, fill_price in scenarios:
        result = broker.place_limit_order("BTC/USDT:USDT", side, 0.01, fill_price)
        signal = SignalContext(
            signal_time=datetime.now(timezone.utc) - timedelta(seconds=2), signal_price=signal_price
        )
        slip = tracker.record_fill(signal, result)
        arah = "BURUK" if slip > 0 else "BAGUS" if slip < 0 else "NETRAL"
        print(f"  {side:4s} sinyal@{signal_price:,.1f} fill@{fill_price:,.1f} -> {slip:+.2f} bps ({arah})")

    print("\n=== Ringkasan ===\n")
    summary = tracker.summarize()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nCatatan: begitu cukup banyak fill sungguhan terkumpul lewat")
    print("runner/paper.py, pakai mean_slippage_bps di atas untuk mengganti")
    print("CostConfig.slippage_bps di src/core/cost_model.py -- lalu set")
    print("slippage_is_measured=True.")

    log_file.unlink(missing_ok=True)