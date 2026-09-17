"""
scripts/dashboard_donchian.py

Dashboard pemantau bot Donchian -- BERDIRI SENDIRI, PROSES TERPISAH.

=========================================================================
JANJI PENTING: file ini TIDAK MENGUBAH file bot manapun, dan TIDAK PERNAH
mengirim order apa pun. Semua operasi ke bursa di sini READ-ONLY
(fetch_balance, fetch_positions, fetch_my_trades). Aman dijalankan
BERSAMAAN dengan bot yang sedang trading -- tidak akan mengganggunya.
=========================================================================

KENAPA DATANYA DARI BURSA, BUKAN DARI FILE BOT
-------------------------------------------------------------------------
Bot kita (runner/paper.py) cuma menulis SATU file: data/session_state.json,
isinya posisi/sinyal/harga terakhir -- TIDAK ADA riwayat trade, P&L, atau
fee di situ. Dashboard rujukan yang Anda lampirkan membaca file-file yang
BOT MEREKA tulis sendiri (status/*.json, *.jsonl riwayat trade).

Normalnya solusinya: tambah penulisan log di paper.py. TAPI Anda melarang
mengubah file bot -- jadi dashboard ini mengambil riwayat trade LANGSUNG
DARI BINANCE. Untuk kebutuhan "bersih = total dikurangi fee" yang Anda
minta, ini justru LEBIH BAIK: angka P&L dan fee-nya SUNGGUHAN dari bursa,
bukan estimasi/hitungan ulang kita sendiri yang bisa meleset.

CATATAN ARSITEKTUR (saya sampaikan terbuka, bukan disembunyikan)
-------------------------------------------------------------------------
Proyek ini punya aturan "cuma execution/broker.py yang boleh menyentuh
ccxt". File ini MELANGGAR aturan itu sedikit: memanggil
broker.exchange.fetch_my_trades() langsung, karena Broker tidak punya
method untuk riwayat trade dan saya TIDAK BOLEH menambahkannya (larangan
Anda). Pilihannya cuma: langgar aturan arsitektur di file dashboard
read-only yang terpisah, ATAU ubah broker.py. Saya pilih yang pertama
karena risikonya jauh lebih kecil terhadap bot yang sedang jalan --
tapi Anda perlu tahu trade-off ini ada.

Cara pakai:
    python -m scripts.dashboard_donchian --symbol "BTC/USDT:USDT"
lalu buka http://localhost:8100 di browser.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SESSION_STATE_PATH = Path("data/session_state.json")


def aggregate_trades(raw_trades: list[dict]) -> dict:
    """
    Ringkas riwayat trade mentah dari ccxt jadi angka siap tampil.
    FUNGSI MURNI (tidak menyentuh jaringan) -- supaya bisa diuji
    tuntas tanpa koneksi bursa sungguhan.

    raw_trades: hasil exchange.fetch_my_trades() -- tiap elemen punya
        'fee' (unified ccxt: {'cost': ..}) dan 'info' (respons MENTAH
        Binance, berisi 'realizedPnl' dan 'commission').

    KUNCI dari permintaan Nero: 'bersih' = total realized P&L DIKURANGI
    total fee. Binance melaporkan realizedPnl TERPISAH dari commission
    (fee TIDAK ikut dipotong di angka realizedPnl), jadi pengurangan ini
    memang perlu dilakukan sendiri -- bukan dobel-potong.

    Fee dibaca dari fee.cost (unified) dulu; kalau kosong, fallback ke
    info.commission (mentah). Kalau DUA-DUANYA kosong, trade itu dihitung
    fee=0 TAPI dicatat di 'n_fee_unknown' -- supaya ketahuan kalau angka
    'bersih' ternyata tidak bisa dipercaya, bukan diam-diam dianggap nol.
    """
    total_realized = 0.0
    total_fee = 0.0
    n_fee_unknown = 0
    wins: list[float] = []
    losses: list[float] = []

    for t in raw_trades:
        info = t.get("info") or {}

        raw_pnl = info.get("realizedPnl")
        pnl = float(raw_pnl) if raw_pnl not in (None, "") else 0.0
        total_realized += pnl

        fee_obj = t.get("fee") or {}
        fee_cost = fee_obj.get("cost")
        if fee_cost in (None, ""):
            fee_cost = info.get("commission")
        if fee_cost in (None, ""):
            n_fee_unknown += 1
            fee_cost = 0.0
        total_fee += float(fee_cost)

        # Cuma trade yang BENAR-BENAR merealisasikan sesuatu yang dihitung
        # menang/kalah -- trade PEMBUKA posisi realizedPnl-nya 0, itu bukan
        # "seri", itu memang belum ada yang direalisasikan.
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(pnl)

    n_closing = len(wins) + len(losses)
    return {
        "n_fills": len(raw_trades),
        "n_closing_trades": n_closing,
        "total_realized": total_realized,
        "total_fee": total_fee,
        "net": total_realized - total_fee,
        "total_profit": sum(wins),
        "total_loss": sum(losses),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": (len(wins) / n_closing) if n_closing else 0.0,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "best_win": max(wins) if wins else 0.0,
        "worst_loss": min(losses) if losses else 0.0,
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None,
        "n_fee_unknown": n_fee_unknown,
    }


def read_session_state() -> dict | None:
    """Baca data/session_state.json (ditulis bot). Return None kalau tidak ada/rusak."""
    try:
        if not SESSION_STATE_PATH.exists():
            return None
        with open(SESSION_STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def session_state_age_seconds() -> float | None:
    """Umur file session_state.json dalam detik -- untuk deteksi bot mati/hidup."""
    try:
        return time.time() - SESSION_STATE_PATH.stat().st_mtime
    except Exception:
        return None


POSITION_LABEL = {1: "LONG", 0: "KOSONG", -1: "SHORT"}


class DashboardState:
    """
    Menyimpan snapshot terakhir dari bursa. Di-refresh oleh thread
    terpisah supaya permintaan HTTP tidak pernah menunggu jaringan
    (kalau bursa lambat, halaman tetap muncul dengan data terakhir,
    bukan menggantung).
    """

    def __init__(self, symbol: str, refresh_seconds: float):
        self.symbol = symbol
        self.refresh_seconds = refresh_seconds
        self.lock = threading.Lock()
        self.snapshot: dict = {"status": "memuat", "last_error": None, "last_update": None}
        self._broker = None

    def _ensure_broker(self):
        if self._broker is None:
            from src.execution.broker import Broker
            self._broker = Broker(exchange_id="binanceusdm", testnet=True)
        return self._broker

    def refresh_once(self) -> None:
        try:
            broker = self._ensure_broker()

            balance = broker.fetch_balance()
            usdt = balance.get("USDT", {})
            wallet = float(usdt.get("total") or 0.0)
            free = float(usdt.get("free") or 0.0)

            position = broker.fetch_position(self.symbol)
            price = broker.fetch_current_price(self.symbol)

            # Riwayat trade LANGSUNG dari bursa -- lihat catatan arsitektur
            # di kepala file soal kenapa ini memanggil exchange langsung.
            raw_trades = broker.exchange.fetch_my_trades(self.symbol, limit=200)
            agg = aggregate_trades(raw_trades)

            recent = []
            for t in reversed(raw_trades[-25:]):
                info = t.get("info") or {}
                fee_obj = t.get("fee") or {}
                fee_cost = fee_obj.get("cost")
                if fee_cost in (None, ""):
                    fee_cost = info.get("commission", 0)
                recent.append({
                    "time": t.get("datetime", ""),
                    "side": t.get("side", ""),
                    "price": float(t.get("price") or 0),
                    "amount": float(t.get("amount") or 0),
                    "fee": float(fee_cost or 0),
                    "pnl": float(info.get("realizedPnl") or 0),
                    "role": "maker" if info.get("maker") in (True, "true") else "taker",
                })

            with self.lock:
                self.snapshot = {
                    "status": "ok",
                    "last_error": None,
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "symbol": self.symbol,
                    "wallet": wallet,
                    "free": free,
                    "price": price,
                    "position": position,
                    "agg": agg,
                    "recent": recent,
                }
        except Exception as e:
            with self.lock:
                self.snapshot = {
                    **self.snapshot,
                    "status": "galat",
                    "last_error": f"{type(e).__name__}: {e}",
                    "last_update": datetime.now(timezone.utc).isoformat(),
                }

    def loop(self) -> None:
        while True:
            self.refresh_once()
            time.sleep(self.refresh_seconds)

    def payload(self) -> dict:
        with self.lock:
            snap = dict(self.snapshot)
        snap["session_state"] = read_session_state()
        snap["session_state_age"] = session_state_age_seconds()
        return snap


HTML_PAGE = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pemantau Bot Donchian</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
    --hijau: #15803d; --merah: #b91c1c; --kuning: #a16207; --panel: #fafafa;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0f1115; --fg: #e8e8e8; --muted: #9ca3af; --line: #2a2f3a;
      --hijau: #4ade80; --merah: #f87171; --kuning: #fbbf24; --panel: #161a21;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 26px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 4px; }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .06em;
       color: var(--muted); font-weight: 600; margin: 32px 0 12px; }
  .panel { border: 1px solid var(--line); border-radius: 12px; padding: 18px; background: var(--panel); }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
  .card { border: 1px solid var(--line); border-radius: 10px; padding: 16px; background: var(--bg); }
  .card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
  .card .big { font-size: 26px; font-weight: 600; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .card .note { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .card.hijau { border-color: var(--hijau); }
  .card.merah { border-color: var(--merah); }
  .card.tebal { border-width: 2px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 18px 14px; }
  .item .label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
  .item .val { font-size: 19px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .hijau-t { color: var(--hijau); } .merah-t { color: var(--merah); } .kuning-t { color: var(--kuning); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
       color: var(--muted); font-weight: 600; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  td { padding: 9px 10px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom: none; }
  .scroll { max-height: 380px; overflow-y: auto; }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
  .warn { border-left: 3px solid var(--kuning); padding: 10px 14px; background: var(--panel);
          border-radius: 6px; font-size: 13px; margin-bottom: 14px; }
  .err { border-left: 3px solid var(--merah); padding: 10px 14px; background: var(--panel);
         border-radius: 6px; font-size: 13px; margin-bottom: 14px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Pemantau Bot Donchian</h1>
  <div class="sub">Donchian close-only &middot; data diambil LANGSUNG dari Binance (read-only, tidak mengganggu bot)</div>
  <div class="sub mono" id="jam">&mdash;</div>
  <div id="isi"></div>
</div>
<script>
const uang = (n, d = 2) => (n >= 0 ? "+" : "") + n.toFixed(d);
const warna = (n) => n > 0 ? "hijau-t" : (n < 0 ? "merah-t" : "");

function render(d) {
  document.getElementById("jam").textContent =
    "diperbarui " + (d.last_update ? new Date(d.last_update).toLocaleTimeString("id-ID") : "?");

  let h = "";

  if (d.status === "galat") {
    h += `<div class="err"><b>Gagal ambil data dari bursa:</b> <span class="mono">${d.last_error || "?"}</span><br>
          Angka di bawah adalah snapshot TERAKHIR yang berhasil, bukan kondisi sekarang.</div>`;
  }
  if (d.agg && d.agg.n_fee_unknown > 0) {
    h += `<div class="warn"><b>Perhatian:</b> ${d.agg.n_fee_unknown} dari ${d.agg.n_fills} fill tidak punya data fee
          dari bursa &mdash; angka "BERSIH" di bawah kemungkinan LEBIH OPTIMIS dari kenyataan.</div>`;
  }

  const ss = d.session_state;
  const umur = d.session_state_age;
  const botHidup = umur !== null && umur < 180;
  h += `<h2>Bot</h2><div class="panel"><table>
    <tr><th>Simbol</th><th>Status bot</th><th>Posisi (bursa)</th><th>Sinyal (bot)</th><th>Harga kini</th><th>Saldo dompet</th></tr>
    <tr>
      <td class="mono">${d.symbol || "?"}</td>
      <td class="${botHidup ? "hijau-t" : "merah-t"}">${
        umur === null ? "tidak terdeteksi" : (botHidup ? "jalan" : `diam (${Math.round(umur)} dtk)`)}</td>
      <td class="${d.position ? (d.position.side === "long" ? "hijau-t" : "merah-t") : ""}">${
        d.position ? d.position.side.toUpperCase() + " " + d.position.contracts : "KOSONG"}</td>
      <td>${ss ? ({"1":"LONG","0":"KOSONG","-1":"SHORT"}[String(ss.latest_signal)] || ss.latest_signal) : "?"}</td>
      <td>${d.price ? d.price.toFixed(2) : "?"}</td>
      <td>${d.wallet !== undefined ? d.wallet.toFixed(2) : "?"}</td>
    </tr></table></div>`;

  if (d.agg) {
    const a = d.agg;
    h += `<h2>Untung dan rugi &mdash; dari riwayat bursa</h2><div class="cards">
      <div class="card hijau"><div class="label">Total untung &middot; ${a.n_wins} transaksi</div>
        <div class="big hijau-t">${uang(a.total_profit)} USDT</div>
        <div class="note">belum dikurangi fee</div></div>
      <div class="card merah"><div class="label">Total rugi &middot; ${a.n_losses} transaksi</div>
        <div class="big merah-t">${uang(a.total_loss)} USDT</div>
        <div class="note">belum dikurangi fee</div></div>
      <div class="card tebal ${a.net >= 0 ? "hijau" : "merah"}"><div class="label">Bersih &mdash; sudah dikurangi fee</div>
        <div class="big ${warna(a.net)}">${uang(a.net)} USDT</div>
        <div class="note">realisasi ${uang(a.total_realized)} &minus; fee ${a.total_fee.toFixed(4)}</div></div>
    </div>`;

    h += `<h2>Rincian</h2><div class="panel"><div class="grid">
      <div class="item"><div class="label">Transaksi penutup</div><div class="val">${a.n_closing_trades}</div></div>
      <div class="item"><div class="label">Menang / kalah</div><div class="val">${a.n_wins} / ${a.n_losses}</div></div>
      <div class="item"><div class="label">Tingkat menang</div><div class="val">${(a.win_rate*100).toFixed(1)}%</div></div>
      <div class="item"><div class="label">Rata menang</div><div class="val hijau-t">${uang(a.avg_win, 4)}</div></div>
      <div class="item"><div class="label">Rata kalah</div><div class="val merah-t">${uang(a.avg_loss, 4)}</div></div>
      <div class="item"><div class="label">Menang terbaik</div><div class="val hijau-t">${uang(a.best_win, 4)}</div></div>
      <div class="item"><div class="label">Kalah terburuk</div><div class="val merah-t">${uang(a.worst_loss, 4)}</div></div>
      <div class="item"><div class="label">Rasio untung/rugi</div><div class="val">${
        a.profit_factor === null ? "&mdash;" : a.profit_factor.toFixed(2)}</div></div>
      <div class="item"><div class="label">Total fee dibayar</div><div class="val merah-t">${a.total_fee.toFixed(4)}</div></div>
      <div class="item"><div class="label">Jumlah fill</div><div class="val">${a.n_fills}</div></div>
      <div class="item"><div class="label">Harga masuk posisi</div><div class="val">${
        d.position && d.position.entry_price ? d.position.entry_price.toFixed(2) : "&mdash;"}</div></div>
      <div class="item"><div class="label">Leverage</div><div class="val">${
        d.position && d.position.leverage ? d.position.leverage.toFixed(0) + "x" : "&mdash;"}</div></div>
    </div></div>`;
  }

  if (d.recent && d.recent.length) {
    h += `<h2>Riwayat transaksi terbaru &mdash; ${d.recent.length} fill</h2>
      <div class="panel scroll"><table>
      <tr><th>Waktu</th><th>Sisi</th><th>Harga</th><th>Jumlah</th><th>Fee</th><th>Realisasi</th><th>Peran</th></tr>`;
    for (const t of d.recent) {
      h += `<tr>
        <td class="mono">${t.time ? new Date(t.time).toLocaleString("id-ID") : "?"}</td>
        <td class="${t.side === "buy" ? "hijau-t" : "merah-t"}">${t.side.toUpperCase()}</td>
        <td>${t.price.toFixed(2)}</td><td>${t.amount}</td>
        <td class="merah-t">${t.fee.toFixed(5)}</td>
        <td class="${warna(t.pnl)}">${t.pnl === 0 ? "&mdash;" : uang(t.pnl, 5)}</td>
        <td class="mono">${t.role}</td></tr>`;
    }
    h += `</table></div>`;
  }

  document.getElementById("isi").innerHTML = h;
}

async function muat() {
  try {
    const r = await fetch("/api");
    render(await r.json());
  } catch (e) {
    document.getElementById("jam").textContent = "gagal menghubungi dashboard: " + e;
  }
}
muat();
setInterval(muat, 5000);
</script>
</body>
</html>
"""


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/api"):
                body = json.dumps(state.payload()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = HTML_PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *args):  # matikan log HTTP -- bikin ramai terminal
            pass

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--port", type=int, default=8100,
                    help="default 8100 -- SENGAJA beda dari 8000 yang dipakai dashboard lama Anda di VPS")
    p.add_argument("--refresh-seconds", type=float, default=5.0)
    args = p.parse_args()

    state = DashboardState(args.symbol, args.refresh_seconds)
    threading.Thread(target=state.loop, daemon=True).start()

    print(f"=== Dashboard Donchian (READ-ONLY, tidak mengirim order apa pun) ===")
    print(f"  Simbol  : {args.symbol}")
    print(f"  Refresh : tiap {args.refresh_seconds:.0f} detik")
    print(f"  Buka    : http://localhost:{args.port}")
    print(f"  Hentikan dengan Ctrl+C\n")

    HTTPServer(("0.0.0.0", args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()