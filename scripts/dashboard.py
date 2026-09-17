"""
scripts/dashboard.py

Server web KECIL (pakai http.server bawaan Python, TIDAK ada dependency
tambahan) yang nyajiin dashboard HTML buat mantau runner/paper.py secara
live lewat browser.

INI PROSES TERPISAH dari runner/paper.py -- jalankan di jendela terminal
LAIN, BUKAN menggantikan bot-nya. Dashboard ini cuma MEMBACA file yang
ditulis bot (data/session_state.json, data/fills.csv), tidak pernah
mengirim order atau mengubah apa pun.

Cara pakai (lokal, cuma bisa diakses dari komputer/VPS itu sendiri):
    python -m scripts.dashboard

Cara pakai (bisa diakses dari internet, WAJIB pakai password):
    export DASHBOARD_USERNAME="pilih_username"
    export DASHBOARD_PASSWORD="pilih_password_kuat"
    python -m scripts.dashboard --host 0.0.0.0 --port 8000

    Lalu buka firewall VPS untuk port ini, mis. (Ubuntu/ufw):
        sudo ufw allow 8000/tcp

PERINGATAN KEAMANAN: proteksi di sini cuma HTTP Basic Auth TANPA HTTPS --
password dikirim dalam bentuk yang gampang didekode (base64, BUKAN
dienkripsi) kalau ada yang menyadap koneksi di jaringan publik. Ini
cukup untuk menghalangi orang iseng menemukan URL dan langsung masuk
tanpa izin, TAPI TIDAK aman dari penyadapan aktif. Untuk keamanan penuh,
pasang reverse proxy (nginx/Caddy) dengan sertifikat HTTPS (Let's
Encrypt, gratis) di depan server ini -- itu di luar cakupan skrip ini.
"""

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from scripts.session_pnl_report import compute_pnl, load_fills, pair_round_trips

SESSION_STATE_PATH = Path("data/session_state.json")
STARTING_CAPITAL: float | None = None  # diset dari CLI di main(), dibaca build_status_json()

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>Dashboard Bot Trading</title>
<style>
  * { box-sizing: border-box; }
  body {
    background: #0f1117; color: #e6e6e6;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #8b92a5; font-size: 13px; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card {
    background: #1a1d29; border: 1px solid #2a2e3f; border-radius: 10px;
    padding: 16px 18px;
  }
  .card .label { color: #8b92a5; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { font-size: 26px; font-weight: 600; margin-top: 6px; }
  .pos-long { color: #26a69a; }
  .pos-flat { color: #8b92a5; }
  .pnl-pos { color: #26a69a; }
  .pnl-neg { color: #ef5350; }
  .warn { color: #ffa726; }
  .progress-wrap { background: #2a2e3f; border-radius: 8px; height: 10px; overflow: hidden; margin-top: 10px; }
  .progress-bar { background: #5c7cfa; height: 100%; transition: width 0.5s; }
  .progress-bar.entry-zone { background: #26a69a; }
  .progress-bar.buffer-zone { background: #ffa726; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #2a2e3f; }
  th { color: #8b92a5; font-weight: 500; text-transform: uppercase; font-size: 11px; }
  .banner {
    background: #3a2a1a; border: 1px solid #ffa726; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 20px; color: #ffcc80; font-size: 14px; display: none;
  }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .dot-live { background: #26a69a; }
  .dot-stale { background: #ef5350; }
  footer { color: #565b6e; font-size: 12px; margin-top: 24px; }
</style>
</head>
<body>
  <h1 id="title">Dashboard Bot Trading</h1>
  <div class="sub" id="subtitle">Memuat...</div>
  <div class="banner" id="entryBanner"></div>

  <div class="grid">
    <div class="card">
      <div class="label">Posisi Saat Ini</div>
      <div class="value" id="position">--</div>
    </div>
    <div class="card">
      <div class="label">Harga Terakhir</div>
      <div class="value" id="price">--</div>
    </div>
    <div class="card">
      <div class="label">P&amp;L Realisasi</div>
      <div class="value" id="pnl">--</div>
    </div>
    <div class="card">
      <div class="label">Round Trip Selesai</div>
      <div class="value" id="trips">--</div>
    </div>
  </div>

  <div class="grid" id="capitalGrid" style="display:none;">
    <div class="card">
      <div class="label">Modal Awal</div>
      <div class="value" id="startCapital">--</div>
    </div>
    <div class="card">
      <div class="label">Modal Saat Ini (estimasi)</div>
      <div class="value" id="currentCapital">--</div>
      <div style="font-size:11px; color:#565b6e; margin-top:4px;">estimasi dari P&amp;L bot, bukan saldo dompet asli</div>
    </div>
    <div class="card">
      <div class="label">Return</div>
      <div class="value" id="returnPct">--</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:24px;">
    <div class="label">Progres Sesi</div>
    <div id="sessionText" style="margin-top:6px; font-size:14px;">--</div>
    <div class="progress-wrap"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>
  </div>

  <div class="card">
    <div class="label" style="margin-bottom:10px;">Transaksi Terbaru (dari data/fills.csv)</div>
    <table>
      <thead><tr><th>#</th><th>Beli</th><th>Jual</th><th>P&amp;L</th></tr></thead>
      <tbody id="tradesBody"><tr><td colspan="4">Memuat...</td></tr></tbody>
    </table>
  </div>

  <footer><span class="status-dot" id="statusDot"></span><span id="lastUpdate">--</span></footer>

<script>
async function refresh() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    render(data);
    document.getElementById('statusDot').className = 'status-dot dot-live';
  } catch (e) {
    document.getElementById('statusDot').className = 'status-dot dot-stale';
  }
}

function render(d) {
  document.getElementById('title').textContent = `Dashboard -- ${d.symbol || '?'} @ ${d.timeframe || '?'}`;
  document.getElementById('subtitle').textContent = d.session_hours
    ? `Sesi ${d.session_hours} jam, buffer entry ${d.min_entry_buffer_hours ?? 0} jam`
    : 'Tanpa batas waktu sesi';

  const posEl = document.getElementById('position');
  const isLong = d.current_position === 1;
  posEl.textContent = isLong ? 'LONG' : 'FLAT';
  posEl.className = 'value ' + (isLong ? 'pos-long' : 'pos-flat');

  document.getElementById('price').textContent = d.latest_price ? d.latest_price.toFixed(2) : '--';

  const pnlEl = document.getElementById('pnl');
  if (d.total_pnl !== null) {
    pnlEl.textContent = (d.total_pnl >= 0 ? '+' : '') + d.total_pnl.toFixed(4);
    pnlEl.className = 'value ' + (d.total_pnl >= 0 ? 'pnl-pos' : 'pnl-neg');
  } else {
    pnlEl.textContent = '0.0000';
    pnlEl.className = 'value pos-flat';
  }

  document.getElementById('trips').textContent = d.completed_trips ?? 0;

  // Modal awal/sekarang -- cuma tampil kalau --starting-capital diisi
  if (d.starting_capital !== null && d.starting_capital !== undefined) {
    document.getElementById('capitalGrid').style.display = 'grid';
    document.getElementById('startCapital').textContent = d.starting_capital.toLocaleString('id-ID');
    document.getElementById('currentCapital').textContent = d.current_capital_estimate.toLocaleString('id-ID', {maximumFractionDigits: 4});
    const retPct = (d.total_pnl / d.starting_capital) * 100;
    const retEl = document.getElementById('returnPct');
    retEl.textContent = (retPct >= 0 ? '+' : '') + retPct.toFixed(3) + '%';
    retEl.className = 'value ' + (retPct >= 0 ? 'pnl-pos' : 'pnl-neg');
  }

  // Progres sesi
  if (d.session_hours) {
    const pct = Math.min(100, Math.max(0, (d.elapsed_hours / d.session_hours) * 100));
    const bar = document.getElementById('progressBar');
    bar.style.width = pct + '%';

    const entryStillOpen = d.remaining_hours > (d.min_entry_buffer_hours || 0);
    bar.className = 'progress-bar ' + (entryStillOpen ? 'entry-zone' : 'buffer-zone');

    document.getElementById('sessionText').textContent =
      `${d.elapsed_hours.toFixed(1)} jam berjalan / ${d.session_hours} jam total ` +
      `-- sisa ${d.remaining_hours.toFixed(1)} jam ` +
      (entryStillOpen ? '(entry baru masih diizinkan)' : '(entry baru DITAHAN -- mendekati akhir sesi)');

    const banner = document.getElementById('entryBanner');
    if (!entryStillOpen && isLong) {
      banner.style.display = 'block';
      banner.textContent = 'Mendekati akhir sesi dan posisi masih terbuka -- pantau terus, mungkin perlu keputusan manual.';
    } else {
      banner.style.display = 'none';
    }
  } else {
    document.getElementById('sessionText').textContent = 'Tanpa batas waktu sesi.';
  }

  // Tabel transaksi
  const tbody = document.getElementById('tradesBody');
  if (!d.trades || d.trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4">Belum ada transaksi selesai.</td></tr>';
  } else {
    tbody.innerHTML = d.trades.map((t, i) => `
      <tr>
        <td>${i + 1}</td>
        <td>${t.buy_price.toFixed(2)}</td>
        <td>${t.sell_price.toFixed(2)}</td>
        <td class="${t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(4)}</td>
      </tr>
    `).join('');
  }

  document.getElementById('lastUpdate').textContent =
    'Update terakhir dari bot: ' + (d.last_update ? new Date(d.last_update).toLocaleString('id-ID') : '--');
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def build_status_json(symbol_filter: str | None) -> dict:
    state = {}
    if SESSION_STATE_PATH.exists():
        with open(SESSION_STATE_PATH) as f:
            state = json.load(f)

    symbol = symbol_filter or state.get("symbol")

    elapsed_hours = remaining_hours = None
    if state.get("session_start") and state.get("session_hours"):
        session_start = datetime.fromisoformat(state["session_start"])
        now = datetime.now(timezone.utc)
        elapsed_hours = (now - session_start).total_seconds() / 3600
        remaining_hours = state["session_hours"] - elapsed_hours

    try:
        fills = load_fills(symbol=symbol)
        round_trips, open_buy = pair_round_trips(fills)
        results = compute_pnl(round_trips)
        completed = [r for r in results if r["pnl"] is not None]
        total_pnl = sum(r["pnl"] for r in completed) if completed else 0.0
        trades = [
            {"buy_price": r["buy"]["fill_price"], "sell_price": r["sell"]["fill_price"], "pnl": r["pnl"]}
            for r in completed
        ]
    except FileNotFoundError:
        completed, total_pnl, trades = [], 0.0, []

    return {
        "symbol": symbol,
        "timeframe": state.get("timeframe"),
        "session_hours": state.get("session_hours"),
        "min_entry_buffer_hours": state.get("min_entry_buffer_hours"),
        "elapsed_hours": elapsed_hours,
        "remaining_hours": remaining_hours,
        "current_position": state.get("current_position", 0),
        "latest_price": state.get("latest_price"),
        "last_update": state.get("last_update"),
        "total_pnl": total_pnl,
        "completed_trips": len(completed),
        "trades": list(reversed(trades))[:20],  # terbaru dulu, maks 20 baris
        "starting_capital": STARTING_CAPITAL,
        "current_capital_estimate": (STARTING_CAPITAL + total_pnl) if STARTING_CAPITAL is not None else None,
        # ESTIMASI, bukan saldo dompet sungguhan -- cuma modal_awal (Anda
        # masukkan manual) + P&L transaksi yang TERCATAT BOT INI. Tidak
        # memperhitungkan transaksi manual atau saldo lain di dompet.
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # bungkam log akses default, biar terminal tidak berisik

    def _is_authorized(self) -> bool:
        """
        True kalau TIDAK ada username/password diset (mode lokal santai),
        ATAU kalau header Authorization cocok. WAJIB dicek di SETIAP
        request, bukan cuma halaman utama -- endpoint API juga harus
        terlindungi, bukan cuma halaman HTML-nya.
        """
        expected_user = os.environ.get("DASHBOARD_USERNAME")
        expected_pass = os.environ.get("DASHBOARD_PASSWORD")
        if not expected_user or not expected_pass:
            return True  # tidak diset -- mode lokal, tidak wajib auth

        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Basic "):
            return False

        try:
            decoded = base64.b64decode(auth_header[6:]).decode()
            user, _, pw = decoded.partition(":")
        except Exception:
            return False

        return user == expected_user and pw == expected_pass

    def _request_auth(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Dashboard Bot Trading"')
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Password diperlukan.")

    def do_GET(self):
        if not self._is_authorized():
            self._request_auth()
            return

        if self.path == "/" or self.path == "":
            self._send(200, DASHBOARD_HTML.encode(), "text/html")
        elif self.path.startswith("/api/status"):
            try:
                data = build_status_json(symbol_filter=None)
                self._send(200, json.dumps(data).encode(), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
        else:
            self._send(404, b"Not found", "text/plain")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard live untuk runner/paper.py")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--host",
        default="localhost",
        help="'localhost' (default, cuma bisa diakses dari komputer/VPS ini sendiri) "
        "atau '0.0.0.0' (bisa diakses dari internet -- WAJIB set DASHBOARD_USERNAME "
        "dan DASHBOARD_PASSWORD dulu kalau pakai ini)",
    )
    parser.add_argument(
        "--starting-capital",
        type=float,
        default=None,
        help="kalau diisi (mis. 10000), dashboard menampilkan estimasi modal saat ini "
        "= starting_capital + P&L transaksi bot ini. INI ESTIMASI, bukan saldo dompet "
        "sungguhan -- tidak memperhitungkan transaksi manual atau saldo lain.",
    )
    args = parser.parse_args()

    global STARTING_CAPITAL
    STARTING_CAPITAL = args.starting_capital

    has_auth = bool(os.environ.get("DASHBOARD_USERNAME")) and bool(os.environ.get("DASHBOARD_PASSWORD"))

    if args.host != "localhost" and not has_auth:
        print("!" * 70)
        print("  PERINGATAN: --host bukan 'localhost' (bisa diakses dari internet)")
        print("  TAPI DASHBOARD_USERNAME/DASHBOARD_PASSWORD belum diset.")
        print("  Dashboard ini akan TERBUKA TANPA PASSWORD untuk siapa pun yang tau URL-nya.")
        print("  Set dulu env var itu sebelum lanjut, atau tekan Ctrl+C untuk batal.")
        print("!" * 70)

    print(f"Dashboard jalan di http://{args.host}:{args.port}  (Ctrl+C untuk berhenti)")
    if has_auth:
        print(f"  Password AKTIF (username: {os.environ['DASHBOARD_USERNAME']})")
    print("Proses ini CUMA membaca data/session_state.json + data/fills.csv --")
    print("tidak pernah mengirim order. Jalankan runner/paper.py di jendela terpisah.")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
