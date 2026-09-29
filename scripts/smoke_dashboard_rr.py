"""
scripts/smoke_dashboard_rr.py

Uji asap perubahan form dashboard ke Risk : Reward.

    python -m scripts.smoke_dashboard_rr
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import dashboard_donchian as dash  # noqa: E402
from src.runner.paper import compute_atr, compute_bracket  # noqa: E402

FAILURES: list[str] = []
NODE = shutil.which("node")


def check(name, cond, detail=""):
    print(f"  [{'LULUS' if cond else 'GAGAL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def script_block() -> str:
    return dash.HTML_PAGE.split("<script>")[1].split("</script>")[0]


def js_function(name: str) -> str:
    """Ambil teks satu fungsi JS dari halaman (sampai kurung kurawal penutupnya)."""
    src = script_block()
    start = src.index(f"function {name}(")
    depth, i = 0, src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1


def run_node(code: str):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(code)
    out = subprocess.run([NODE, f.name], capture_output=True, text=True, timeout=20)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def test_command():
    print("== 1. Perintah dari form diterima launcher baru ==")
    cfg = {"symbol": "BTC/USDT:USDT", "timeframe": "1m", "lookback": 200, "amount": 0.01,
           "session_hours": 5, "risk_reward": 2, "sl_atr_mult": 2, "stop_after_take_profit": True,
           "backfill_bars": 200}
    cmd = dash.build_bot_command(cfg)
    s = " ".join(cmd)
    check("--risk-reward 2.0 dan --sl-atr-mult 2.0 dikirim", "--risk-reward 2.0" in s and "--sl-atr-mult 2.0" in s, s)
    check("--take-profit-pct TIDAK dikirim (launcher menolak kombinasinya)", "--take-profit-pct" not in s)
    check("--live-take-profit-poll-seconds tidak dikirim", "--live-take-profit-poll-seconds" not in s)

    from scripts import run_paper_donchian_futures as L
    argv = [a for a in cmd[cmd.index("scripts.run_paper_donchian_futures") + 1:]]
    argv = ["--mock" if a == "--live" else a for a in argv]
    asli = sys.argv
    try:
        sys.argv = ["run_paper_donchian_futures"] + argv
        buf = io.StringIO()
        with redirect_stdout(buf):
            L.main()
        check("launcher MENERIMA perintah dari dashboard (diuji di mode --mock)",
              "SL + TP rasio 2:1" in buf.getvalue(), buf.getvalue()[-200:])
    except SystemExit as e:
        check("launcher MENERIMA perintah dari dashboard (diuji di mode --mock)", False, f"ditolak: {e}")
    finally:
        sys.argv = asli

    kosong = " ".join(dash.build_bot_command({**cfg, "risk_reward": None}))
    check("R:R dikosongkan -> tidak ada flag SL/TP sama sekali", "--risk-reward" not in kosong and "--sl-atr-mult" not in kosong)
    lama = " ".join(dash.build_bot_command({**cfg, "risk_reward": None, "take_profit_pct": 0.003}))
    check("jalur TP lama masih didukung kalau dikirim eksplisit", "--take-profit-pct 0.003" in lama)


def test_html():
    print("\n== 2. Form dan skrip halaman ==")
    h = dash.HTML_PAGE
    check("field Untung : Rugi & Jarak SL ada", 'id="c_rr"' in h and 'id="c_slatr"' in h)
    check("field Take profit & Poll TP lama sudah dihapus", 'id="c_tp"' not in h and 'id="c_poll"' not in h)
    check("konfig() mengirim risk_reward, tidak take_profit_pct",
          "risk_reward:" in js_function("konfig") and "take_profit_pct" not in js_function("konfig"))
    check("batas fee JS diisi dari konstanta Python", f"const SL_MIN_FEE_MULT_JS = {float(dash.SL_MIN_FEE_MULT)!r};" in h)
    if not NODE:
        check("node tersedia untuk uji JavaScript", False, "pasang nodejs untuk menjalankan uji 2-4")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(script_block())
    r = subprocess.run([NODE, "--check", f.name], capture_output=True, text=True)
    check("sintaks JavaScript halaman valid", r.returncode == 0, r.stderr[:200])


def test_preview_matches_bot():
    print("\n== 3. Pratinjau di layar = angka yang dipakai bot ==")
    if not NODE:
        return
    kasus = [(85000.0, 0.01, 40.0, 0.0005, 2.0, 2.0),    # 1m: ATR kecil -> batas bawah fee
             (85000.0, 0.01, 400.0, 0.0005, 2.0, 2.0),   # ATR besar -> ATR yang menentukan
             (60000.0, 0.02, None, 0.0004, 3.0, 1.5),    # ATR belum ada
             (85000.0, 0.01, 212.5, 0.0005, 2.0, 2.0)]   # tepat di perbatasan
    js = js_function("pratinjauRR") + "\nconsole.log(JSON.stringify([" + ",".join(
        f"pratinjauRR({p},{q},{'null' if a is None else a},{fee},{rr},{k},{dash.SL_MIN_FEE_MULT})"
        for p, q, a, fee, rr, k in kasus) + "]));"
    hasil = run_node(js)
    for (p, q, a, fee, rr, k), j in zip(kasus, hasil):
        b = compute_bracket(p, 1, a, fee, rr, k, dash.SL_MIN_FEE_MULT)
        rugi, untung = p * q * (b["sl_dist"] + b["fee_rt"]), p * q * (b["tp_dist"] - b["fee_rt"])
        sama = all(abs(x - y) < 1e-9 for x, y in
                   [(j["s"], b["sl_dist"]), (j["t"], b["tp_dist"]), (j["rugi"], rugi), (j["untung"], untung)])
        check(f"harga {p:.0f}, ATR {a}, R {rr}: SL {j['s']:.3%} TP {j['t']:.3%} "
              f"rugi {j['rugi']:.2f} untung {j['untung']:.2f}", sama)
        check(f"   ... rasio untung/rugi bersih = {rr}:1", abs(j["untung"] / j["rugi"] - rr) < 1e-9)


def test_panel():
    print("\n== 4. Panel SL/TP di bursa ==")
    if not NODE:
        return
    stub = "const pilih = () => null;\n" + js_function("panelSlTp") + "\n"
    pos = {"entry_price": 85000.0}
    kasus = {
        "sl_hilang": ({"position": pos, "bracket_orders": [{"type": "TAKE_PROFIT_MARKET", "trigger_price": 85600}]}, True),
        "lengkap": ({"position": pos, "bracket_orders": [
            {"type": "STOP_MARKET", "trigger_price": 84830}, {"type": "TAKE_PROFIT_MARKET", "trigger_price": 85595}]}, True),
        "tak_terbaca": ({"position": pos, "bracket_orders": None}, True),
        "mode_lama": ({"position": pos, "bracket_orders": []}, False),
    }
    js = stub + "console.log(JSON.stringify({" + ",".join(
        f"{k}: panelSlTp({json.dumps(d)}, {'true' if m else 'false'})" for k, (d, m) in kasus.items()) + "}));"
    out = run_node(js)
    check("mode SL/TP + SL tidak ada -> PERINGATAN merah", "TIDAK dilindungi stop loss" in out["sl_hilang"])
    check("SL & TP ada -> harga pemicu tampil", "84830.00" in out["lengkap"] and "85595.00" in out["lengkap"])
    check("... dengan jarak dari harga masuk", "0.20% dari harga masuk" in out["lengkap"], out["lengkap"][:300])
    check("tidak ada peringatan palsu kalau SL ada", "TIDAK dilindungi" not in out["lengkap"])
    check("gagal dibaca -> dibilang tidak terbaca, bukan 'tidak ada'", "tidak bisa dibaca" in out["tak_terbaca"])
    check("mode lama tanpa SL/TP -> tidak ada peringatan", "TIDAK dilindungi" not in out["mode_lama"])

    ekspr = r'(b.cmd.match(/--risk-reward (\S+)/) || [,"?"])[1].replace(/\.0$/, "")'
    ada = re.search(re.escape("--risk-reward (\\S+)"), dash.HTML_PAGE)
    check("label status membaca rasio dari perintah bot", ada is not None)
    r = run_node("const b={cmd:'python -m x --live --risk-reward 2.0 --sl-atr-mult 2.0'};"
                 f"console.log(JSON.stringify({ekspr}));")
    check("... hasilnya '2' (tampil sebagai SL/TP 2:1)", r == "2", repr(r))


class FakeBroker:
    def __init__(self, cond=True, cond_error=False):
        self.exchange = type("E", (), {"fetch_my_trades": lambda *a, **k: [],
                                        "fetch_positions": lambda *a, **k: []})()
        self.cond_error = cond_error
        if not cond:
            self.fetch_open_conditional_orders = None
        self.bars = [{"timestamp": i, "open": 85000, "high": 85000 + (i % 5) * 10 + 20,
                      "low": 85000 - 20, "close": 85000 + (i % 3), "volume": 1} for i in range(201)]

    def fetch_balance(self):
        return {"USDT": {"total": 5000, "free": 4900, "used": 100}}

    def fetch_position(self, symbol):
        return {"side": "long", "contracts": 0.01, "leverage": 20, "entry_price": 85000}

    def fetch_current_price(self, symbol):
        return 85100.0

    def fetch_recent_bars(self, symbol, timeframe, limit=100):
        return self.bars[-limit:]

    def fetch_trading_fee_pct(self, symbol):
        return 0.0004

    def fetch_open_conditional_orders(self, symbol):
        if self.cond_error:
            raise ConnectionError("bursa sibuk")
        return [{"id": "1", "type": "STOP_MARKET", "side": "sell", "trigger_price": 84830.0}]


def test_refresh():
    print("\n== 5. Data yang dikirim dashboard ke halaman ==")
    st = dash.DashboardState("BTC/USDT:USDT", 5, 1, 200, "1m")
    st._broker = FakeBroker()
    st.refresh_once()
    snap = st.snapshot
    check("refresh tetap sukses", snap.get("status") == "ok", snap.get("last_error"))
    check("ATR sama persis dengan rumus bot", abs(snap["atr"] - compute_atr(st._broker.bars, 20)) < 1e-12)
    check("fee taker akun ikut dikirim", snap["fee_taker"] == 0.0004)
    check("SL/TP di bursa ikut dikirim", snap["bracket_orders"][0]["trigger_price"] == 84830.0)

    st2 = dash.DashboardState("BTC/USDT:USDT", 5, 1, 200, "1m")
    st2._broker = FakeBroker(cond_error=True)
    st2.refresh_once()
    check("gagal baca SL/TP -> None, refresh lain tetap jalan",
          st2.snapshot["status"] == "ok" and st2.snapshot["bracket_orders"] is None)

    class BrokerLama(FakeBroker):
        fetch_open_conditional_orders = property(lambda self: (_ for _ in ()).throw(AttributeError()))
    st3 = dash.DashboardState("BTC/USDT:USDT", 5, 1, 200, "1m")
    st3._broker = BrokerLama()
    st3.refresh_once()
    check("broker.py versi lama (tanpa fungsi SL/TP) -> tidak crash",
          st3.snapshot["status"] == "ok" and st3.snapshot["bracket_orders"] is None, st3.snapshot.get("last_error"))


def main() -> int:
    test_command()
    test_html()
    test_preview_matches_bot()
    test_panel()
    test_refresh()
    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())