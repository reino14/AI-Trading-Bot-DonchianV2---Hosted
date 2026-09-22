"""
scripts/smoke_notifier_email.py

Uji asap notifier Gmail. Tanpa Binance dan tanpa Gmail sungguhan: bursa,
jam, dan server SMTP diganti versi palsu. Dua kasus memakai data trade
NYATA dari akun demo (21 Sep 2026).

    python -m scripts.smoke_notifier_email
"""

from __future__ import annotations

import io
import smtplib
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import notifier_email as ne  # noqa: E402

FAILURES: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'LULUS' if cond else 'GAGAL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def raw(tid, ts, side, amount, price, fee, pnl=0.0, order=None):
    """Bentuk trade seperti keluaran ccxt fetch_my_trades."""
    return {"id": str(tid), "order": str(order or tid), "timestamp": ts, "side": side,
            "amount": amount, "price": price, "fee": {"cost": fee, "currency": "USDT"},
            "info": {"realizedPnl": str(pnl), "commission": str(fee)}}


def run(fills):
    st = {"qty": 0.0, "trip": None}
    ev = []
    for f in ne.merge_by_order([ne.normalize_fill(t) for t in fills]):
        ev.extend(ne.apply_fill(st, f))
    return st, ev


def test_reconstruction():
    print("== 1. Rekonstruksi trade dari fill ==")
    st, ev = run([raw(1, 1000, "buy", 0.01, 100.0, 0.04), raw(2, 2000, "sell", 0.01, 101.0, 0.0404, 0.01)])
    c = [e for e in ev if e["type"] == "CLOSE"]
    check("1 OPEN + 1 CLOSE", [e["type"] for e in ev] == ["OPEN", "CLOSE"], str([e["type"] for e in ev]))
    check("BERSIH = pnl - fee buka - fee tutup", abs(c[0]["net"] - (0.01 - 0.0804)) < 1e-9, str(c[0]["net"]))
    check("harga masuk & keluar benar", c[0]["entry_avg"] == 100.0 and c[0]["exit_avg"] == 101.0)
    check("posisi kembali nol", st["qty"] == 0.0 and st["trip"] is None)

    print("\n== 2. DATA NYATA: short 0.03, 21 Sep 01:52 -> 01:57 ==")
    _, ev = run([raw("s", 1, "sell", 0.03, 80968.2, 1.214523, 0.0),
                 raw("b", 2, "buy", 0.03, 80948.5, 1.2142275, 0.591)])
    c = [e for e in ev if e["type"] == "CLOSE"][0]
    check("terbaca sebagai SHORT", c["side"] == "short")
    check("fee dua sisi = 2.4288 USDT", abs(c["fees"] - 2.4287505) < 1e-9, f"{c['fees']:.7f}")
    check("BERSIH -1.8378 USDT (kotor +0.591 habis dimakan fee)",
          abs(c["net"] - (0.591 - 2.4287505)) < 1e-9, f"{c['net']:.7f}")
    subj, body = ne.format_event(c, "DEMO", "BTC/USDT:USDT", 20.0, 0.004)
    check("subjek email menyebut RUGI dan angkanya", "RUGI" in subj and "-1.8378" in subj, subj)
    check("isi email jujur soal alasan penutupan yang tidak diketahui", "tidak mencatat ALASAN" in body)

    print("\n== 3. DATA NYATA: 12 fill 20-21 Sep -> 6 trade utuh ==")
    seq = [("buy", .005, 80836.5), ("sell", .005, 80965.7), ("sell", .01, 80968.4), ("buy", .01, 80848.7),
           ("sell", .01, 80680.4), ("buy", .01, 80584.1), ("buy", .01, 81247.0), ("sell", .01, 81406.9),
           ("buy", .01, 81789.0), ("sell", .01, 81967.5), ("sell", .03, 80968.2), ("buy", .03, 80948.5)]
    _, ev = run([raw(i, i, s, a, p, 0.4) for i, (s, a, p) in enumerate(seq)])
    tipe = [e["type"] for e in ev]
    check("6 OPEN, 6 CLOSE, 0 ukuran-berubah",
          tipe.count("OPEN") == 6 and tipe.count("CLOSE") == 6 and tipe.count("SIZE") == 0, str(tipe))

    print("\n== 4. Satu order terisi 10 pecahan (pola order 28587818567) ==")
    pecahan = [0.0007, 0.0008, 0.0011, 0.0008, 0.0007, 0.0009, 0.0008, 0.0010, 0.0012, 0.0020]
    st, ev = run([raw(f"f{i}", 100 + i, "sell", a, 75631.2, 0.015, order="28587818567") for i, a in enumerate(pecahan)])
    check("tepat 1 event MASUK, bukan 10", [e["type"] for e in ev] == ["OPEN"], str([e["type"] for e in ev]))
    check("jumlahnya 0.01 utuh", abs(ev[0]["qty"] - 0.01) < 1e-9, str(ev[0]["qty"]))

    print("\n== 5. Fill menyeberangi nol (tombol Reverse manual) ==")
    _, ev = run([raw(1, 1, "buy", 0.01, 100.0, 0.04), raw(2, 2, "sell", 0.03, 102.0, 0.12, 0.02)])
    check("CLOSE long lalu OPEN short", [(e["type"], e["side"]) for e in ev] ==
          [("OPEN", "long"), ("CLOSE", "long"), ("OPEN", "short")], str([(e["type"], e["side"]) for e in ev]))
    c = ev[1]
    check("fee penutup = 1/3 fee order (0.04)", abs(c["fees"] - (0.04 + 0.04)) < 1e-9, str(c["fees"]))
    check("short baru 0.02", abs(ev[2]["qty"] - 0.02) < 1e-9)

    print("\n== 6. Posisi sudah terbuka saat notifier pertama aktif ==")
    hist = [ne.normalize_fill(t) for t in [raw(1, 1, "buy", 0.01, 100, 0.04), raw(2, 2, "sell", 0.01, 101, 0.04, 0.01),
                                            raw(3, 3, "sell", 0.02, 99, 0.08)]]
    trip = ne.reconstruct_open_trip(hist, -0.02)
    check("trip berjalan ditemukan utuh (fee pembuka ikut)",
          trip and trip["complete"] and abs(sum(p["fee"] for p in trip["parts"]) - 0.08) < 1e-9, str(trip))
    trip2 = ne.reconstruct_open_trip([ne.normalize_fill(raw(9, 9, "sell", 0.01, 99, 0.04))], -0.03)
    check("riwayat kurang jauh -> ditandai tidak lengkap, bukan ditebak", trip2["complete"] is False)


class FakeExchange:
    def __init__(self):
        self.trades = []


class FakeBroker:
    def __init__(self):
        self.exchange = FakeExchange()
        self.pos = None

    def fetch_position(self, symbol):
        print("  [fetch_position] leverage dihitung dari notional/initialMargin = 20.00x")  # tiru log berisik VPS
        return self.pos


class FakeSender:
    def __init__(self):
        self.sent = []

    def send(self, subject, body):
        self.sent.append((subject, body))
        return True


def fake_fetch(exchange, symbol, since, until):
    return [t for t in exchange.trades if since <= t["timestamp"] <= until]


def test_notifier(tmp: Path):
    print("\n== 7. Pemantau utuh: jam & bursa palsu ==")
    clock = {"t": 1_000_000}
    br, sd = FakeBroker(), FakeSender()
    path = tmp / "state.json"
    mk = lambda: ne.Notifier(br, "BTC/USDT:USDT", path, sd, label="DEMO", tp_pct=0.004,
                             fetch_trades=fake_fetch, now_ms=lambda: clock["t"])
    n = mk()
    out = io.StringIO()
    with redirect_stdout(out):
        n.start()
    check("start pertama kirim email 'Notifier aktif'", len(sd.sent) == 1 and "Notifier aktif" in sd.sent[0][0])
    check("log berisik [fetch_position] dibungkam", "[fetch_position]" not in out.getvalue())

    t0 = clock["t"]
    br.exchange.trades += [raw(f"o{i}", t0 + 1000 + i, "buy", 0.005, 80000.0, 0.2, order="ORD1") for i in range(2)]
    br.pos = {"side": "long", "contracts": 0.01, "leverage": 20}
    clock["t"] = t0 + 5_000
    with redirect_stdout(io.StringIO()):
        n.tick()
    check("fill baru berumur < 20 dtk: BELUM diumumkan (menunggu pecahan lain)", len(sd.sent) == 1)
    clock["t"] = t0 + 30_000
    with redirect_stdout(io.StringIO()):
        n.tick()
    check("setelah mengendap: tepat 1 email MASUK LONG", len(sd.sent) == 2 and "MASUK LONG" in sd.sent[1][0],
          sd.sent[-1][0])
    with redirect_stdout(io.StringIO()):
        n.tick()
    check("putaran berikutnya tidak mengulang email", len(sd.sent) == 2)

    n2 = mk()  # simulasi restart service
    with redirect_stdout(io.StringIO()):
        n2.start()
        n2.tick()
    check("restart: tidak ada email ganda", len(sd.sent) == 2, str([s for s, _ in sd.sent]))

    t1 = clock["t"]
    br.exchange.trades.append(raw("x1", t1 + 100, "buy", 0.02, 80100.0, 0.8, order="ORD2"))
    br.pos = {"side": "long", "contracts": 0.03, "leverage": 20}
    clock["t"] = t1 + 30_000
    with redirect_stdout(io.StringIO()):
        n2.tick()
    check("order BARU menambah posisi -> email UKURAN BERTAMBAH 0.01 -> 0.03",
          "BERTAMBAH" in sd.sent[-1][0] and "0.03" in sd.sent[-1][0], sd.sent[-1][0])

    t2 = clock["t"]
    br.exchange.trades.append(raw("c1", t2 + 100, "sell", 0.03, 80500.0, 1.2, pnl=13.0, order="ORD3"))
    br.pos = None
    clock["t"] = t2 + 30_000
    with redirect_stdout(io.StringIO()):
        n2.tick()
    subj, body = sd.sent[-1]
    fee_total = 0.2 + 0.2 + 0.8 + 1.2
    check("email TUTUP UNTUNG dengan angka bersih", "TUTUP LONG" in subj and "UNTUNG" in subj, subj)
    check("BERSIH = 13.0 - semua fee (2.4)", f"{13.0 - fee_total:+.4f}" in subj, subj)
    check("isi memuat ROI & ambang TP", "ROI bersih" in body and "Ambang TP" in body)

    print("\n== 8. Selisih posisi bertahan -> peringatan, lalu selaras ==")
    br.pos = {"side": "short", "contracts": 0.01, "leverage": 20}  # berubah tanpa fill (mis. data fill terlambat)
    n_before = len(sd.sent)
    with redirect_stdout(io.StringIO()):
        n2.tick()
    check("selisih sekali: belum diperingatkan (bisa sesaat)", len(sd.sent) == n_before)
    with redirect_stdout(io.StringIO()):
        n2.tick()
    check("selisih dua kali: email PERINGATAN", "PERINGATAN" in sd.sent[-1][0])
    check("notifier menyelaraskan ke posisi bursa", abs(n2.state["qty"] - (-0.01)) < 1e-9)


class FakeSMTP:
    log = []
    fail_times = 0
    auth_fail = False

    def __init__(self, host, port, timeout=None):
        FakeSMTP.log.append(("connect", host, port))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        FakeSMTP.log.append(("starttls",))

    def login(self, u, p):
        if FakeSMTP.auth_fail:
            raise smtplib.SMTPAuthenticationError(535, b"bad")
        FakeSMTP.log.append(("login", u, p))

    def send_message(self, msg):
        if FakeSMTP.fail_times > 0:
            FakeSMTP.fail_times -= 1
            raise OSError("jaringan putus")
        FakeSMTP.log.append(("send", msg["To"], msg["Subject"]))


def test_gmail():
    print("\n== 9. Pengirim Gmail ==")
    FakeSMTP.log = []
    g = ne.GmailSender("bot@gmail.com", "abcd efgh ijkl mnop", ["a@x.com", " b@x.com"],
                       smtp_factory=FakeSMTP, sleep=lambda s: None)
    check("kirim berhasil", g.send("uji", "isi") is True)
    urutan = [e[0] for e in FakeSMTP.log]
    check("STARTTLS sebelum login (password tidak dikirim polos)",
          urutan.index("starttls") < urutan.index("login"), str(urutan))
    check("spasi di App Password dibuang", ("login", "bot@gmail.com", "abcdefghijklmnop") in FakeSMTP.log)
    check("ke smtp.gmail.com:587", ("connect", "smtp.gmail.com", 587) in FakeSMTP.log)
    check("semua penerima tercantum", any(e[0] == "send" and e[1] == "a@x.com, b@x.com" for e in FakeSMTP.log))

    FakeSMTP.log, FakeSMTP.fail_times = [], 2
    with redirect_stdout(io.StringIO()):
        ok = g.send("uji", "isi")
    check("jaringan putus 2x lalu pulih -> tetap terkirim", ok and FakeSMTP.log[-1][0] == "send")

    FakeSMTP.log, FakeSMTP.auth_fail = [], True
    with redirect_stdout(io.StringIO()):
        ok = g.send("uji", "isi")
    check("password salah -> berhenti, tidak dicoba ulang", ok is False and
          sum(1 for e in FakeSMTP.log if e[0] == "connect") == 1)
    FakeSMTP.auth_fail = False


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        test_reconstruction()
        test_notifier(Path(d))
        test_gmail()
    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())