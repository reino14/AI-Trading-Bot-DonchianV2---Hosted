"""
scripts/smoke_dashboard_donchian.py

Uji asap dashboard: perbaikan paginasi (bug "cuma tanggal 16"),
filter tanggal, hitungan channel, dan penyusunan perintah bot.
Semua dengan bursa PALSU -- tanpa koneksi sungguhan.
"""

import sys

sys.path.insert(0, ".")
from scripts.dashboard_donchian import (  # noqa: E402
    CHUNK_MS, aggregate_trades, build_bot_command, compute_channel,
    fetch_all_trades, filter_trades_by_time,
)

FAILURES: list[str] = []
HARI = 24 * 60 * 60 * 1000


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class BursaPalsu:
    """
    Meniru perilaku Binance yang menyebabkan bug: mengembalikan trade
    URUT NAIK dari yang terlama, dibatasi `limit` per permintaan, dan
    menghormati startTime/endTime.
    """

    def __init__(self, trades):
        self.trades = sorted(trades, key=lambda t: t["timestamp"])
        self.calls = 0

    def fetch_my_trades(self, symbol, since=None, limit=None, params=None):
        self.calls += 1
        params = params or {}
        end = params.get("endTime")
        hasil = [t for t in self.trades
                 if (since is None or t["timestamp"] >= since)
                 and (end is None or t["timestamp"] <= end)]
        return hasil[: (limit or 1000)]


def buat_trade(ts, pnl=0.0, fee=0.01, tid=None):
    return {
        "id": tid or f"t{ts}", "timestamp": ts, "order": f"o{ts}",
        "side": "buy", "price": 100.0, "amount": 0.01,
        "fee": {"cost": fee, "currency": "USDT"},
        "info": {"realizedPnl": str(pnl), "commission": str(fee)},
    }


def main() -> int:
    base = 1_789_000_000_000  # titik awal sembarang

    print("== 1. BUG REPRODUKSI: 300 fill di hari-1, 5 fill di hari-2 dan hari-3 ==")
    # Persis pola Nero: hari pertama PENUH split-fill (ratusan), hari
    # berikutnya sedikit. Limit per halaman sengaja 200 -- versi LAMA
    # (sekali panggil limit=200) cuma akan dapat hari-1 saja.
    trades = [buat_trade(base + i * 1000) for i in range(300)]              # hari-1
    trades += [buat_trade(base + 1 * HARI + i * 1000, pnl=5.0) for i in range(5)]   # hari-2
    trades += [buat_trade(base + 2 * HARI + i * 1000, pnl=-2.0) for i in range(5)]  # hari-3

    bursa = BursaPalsu(trades)

    # Simulasikan versi LAMA (satu panggilan, limit 200) untuk perbandingan.
    lama = bursa.fetch_my_trades("BTC/USDT:USDT", limit=200)
    hari_di_lama = {(t["timestamp"] - base) // HARI for t in lama}
    check("versi LAMA memang cuma dapat hari-1 (bug terkonfirmasi)",
          hari_di_lama == {0}, f"hari yang terambil: {sorted(hari_di_lama)}")

    # Versi BARU -- paginasi.
    baru = fetch_all_trades(bursa, "BTC/USDT:USDT", base - 1000, base + 3 * HARI, page_limit=200)
    hari_di_baru = {(t["timestamp"] - base) // HARI for t in baru}
    check("versi BARU dapat SEMUA 310 fill", len(baru) == 310, f"dapat {len(baru)}")
    check("versi BARU mencakup hari-1, hari-2, DAN hari-3",
          hari_di_baru == {0, 1, 2}, f"hari yang terambil: {sorted(hari_di_baru)}")

    print("\n== 2. Tidak ada duplikat walau bursa kirim ulang di tepi halaman ==")
    ids = [t["id"] for t in baru]
    check("semua id unik (tidak ada trade dihitung dobel)",
          len(ids) == len(set(ids)), f"{len(ids)} trade, {len(set(ids))} unik")

    print("\n== 3. Rentang > 7 hari dipotong jadi beberapa permintaan (batas Binance) ==")
    trades_panjang = [buat_trade(base + i * HARI) for i in range(20)]  # 20 hari
    bursa2 = BursaPalsu(trades_panjang)
    hasil = fetch_all_trades(bursa2, "BTC/USDT:USDT", base - 1000, base + 20 * HARI, page_limit=1000)
    check("semua 20 trade lintas 20 hari terambil", len(hasil) == 20, f"dapat {len(hasil)}")
    check("dipecah jadi >= 3 chunk (20 hari / 7 hari per chunk)",
          bursa2.calls >= 3, f"jumlah panggilan: {bursa2.calls}")

    print("\n== 4. Filter tanggal: cuma hari-2 ==")
    hanya_hari2 = filter_trades_by_time(baru, base + 1 * HARI, base + 2 * HARI - 1)
    check("5 fill (semua dari hari-2)", len(hanya_hari2) == 5, f"dapat {len(hanya_hari2)}")
    a2 = aggregate_trades(hanya_hari2)
    check("total realisasi hari-2 = +25.0 (5 x 5.0)", abs(a2["total_realized"] - 25.0) < 1e-9,
          f"dapat {a2['total_realized']}")
    check("bersih = 25.0 - fee 0.05 = 24.95", abs(a2["net"] - 24.95) < 1e-9, f"dapat {a2['net']}")

    print("\n== 5. Filter tanggal: cuma hari-3 (rugi) -- terpisah dari hari-2 yang untung ==")
    hanya_hari3 = filter_trades_by_time(baru, base + 2 * HARI, base + 3 * HARI)
    a3 = aggregate_trades(hanya_hari3)
    check("realisasi hari-3 = -10.0 (5 x -2.0)", abs(a3["total_realized"] - (-10.0)) < 1e-9,
          f"dapat {a3['total_realized']}")
    check("hari-2 dan hari-3 BENAR-BENAR terpisah (tidak terakumulasi)",
          a2["total_realized"] > 0 and a3["total_realized"] < 0)

    print("\n== 6. Tanpa filter -> semua terakumulasi (+25 -10 = +15) ==")
    a_all = aggregate_trades(filter_trades_by_time(baru, None, None))
    check("realisasi total = +15.0", abs(a_all["total_realized"] - 15.0) < 1e-9,
          f"dapat {a_all['total_realized']}")

    print("\n== 7. compute_channel: rumus sama dengan sinyal bot (max/min close SEBELUM bar terakhir) ==")
    closes = [10, 20, 30, 25, 15, 12]  # lookback=5 -> window = 5 bar pertama, bar terakhir (12) TIDAK ikut
    c = compute_channel(closes, lookback=5)
    check("atas = 30 (max dari 5 bar sebelum terakhir)", c["upper"] == 30, f"dapat {c['upper']}")
    check("bawah = 10 (min dari 5 bar sebelum terakhir)", c["lower"] == 10, f"dapat {c['lower']}")
    check("harga acuan = 12 (bar terakhir)", c["current"] == 12)
    check("jarak ke atas = (30-12)/12 = 150%", abs(c["dist_upper"] - 1.5) < 1e-9, f"dapat {c['dist_upper']}")
    check("jarak ke bawah = (12-10)/12 = 16.67%", abs(c["dist_lower"] - (2/12)) < 1e-9)

    print("\n== 8. compute_channel: data kurang -> None, bukan angka ngasal ==")
    check("None untuk data kurang dari lookback+1", compute_channel([1, 2, 3], lookback=10) is None)

    print("\n== 9. build_bot_command: persis perintah yang biasa Nero ketik ==")
    cmd = build_bot_command({
        "symbol": "BTC/USDT:USDT", "timeframe": "1m", "lookback": 200, "amount": 0.01,
        "session_hours": 24, "take_profit_pct": 0.003, "stop_after_take_profit": True,
        "backfill_bars": 200, "live_take_profit_poll_seconds": 5,
    })
    s = " ".join(cmd)
    for bagian in ["--live", "--symbol BTC/USDT:USDT", "--timeframe 1m", "--lookback 200",
                    "--amount 0.01", "--session-hours 24.0", "--take-profit-pct 0.003",
                    "--stop-after-take-profit", "--backfill-bars 200",
                    "--live-take-profit-poll-seconds 5.0"]:
        check(f"mengandung '{bagian}'", bagian in s)
    check("memanggil modul bot yang SUDAH ADA (bukan logika disalin)",
          "scripts.run_paper_donchian_futures" in s)

    print("\n== 10. build_bot_command: opsi kosong TIDAK dikirim (bukan dikirim nilai palsu) ==")
    cmd2 = build_bot_command({
        "symbol": "ETH/USDT:USDT", "timeframe": "5m", "lookback": 50, "amount": 0.1,
        "session_hours": None, "take_profit_pct": None, "stop_after_take_profit": False,
        "backfill_bars": None, "live_take_profit_poll_seconds": None,
    })
    s2 = " ".join(cmd2)
    check("tidak ada --session-hours", "--session-hours" not in s2)
    check("tidak ada --take-profit-pct", "--take-profit-pct" not in s2)
    check("tidak ada --stop-after-take-profit", "--stop-after-take-profit" not in s2)
    check("tidak ada --backfill-bars", "--backfill-bars" not in s2)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())