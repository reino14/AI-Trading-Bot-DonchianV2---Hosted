"""
scripts/smoke_smc_liquidity.py

Uji asap microstructure/smc_liquidity.py -- sweep_high yang bisa
dihitung tangan, sweep BOLEH berulang di level sama (beda dari BOS),
sweep BUKAN break sungguhan (close tidak tembus), dan kenari look-ahead.
"""

import sys

import pandas as pd

from src.microstructure.smc_liquidity import detect_liquidity_sweeps

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_df(rows: list[tuple], start="2026-01-01", freq="1h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def main() -> int:
    print("== 1. sweep_high: wick tembus swing high, close kembali di bawahnya ==")
    # Bentuk swing high BERSIH di index 2 (high=50), lalu bar dengan
    # wick tembus 50 tapi CLOSE kembali di bawah 50.
    rows = [
        (10, 20, 9, 18),
        (18, 30, 17, 28),
        (28, 50, 27, 45),   # swing high di 50 (index 2)
        (45, 40, 38, 39),   # turun
        (39, 35, 30, 32),   # turun lagi -- konfirmasi swing high index2 (lookback perlu bar setelahnya)
        (32, 55, 31, 48),   # WICK tembus 50 (high=55) TAPI close=48 < 50 -> SWEEP
    ]
    df = make_df(rows)
    sweeps = detect_liquidity_sweeps(df, lookback=2)
    sweep_high_events = sweeps[sweeps["event_type"] == "sweep_high"]
    check("ADA sweep_high terdeteksi", not sweep_high_events.empty,
          f"jumlah: {len(sweeps)}, tipe: {sweeps['event_type'].tolist() if not sweeps.empty else []}")
    if not sweep_high_events.empty:
        ev = sweep_high_events.iloc[0]
        check("swept_level = 50 (swing high yang di-sweep)", abs(ev["swept_level"] - 50) < 1e-9)
        check("wick_extreme = 55 (high bar sweep)", abs(ev["wick_extreme"] - 55) < 1e-9)
        check("close_price = 48 (close bar sweep, kembali di bawah level)", abs(ev["close_price"] - 48) < 1e-9)

    print("\n== 2. Sweep BOLEH berulang di level yang SAMA (beda dari BOS yang sekali saja) ==")
    # Wick sweep pertama (55) SENGAJA diberi duplikat (bar berikutnya
    # juga high=55) supaya TIDAK lolos jadi swing baru sendiri (gagal
    # syarat unik di find_swing_points) -- level yang diawasi tetap di
    # 50, jadi sweep kedua bisa diuji di level PERSIS SAMA.
    rows2 = rows + [
        (48, 55, 40, 50),   # duplikat high=55 -- membatalkan keunikan bar sweep sebagai swing
        (50, 40, 35, 38),
        (38, 45, 32, 44),
        (44, 53, 42, 47),   # SWEEP KEDUA: wick 53>50, close 47<50, level SAMA (50)
    ]
    df2 = make_df(rows2)
    sweeps2 = detect_liquidity_sweeps(df2, lookback=2)
    sweep_high_2 = sweeps2[sweeps2["event_type"] == "sweep_high"]
    check("ADA 2 sweep_high, KEDUANYA di level 50 yang PERSIS SAMA",
          len(sweep_high_2) == 2 and (abs(sweep_high_2["swept_level"] - 50) < 1e-9).all(),
          f"jumlah sweep_high: {len(sweep_high_2)}, level: {sweep_high_2['swept_level'].tolist()}")

    print("\n== 3. Bar yang BENAR-BENAR tembus (close > level) BUKAN sweep ==")
    rows3 = rows[:5] + [
        (32, 55, 31, 52),   # close=52 > 50 -- TEMBUS SUNGGUHAN, bukan sweep
    ]
    df3 = make_df(rows3)
    sweeps3 = detect_liquidity_sweeps(df3, lookback=2)
    check("TIDAK ADA sweep_high (karena close tembus, bukan ditolak)",
          not (sweeps3["event_type"] == "sweep_high").any() if not sweeps3.empty else True)

    print("\n== 4. KENARI LOOK-AHEAD: ubah bar SETELAH sweep -> sweep lama tidak berubah ==")
    sweeps_before = detect_liquidity_sweeps(df, lookback=2)
    df_future = df.copy()
    extra = pd.DataFrame([[999.0, 999.0, 999.0, 999.0]], columns=["open", "high", "low", "close"],
                          index=[df_future.index[-1] + pd.Timedelta(hours=1)])
    df_extended = pd.concat([df_future, extra])
    sweeps_after = detect_liquidity_sweeps(df_extended, lookback=2)

    first_time = sweeps_before.index[0] if not sweeps_before.empty else None
    check("sweep pertama tidak berubah walau bar baru ditambahkan setelahnya",
          first_time is not None and first_time in sweeps_after.index
          and sweeps_before.loc[first_time, "swept_level"] == sweeps_after.loc[first_time, "swept_level"])

    print("\n== 5. Tidak ada swing sama sekali -> DataFrame kosong, bukan error ==")
    flat_df = make_df([(100, 100, 100, 100)] * 15)
    sweeps5 = detect_liquidity_sweeps(flat_df, lookback=3)
    check("DataFrame kosong untuk data datar tanpa swing", sweeps5.empty)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())