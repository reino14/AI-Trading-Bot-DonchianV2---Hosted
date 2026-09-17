"""
scripts/smoke_smc_structure.py

Uji asap microstructure/smc_structure.py -- uptrend jelas dengan BOS
bullish berturut-turut, lalu tembus swing low untuk CHoCH bearish.
Plus kenari look-ahead dan bukti peristiwa cuma tercatat SEKALI per
level (bukan tiap bar selama harga bertahan di atas/bawah level itu).
"""

import sys

import pandas as pd

from src.microstructure.smc_structure import detect_bos_choch

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_ohlc(closes: list[float], start="2026-01-01", freq="1h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
    }, index=idx)


def main() -> int:
    print("== 1. Uptrend zigzag naik: BOS bullish tiap kali tembus swing high sebelumnya ==")
    # Zigzag naik: low->high->higher_low->higher_high->dst, HH/HL murni.
    closes = []
    for lo, hi in [(10, 30), (20, 45), (35, 60), (50, 75)]:
        for i in range(8):
            closes.append(lo + (hi - lo) * (i + 1) / 8)
    df = make_ohlc(closes)
    events = detect_bos_choch(df, lookback=3)
    check("ADA peristiwa BOS_bullish terdeteksi", (events["event_type"] == "BOS_bullish").any(),
          f"jenis peristiwa: {events['event_type'].tolist() if not events.empty else []}")
    # Peristiwa PERTAMA di awal zigzag WAJAR berupa CHoCH_bullish -- di
    # titik itu struktur belum punya cukup swing untuk diklasifikasi
    # "up" (butuh 2 high + 2 low dulu), jadi tembus pertama terjadi
    # saat tren masih "sideways" -- itu PERSIS definisi CHoCH ("sideways
    # + tembus = sinyal awal tren baru"). Yang tidak boleh ada adalah
    # CHoCH SETELAH tren sudah mapan "up" tanpa pembalikan sungguhan.
    established_up = events[events["trend_before"] == "up"]
    check("SETELAH tren mapan 'up', semua tembus high berikutnya = BOS (bukan CHoCH)",
          (established_up["event_type"] != "CHoCH_bullish").all() if not established_up.empty else True,
          f"peristiwa saat trend_before='up': {established_up['event_type'].tolist() if not established_up.empty else []}")

    print("\n== 2. Uptrend lalu tembus swing LOW -> CHoCH_bearish (sinyal pembalikan) ==")
    # Lanjutkan uptrend di atas, lalu jatuhkan tajam di bawah swing low terakhir.
    closes2 = closes + [70, 60, 40, 20, 5, 0, -5]  # jatuh tajam di akhir
    df2 = make_ohlc(closes2)
    events2 = detect_bos_choch(df2, lookback=3)
    check("ADA CHoCH_bearish di bagian akhir (setelah harga jatuh tembus swing low)",
          (events2["event_type"] == "CHoCH_bearish").any(),
          f"jenis peristiwa: {events2['event_type'].tolist()}")

    print("\n== 3. Peristiwa cuma tercatat SEKALI per level, bukan tiap bar ==")
    # Swing high BERSIH di index 2 (harga 30) -- turun setelahnya (index
    # 3-5) supaya terkonfirmasi, lalu naik TERUS-MENERUS tanpa pernah
    # turun lagi (index 6 dst) -- jadi TIDAK ADA swing high baru yang
    # bisa terbentuk (butuh turun dulu baru bisa ada puncak baru).
    # Puluhan bar berikutnya SEMUA close di atas 30 -- kalau state
    # machine salah, ini akan mencatat peristiwa berulang tiap bar.
    closes3 = [10, 20, 30, 25, 20, 15] + list(range(40, 110, 5))
    df3 = make_ohlc(closes3, freq="1h")
    events3 = detect_bos_choch(df3, lookback=2)
    bullish_events = events3[events3["event_type"].str.endswith("bullish")]
    check("CUMA 1 peristiwa bullish tercatat (bukan berulang di puluhan bar berikutnya)",
          len(bullish_events) == 1,
          f"dapat {len(bullish_events)} peristiwa: {bullish_events['event_type'].tolist()}")
    if len(bullish_events) == 1:
        check("level yang ditembus = 30.5 (kolom HIGH bar itu, bukan close -- swing dihitung dari high/low)",
              abs(bullish_events.iloc[0]["broken_price"] - 30.5) < 1e-9,
              f"dapat {bullish_events.iloc[0]['broken_price']}")

    print("\n== 4. KENARI LOOK-AHEAD: ubah bar SETELAH suatu peristiwa -> peristiwa lama TIDAK berubah ==")
    cutoff_events = detect_bos_choch(df, lookback=3)
    if not cutoff_events.empty:
        first_event_time = cutoff_events.index[0]
        first_event_type = cutoff_events.iloc[0]["event_type"]

        df_future_changed = df.copy()
        # Ubah drastis SEMUA bar SETELAH peristiwa pertama.
        change_from = df.index.get_loc(first_event_time) + 1
        df_future_changed.iloc[change_from:, df_future_changed.columns.get_loc("close")] = 0.001
        df_future_changed.iloc[change_from:, df_future_changed.columns.get_loc("high")] = 0.002
        df_future_changed.iloc[change_from:, df_future_changed.columns.get_loc("low")] = 0.0005

        events_changed = detect_bos_choch(df_future_changed, lookback=3)
        still_present = (
            not events_changed.empty
            and first_event_time in events_changed.index
            and events_changed.loc[first_event_time, "event_type"] == first_event_type
        )
        check("peristiwa PERTAMA tidak berubah walau semua data SETELAHNYA diubah total",
              still_present, f"event pertama masih ada dan sama: {still_present}")
    else:
        check("(dilewati -- tidak ada peristiwa di skenario 1 untuk diuji)", True)

    print("\n== 5. Tidak ada peristiwa sama sekali -> DataFrame kosong, bukan error ==")
    flat_df = make_ohlc([100.0] * 20)  # datar total, tidak ada swing
    events5 = detect_bos_choch(flat_df, lookback=3)
    check("DataFrame kosong untuk data datar tanpa swing", events5.empty)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())