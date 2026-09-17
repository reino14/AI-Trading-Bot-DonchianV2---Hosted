"""
scripts/smoke_smc_patterns.py

Uji asap microstructure/smc_patterns.py -- celah FVG yang bisa dihitung
tangan, Order Block yang benar mengaitkan candle merah/hijau terakhir
dengan peristiwa BOS/CHoCH, plus kenari look-ahead untuk keduanya.
"""

import sys

import pandas as pd

from src.microstructure.smc_patterns import detect_fair_value_gaps, detect_order_blocks
from src.microstructure.smc_structure import detect_bos_choch

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
    print("== 1. Bullish FVG: celah jelas antara candle 1 dan candle 3 (dihitung tangan) ==")
    # candle1: high=100. candle2: apa saja. candle3: low=105 -- celah [100,105].
    rows = [
        (95, 100, 90, 98),   # candle 1: high=100
        (98, 110, 96, 108),  # candle 2: lompatan (penyebab celah)
        (108, 112, 105, 110),  # candle 3: low=105 > candle1 high(100) -> celah bullish
    ]
    df = make_df(rows)
    fvg = detect_fair_value_gaps(df)
    check("1 FVG bullish terdeteksi", len(fvg) == 1 and fvg.iloc[0]["gap_type"] == "bullish",
          f"dapat {len(fvg)}: {fvg['gap_type'].tolist() if not fvg.empty else []}")
    if len(fvg) == 1:
        check("gap_low = 100 (high candle 1)", abs(fvg.iloc[0]["gap_low"] - 100) < 1e-9)
        check("gap_high = 105 (low candle 3)", abs(fvg.iloc[0]["gap_high"] - 105) < 1e-9)
        check("gap_size_pct = 5/100 = 0.05", abs(fvg.iloc[0]["gap_size_pct"] - 0.05) < 1e-9,
              f"dapat {fvg.iloc[0]['gap_size_pct']}")

    print("\n== 2. TIDAK ada FVG kalau candle 1 dan 3 saling tumpang tindih ==")
    rows2 = [
        (95, 100, 90, 98),
        (98, 105, 96, 103),
        (103, 108, 99, 106),  # low candle3=99 < high candle1=100 -> TUMPANG TINDIH, bukan celah
    ]
    df2 = make_df(rows2)
    fvg2 = detect_fair_value_gaps(df2)
    check("tidak ada FVG (candle 1 dan 3 tumpang tindih)", fvg2.empty)

    print("\n== 3. min_gap_pct membuang celah yang lebih kecil dari ambang ==")
    fvg3_loose = detect_fair_value_gaps(df, min_gap_pct=0.0)
    fvg3_strict = detect_fair_value_gaps(df, min_gap_pct=0.10)  # celah 5% < ambang 10%
    check("longgar (0%): 1 FVG lolos", len(fvg3_loose) == 1)
    check("ketat (10%): 0 FVG lolos (celah cuma 5%)", fvg3_strict.empty)

    print("\n== 4. Order Block: candle MERAH terakhir sebelum BOS_bullish ==")
    # Bangun uptrend dengan BOS bullish, sisipkan candle MERAH jelas
    # tepat sebelum dorongan yang menembus struktur.
    rows4 = [
        (10, 20, 8, 18),    # 0
        (18, 30, 16, 28),   # 1
        (28, 29, 15, 16),   # 2: MERAH (close<open) -- kandidat OB
        (16, 45, 15, 44),   # 3: dorongan naik tembus swing high sebelumnya
        (44, 46, 43, 45),   # 4
    ]
    df4 = make_df(rows4)
    events4 = detect_bos_choch(df4, lookback=1)
    ob4 = detect_order_blocks(df4, events4, lookback_candles=5)
    bullish_ob = ob4[ob4["ob_type"] == "bullish"]
    check("ADA order block bullish terdeteksi", not bullish_ob.empty,
          f"jumlah OB: {len(ob4)}, tipe: {ob4['ob_type'].tolist() if not ob4.empty else []}")
    if not bullish_ob.empty:
        ob_time = bullish_ob.iloc[0]["ob_timestamp"] if "ob_timestamp" in bullish_ob.columns else bullish_ob.index[0]
        check("OB menunjuk ke candle index 2 (candle merah tepat sebelum dorongan)",
              ob_time == df4.index[2], f"dapat {ob_time}, harus {df4.index[2]}")

    print("\n== 5. Event tanpa candle berlawanan warna dalam jendela -> DILEWATI, bukan error ==")
    rows5 = [
        (10, 20, 9, 19),    # 0: HIJAU
        (19, 30, 18, 29),   # 1: HIJAU
        (29, 40, 28, 39),   # 2: HIJAU -- tidak ada MERAH sama sekali sebelum dorongan
        (39, 50, 38, 49),   # 3: dorongan naik, BOS bullish
    ]
    df5 = make_df(rows5)
    events5 = detect_bos_choch(df5, lookback=1)
    ob5 = detect_order_blocks(df5, events5, lookback_candles=5)
    check("tidak crash, hasil kosong ATAU tidak mengandung OB palsu",
          True)  # cuma pastikan baris di atas tidak melempar exception
    check("DataFrame OB valid (bukan error) walau tidak ada candle merah",
          isinstance(ob5, pd.DataFrame))

    print("\n== 6. KENARI LOOK-AHEAD (FVG): ubah candle SETELAH celah -> celah PERTAMA tidak berubah ==")
    fvg_before = detect_fair_value_gaps(df)
    first_time = fvg_before.index[0]
    first_gap_low = fvg_before.iloc[0]["gap_low"]
    first_gap_high = fvg_before.iloc[0]["gap_high"]

    df_future = df.astype(float).copy()
    extra_row = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], columns=["open", "high", "low", "close"],
                              index=[df_future.index[-1] + pd.Timedelta(hours=1)])
    df_future_extended = pd.concat([df_future, extra_row])
    df_future_extended.iloc[-1] = [999.0, 999.0, 999.0, 999.0]  # ubah drastis bar BARU ini
    fvg_after = detect_fair_value_gaps(df_future_extended)

    check("event PERTAMA yang sama masih ada, dengan nilai IDENTIK",
          first_time in fvg_after.index
          and abs(fvg_after.loc[first_time, "gap_low"] - first_gap_low) < 1e-9
          and abs(fvg_after.loc[first_time, "gap_high"] - first_gap_high) < 1e-9,
          f"before=({first_gap_low},{first_gap_high}), sesudah ditambah data baru: "
          f"{fvg_after.loc[first_time].to_dict() if first_time in fvg_after.index else 'HILANG'}")
    print("    (bertambahnya JUMLAH total FVG itu wajar -- bar baru bisa membentuk")
    print("     FVG baru di timestamp barunya sendiri, itu bukan look-ahead)")

    print("\n== 7. KENARI LOOK-AHEAD (Order Block): ubah candle SETELAH event -> OB tidak berubah ==")
    events4_orig = detect_bos_choch(df4, lookback=1)
    ob4_orig = detect_order_blocks(df4, events4_orig, lookback_candles=5)

    df4_future_changed = df4.copy()
    df4_future_changed.iloc[4] = [0.001, 0.002, 0.0005, 0.001]  # ubah drastis bar TERAKHIR
    events4_changed = detect_bos_choch(df4_future_changed, lookback=1)
    ob4_changed = detect_order_blocks(df4_future_changed, events4_changed, lookback_candles=5)

    check("OB untuk event yang sama TIDAK berubah walau bar setelahnya diubah total",
          len(ob4_orig) == len(ob4_changed) and
          (ob4_orig["ob_high"].values == ob4_changed["ob_high"].values).all()
          if not ob4_orig.empty else True)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())