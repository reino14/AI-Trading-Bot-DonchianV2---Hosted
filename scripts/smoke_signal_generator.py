"""
scripts/smoke_signal_generator.py

Uji asap microstructure/signal_generator.py -- satu skenario valid
(semua syarat terpenuhi), dan satu skenario per ALASAN PENOLAKAN
(supaya tahu tiap syarat benar-benar diperiksa, bukan cuma salah satu
yang kebetulan lolos). Plus kenari look-ahead untuk value area.
"""

import sys

import pandas as pd

from src.microstructure.market_structure import find_swing_points
from src.microstructure.signal_generator import generate_setups

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_context():
    """Konteks dasar: bias up (1H+4H sepakat), profil hari sebelumnya
    dengan VAL=95, POC=100, VAH=105."""
    idx_1h = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    structure_1h = pd.Series(["up"] * len(idx_1h), index=idx_1h)
    idx_4h = pd.date_range("2026-01-01", periods=12, freq="4h", tz="UTC")
    structure_4h = pd.Series(["up"] * len(idx_4h), index=idx_4h)
    daily_profile = pd.DataFrame({
        "session_id": ["2026-01-01"],
        "val_price": [95.0], "poc_price": [100.0], "vah_price": [105.0],
    })
    return structure_1h, structure_4h, daily_profile


def make_bar(t, open_, high, low, close, is_absorption):
    return pd.DataFrame({
        "open": [open_], "high": [high], "low": [low], "close": [close],
        "is_absorption": [is_absorption],
    }, index=pd.DatetimeIndex([t], tz="UTC"))


BAR_TIME = pd.Timestamp("2026-01-02 10:00:00", tz="UTC")  # hari SETELAH profil


def main() -> int:
    structure_1h, structure_4h, daily_profile = make_context()

    print("== 1. Setup valid: semua syarat LONG terpenuhi ==")
    bar = make_bar(BAR_TIME, open_=93.0, high=94.5, low=92.0, close=94.0, is_absorption=True)
    setups = generate_setups(bar, structure_1h, structure_4h, daily_profile)
    check("1 setup dihasilkan", len(setups) == 1, f"dapat {len(setups)}")
    if len(setups) == 1:
        s = setups.iloc[0]
        check("arah = long", s["direction"] == "long")
        check("entry = close bar (94.0)", s["entry_price"] == 94.0)
        check("stop = low bar (92.0)", s["stop_price"] == 92.0)
        check("target = POC hari sebelumnya (100.0)", s["target_price"] == 100.0)

    print("\n== 2. GAGAL: bukan absorption ==")
    bar2 = make_bar(BAR_TIME, 93.0, 94.5, 92.0, 94.0, is_absorption=False)
    check("tidak ada setup", generate_setups(bar2, structure_1h, structure_4h, daily_profile).empty)

    print("\n== 3. GAGAL: 1H dan 4H TIDAK sepakat ==")
    structure_4h_down = structure_4h.copy()
    structure_4h_down[:] = "down"
    bar3 = make_bar(BAR_TIME, 93.0, 94.5, 92.0, 94.0, is_absorption=True)
    check("tidak ada setup (bias tidak sepakat)",
          generate_setups(bar3, structure_1h, structure_4h_down, daily_profile).empty)

    print("\n== 4. GAGAL: bias up tapi candle MERAH (bukan flip naik) ==")
    bar4 = make_bar(BAR_TIME, open_=95.0, high=95.5, low=92.0, close=94.0, is_absorption=True)
    check("tidak ada setup (candle merah, bukan konfirmasi long)",
          generate_setups(bar4, structure_1h, structure_4h, daily_profile).empty)

    print("\n== 5. GAGAL: bias up tapi harga BUKAN di discount (close > VAL) ==")
    bar5 = make_bar(BAR_TIME, open_=98.0, high=99.5, low=97.0, close=99.0, is_absorption=True)
    check("tidak ada setup (close 99 > VAL 95, bukan discount)",
          generate_setups(bar5, structure_1h, structure_4h, daily_profile).empty)

    print("\n== 6. GAGAL: sideways -> tidak ada bias sama sekali ==")
    structure_1h_side = structure_1h.copy()
    structure_1h_side[:] = "sideways"
    structure_4h_side = structure_4h.copy()
    structure_4h_side[:] = "sideways"
    bar6 = make_bar(BAR_TIME, 93.0, 94.5, 92.0, 94.0, is_absorption=True)
    check("tidak ada setup (sideways bukan bias)",
          generate_setups(bar6, structure_1h_side, structure_4h_side, daily_profile).empty)

    print("\n== 7. GAGAL: hari sebelumnya tidak punya profil (data belum cukup) ==")
    too_early = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")  # hari YANG SAMA dgn profil
    bar7 = make_bar(too_early, 93.0, 94.5, 92.0, 94.0, is_absorption=True)
    check("tidak ada setup (belum ada profil hari SEBELUM 2026-01-01)",
          generate_setups(bar7, structure_1h, structure_4h, daily_profile).empty)

    print("\n== 8. KENARI LOOK-AHEAD: mengubah VAL/POC hari YANG SAMA (belum final) ==")
    print("      tidak boleh mempengaruhi setup -- cuma hari SEBELUMNYA yang dipakai")
    daily_profile_2days = pd.concat([
        daily_profile,
        pd.DataFrame({"session_id": ["2026-01-02"], "val_price": [999.0],
                       "poc_price": [999.0], "vah_price": [999.0]}),
    ], ignore_index=True)
    bar8 = make_bar(BAR_TIME, 93.0, 94.5, 92.0, 94.0, is_absorption=True)
    setups8 = generate_setups(bar8, structure_1h, structure_4h, daily_profile_2days)
    check("target TETAP 100.0 (dari hari sebelumnya), BUKAN 999.0 (hari berjalan)",
          not setups8.empty and setups8.iloc[0]["target_price"] == 100.0,
          f"dapat={setups8.iloc[0]['target_price'] if not setups8.empty else 'kosong'}")

    print("\n" + "=" * 62)
    print("TARGET_MODE='swing' -- trial terpisah")
    print("=" * 62)

    print("\n== 9. Swing target: setup valid, target = swing high terkonfirmasi ==")
    # Bangun footprint 5-menit dengan swing high JELAS di bar index 5
    # (nilai tertinggi, dikelilingi nilai lebih rendah di kedua sisi),
    # lalu bar konfirmasi jauh setelahnya supaya swing itu sudah
    # terkonfirmasi (index+lookback <= posisi bar konfirmasi).
    idx_fp = pd.date_range("2026-01-02 00:00:00", periods=20, freq="5min", tz="UTC")
    highs = [90, 91, 92, 93, 94, 99, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 94.5]
    lows = [h - 1 for h in highs]
    closes = [h - 0.5 for h in highs]
    opens = [h - 0.6 for h in highs]
    is_absorption = [False] * 19 + [True]
    footprint_swing = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "is_absorption": is_absorption,
    }, index=idx_fp)
    # Bar terakhir (index 19): open < close (hijau), close 93.9 <= VAL 95 -> discount.
    footprint_swing.loc[idx_fp[19], ["open", "close"]] = [93.0, 93.9]

    setups_swing = generate_setups(
        footprint_swing, structure_1h, structure_4h, daily_profile,
        target_mode="swing", swing_lookback=3,
    )
    check("1 setup dihasilkan dengan target_mode=swing", len(setups_swing) == 1,
          f"dapat {len(setups_swing)}")
    if len(setups_swing) == 1:
        check("target = swing high di index 5 (harga 99), BUKAN prev_poc (100)",
              abs(setups_swing.iloc[0]["target_price"] - 99.0) < 1e-6,
              f"dapat {setups_swing.iloc[0]['target_price']}")

    print("\n== 10. KENARI LOOK-AHEAD swing: ubah bar SETELAH konfirmasi -> target sama ==")
    footprint_swing_b = footprint_swing.copy()
    # Bar konfirmasi (index 19) sudah yang TERAKHIR di sini, jadi kita
    # uji dengan menambah bar-bar BARU setelahnya yang membentuk swing
    # high lebih besar -- target untuk setup yang SUDAH dihasilkan tidak
    # boleh berubah walau data baru ini ditambahkan.
    extra_idx = pd.date_range(idx_fp[-1] + pd.Timedelta(minutes=5), periods=10, freq="5min", tz="UTC")
    extra = pd.DataFrame({
        "open": [200]*10, "high": [201, 202, 500, 202, 201, 200, 199, 198, 197, 196],
        "low": [195]*10, "close": [200]*10, "is_absorption": [False]*10,
    }, index=extra_idx)
    footprint_extended = pd.concat([footprint_swing_b, extra])
    setups_extended = generate_setups(
        footprint_extended, structure_1h, structure_4h, daily_profile,
        target_mode="swing", swing_lookback=3,
    )
    check("target TETAP 99.0 walau ada swing high 500 di masa depan",
          not setups_extended.empty and abs(setups_extended.iloc[0]["target_price"] - 99.0) < 1e-6,
          f"dapat={setups_extended.iloc[0]['target_price'] if not setups_extended.empty else 'kosong'}")
    print("    (kalau GAGAL, target swing mengintip masa depan -- bug serius)")

    print("\n== 11. Swing target: belum ada swing terkonfirmasi -> tidak ada setup ==")
    footprint_no_swing = footprint_swing.iloc[:6].copy()  # cuma sampai index 5, swing belum terkonfirmasi
    footprint_no_swing.loc[idx_fp[5], "is_absorption"] = True
    footprint_no_swing.loc[idx_fp[5], ["open", "close"]] = [93.0, 93.9]
    setups_none = generate_setups(
        footprint_no_swing, structure_1h, structure_4h, daily_profile,
        target_mode="swing", swing_lookback=3,
    )
    check("tidak ada setup (swing belum terkonfirmasi pada bar ini)", setups_none.empty)

    print("\n== 12. target_mode tidak valid -> ValueError ==")
    try:
        generate_setups(bar, structure_1h, structure_4h, daily_profile, target_mode="salah")
        check("melempar ValueError untuk target_mode tidak dikenal", False, "tidak melempar")
    except ValueError:
        check("melempar ValueError untuk target_mode tidak dikenal", True)

    print("\n" + "=" * 62)
    print("require_two_attempts=True -- trial terpisah lagi")
    print("=" * 62)

    def make_multi_bar(rows, is_absorption_list, start="2026-01-02 00:00:00"):
        idx = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
        df["is_absorption"] = is_absorption_list
        return df

    print("\n== 13. Dua percobaan: percobaan ke-2 dengan low LEBIH TINGGI -> setup ==")
    rows = [
        (93.0, 94.5, 92.0, 94.0),   # percobaan 1: low=92
        (94.0, 94.2, 93.8, 94.1),   # netral, diabaikan (bukan absorption)
        (93.5, 94.8, 93.0, 94.3),   # percobaan 2: low=93 > 92 -> HARUS jadi setup
    ]
    fp2 = make_multi_bar(rows, [True, False, True])
    setups_2att = generate_setups(
        fp2, structure_1h, structure_4h, daily_profile,
        require_two_attempts=True, two_attempt_max_gap_bars=48,
    )
    check("1 setup dihasilkan (dari percobaan KEDUA)", len(setups_2att) == 1,
          f"dapat {len(setups_2att)}")
    if len(setups_2att) == 1:
        check("entry dari bar percobaan KEDUA (94.3), bukan pertama (94.0)",
              abs(setups_2att.iloc[0]["entry_price"] - 94.3) < 1e-6,
              f"dapat {setups_2att.iloc[0]['entry_price']}")

    print("\n== 14. Cuma SATU percobaan (tidak ada percobaan kedua) -> tidak ada setup ==")
    rows_one = [(93.0, 94.5, 92.0, 94.0)]
    fp_one = make_multi_bar(rows_one, [True])
    setups_one = generate_setups(
        fp_one, structure_1h, structure_4h, daily_profile, require_two_attempts=True,
    )
    check("tidak ada setup (baru 1 percobaan, belum ada percobaan kedua)", setups_one.empty)

    print("\n== 15. Percobaan ke-2 dengan low LEBIH RENDAH (penjual menang) -> RESET, bukan setup ==")
    rows_reset = [
        (93.0, 94.5, 92.0, 94.0),   # percobaan 1: low=92
        (92.0, 92.5, 90.0, 92.3),   # low=90 < 92 -> penjual MENANG, reset (bukan setup)
        (91.5, 92.8, 91.0, 92.5),   # percobaan 1 BARU (low=91), bukan percobaan ke-2 dari yang lama
    ]
    fp_reset = make_multi_bar(rows_reset, [True, True, True])
    setups_reset = generate_setups(
        fp_reset, structure_1h, structure_4h, daily_profile, require_two_attempts=True,
    )
    check("tidak ada setup dari bar ke-2 (low lebih rendah = reset, bukan konfirmasi)",
          setups_reset.empty or setups_reset.index[0] != fp_reset.index[1],
          f"setup index: {list(setups_reset.index) if not setups_reset.empty else []}")

    print("\n== 16. Percobaan kedua di luar max_gap_bars -> kedaluarsa, tidak ada setup ==")
    rows_gap = [(93.0, 94.5, 92.0, 94.0)] + [(94.0, 94.2, 93.8, 94.1)] * 10 + [(93.5, 94.8, 93.0, 94.3)]
    fp_gap = make_multi_bar(rows_gap, [True] + [False] * 10 + [True])
    setups_gap = generate_setups(
        fp_gap, structure_1h, structure_4h, daily_profile,
        require_two_attempts=True, two_attempt_max_gap_bars=5,
    )
    check("tidak ada setup (percobaan pertama sudah kedaluarsa)", setups_gap.empty)

    print("\n== 17. require_two_attempts=False (default) TIDAK berubah ==")
    setups_default = generate_setups(fp2, structure_1h, structure_4h, daily_profile)
    check("2 setup (kedua bar absorption langsung jadi setup, bukan cuma yang kedua)",
          len(setups_default) == 2, f"dapat {len(setups_default)}")

    print("\n" + "=" * 62)
    print("min_avg_volume -- trial terpisah lagi")
    print("=" * 62)

    from src.microstructure.signal_generator import compute_participation_threshold

    print("\n== 18. Kalibrasi ambang partisipasi sesuai rolling mean SEBELUM bar (dihitung tangan) ==")
    vols = pd.Series([10.0] * 5 + [100.0] * 5 + [10.0] * 5)
    df_vol = pd.DataFrame({"volume": vols})
    th_vol = compute_participation_threshold(df_vol, window_bars=3, percentile=0.5)
    expected_roll = vols.shift(1).rolling(3, min_periods=3).mean().dropna()
    check("ambang = median rolling mean SEBELUM bar (shift(1) sebelum rolling)",
          abs(th_vol - float(expected_roll.quantile(0.5))) < 1e-9,
          f"dapat={th_vol}")

    print("\n== 19. Setup di rezim SEPI dibuang walau bar itu sendiri absorption ==")
    rows_quiet = [
        (100.0, 100.1, 100.0, 100.05, 5.0),   # volume amat kecil sekitarnya
        (100.0, 100.1, 100.0, 100.05, 5.0),
        (100.0, 100.1, 100.0, 100.05, 5.0),
        (93.0, 94.5, 92.0, 94.0, 6.0),          # bar absorption, tapi rezim sekitarnya tetap sepi
    ]
    idx_q = pd.date_range("2026-01-02", periods=4, freq="5min", tz="UTC")
    fp_quiet = pd.DataFrame(rows_quiet, columns=["open", "high", "low", "close", "volume"], index=idx_q)
    fp_quiet["is_absorption"] = [False, False, False, True]
    setups_quiet = generate_setups(
        fp_quiet, structure_1h, structure_4h, daily_profile,
        min_avg_volume=50.0, avg_volume_window=3,
    )
    check("tidak ada setup (rezim volume di bawah ambang)", setups_quiet.empty)

    print("\n== 20. Rezim RAMAI -> setup tetap lolos seperti biasa ==")
    rows_busy = [
        (100.0, 100.1, 100.0, 100.05, 200.0),
        (100.0, 100.1, 100.0, 100.05, 200.0),
        (100.0, 100.1, 100.0, 100.05, 200.0),
        (93.0, 94.5, 92.0, 94.0, 200.0),
    ]
    fp_busy = pd.DataFrame(rows_busy, columns=["open", "high", "low", "close", "volume"], index=idx_q)
    fp_busy["is_absorption"] = [False, False, False, True]
    setups_busy = generate_setups(
        fp_busy, structure_1h, structure_4h, daily_profile,
        min_avg_volume=50.0, avg_volume_window=3,
    )
    check("1 setup dihasilkan (rezim volume di atas ambang)", len(setups_busy) == 1)

    print("\n== 21. min_avg_volume=None (default) -- perilaku lama TIDAK berubah, TIDAK butuh kolom volume ==")
    setups_no_filter = generate_setups(bar, structure_1h, structure_4h, daily_profile)
    check("tetap jalan tanpa kolom 'volume' sama sekali saat filter tidak dipakai",
          len(setups_no_filter) == 1)

    print("\n== 22. BUKTI PERBAIKAN BUG: bar konfirmasi volume TINGGI dikelilingi rezim SEPI -> DITOLAK ==")
    print("      (versi lama akan meloloskan ini -- volume bar itu sendiri ikut menaikkan rolling mean)")
    rows_spike = [
        (100.0, 100.1, 100.0, 100.05, 5.0),   # konteks sepi
        (100.0, 100.1, 100.0, 100.05, 5.0),
        (100.0, 100.1, 100.0, 100.05, 5.0),
        (93.0, 94.5, 92.0, 94.0, 5000.0),       # bar absorption, volume BESAR -- tapi konteks TETAP sepi
    ]
    fp_spike = pd.DataFrame(rows_spike, columns=["open", "high", "low", "close", "volume"], index=idx_q)
    fp_spike["is_absorption"] = [False, False, False, True]
    setups_spike = generate_setups(
        fp_spike, structure_1h, structure_4h, daily_profile,
        min_avg_volume=50.0, avg_volume_window=3,
    )
    check("DITOLAK -- volume bar konfirmasi sendiri tidak ikut dihitung ke rata-rata",
          setups_spike.empty, f"setup dihasilkan: {len(setups_spike)} (harus 0)")

    print("\n" + "=" * 62)
    print("require_fib_zone=True -- trial terpisah lagi")
    print("=" * 62)

    print("\n== 23. Zona Fib dihitung benar dari swing low->high (dihitung tangan) ==")
    # 30 bar DATAR (150) kecuali dua titik sengaja: swing low BERSIH di
    # index 5 (harga 100), swing high BERSIH di index 15 (harga 200).
    # Datar di sekitarnya supaya TIDAK ADA swing lain yang tercipta
    # tanpa sengaja (versi sebelumnya salah karena pola dasarnya naik-
    # turun, menciptakan swing tambahan yang tidak diinginkan).
    from src.microstructure.signal_generator import _fib_zone_uptrend
    n_bars_fib = 30
    idx_fib = pd.date_range("2026-01-02", periods=n_bars_fib, freq="5min", tz="UTC")
    fp_fib_swings = pd.DataFrame({"high": [150.0] * n_bars_fib, "low": [149.0] * n_bars_fib}, index=idx_fib)
    fp_fib_swings.loc[idx_fib[5], ["high", "low"]] = [100.5, 99.5]    # swing low BERSIH di 100
    fp_fib_swings.loc[idx_fib[15], ["high", "low"]] = [200.5, 199.5]  # swing high BERSIH di 200

    swings_fib = find_swing_points(fp_fib_swings, lookback=3)
    zone = _fib_zone_uptrend(swings_fib, as_of_pos=25, lookback=3)
    # Zona = [200-0.886*100, 200-0.705*100] = [111.4, 129.5]
    check("zona Fib = [111.4, 129.5] (dihitung tangan)",
          zone is not None and abs(zone[0] - 111.4) < 1.0 and abs(zone[1] - 129.5) < 1.0,
          f"dapat {zone}")

    print("\n== 24. Setup valid: close DI DALAM zona Fib DAN di bawah VAL -> setup ==")
    # Baseline DATAR di 95 (bukan 150 seperti tes 23) -- supaya swing
    # high di 100 (index 15) genuinely LEBIH TINGGI dari baseline dan
    # terdeteksi sebagai swing high, bukan malah ikut terbaca sebagai
    # swing low tambahan (itu bug di versi tes sebelumnya).
    fp_fib_swings2 = pd.DataFrame({"high": [95.0] * n_bars_fib, "low": [94.0] * n_bars_fib}, index=idx_fib)
    fp_fib_swings2.loc[idx_fib[5], ["high", "low"]] = [90.5, 89.5]     # swing low di 90
    fp_fib_swings2.loc[idx_fib[15], ["high", "low"]] = [100.5, 99.5]   # swing high di 100
    # Zona = [100-0.886*10, 100-0.705*10] = [91.14, 92.95]
    fp_fib_swings2["open"] = fp_fib_swings2["low"]
    fp_fib_swings2["close"] = fp_fib_swings2["high"]
    fp_fib_swings2["is_absorption"] = False
    # bar konfirmasi di index 25: close=92.0 (DI DALAM zona [91.14,92.95] DAN <=VAL 95)
    fp_fib_swings2.loc[idx_fib[25], ["open", "close", "high", "low"]] = [91.0, 92.0, 92.2, 90.8]
    fp_fib_swings2.loc[idx_fib[25], "is_absorption"] = True

    setups_fib_ok = generate_setups(
        fp_fib_swings2, structure_1h, structure_4h, daily_profile,
        require_fib_zone=True, swing_lookback=3,
    )
    check("1 setup dihasilkan (di dalam zona Fib DAN di bawah VAL)",
          len(setups_fib_ok) == 1, f"dapat {len(setups_fib_ok)}")

    print("\n== 25. GAGAL: di bawah VAL tapi DI LUAR zona Fib -> tidak ada setup ==")
    fp_fib_bad = fp_fib_swings2.copy()
    # close=80 -- di bawah VAL(95) DAN di bawah zona Fib [91.14,92.95] (terlalu dalam)
    fp_fib_bad.loc[idx_fib[25], ["open", "close", "high", "low"]] = [79.0, 80.0, 80.2, 78.8]
    setups_fib_bad = generate_setups(
        fp_fib_bad, structure_1h, structure_4h, daily_profile,
        require_fib_zone=True, swing_lookback=3,
    )
    check("tidak ada setup (di bawah VAL tapi retracement terlalu dalam, di luar zona Fib)",
          setups_fib_bad.empty)

    print("\n== 26. GAGAL: pasangan swing low->high belum terbentuk -> tidak ada setup ==")
    fp_fib_none = fp_fib_swings2.iloc[:9].copy()  # cuma sampai sebelum swing high (index 15) ada
    fp_fib_none.loc[idx_fib[6], "is_absorption"] = True
    fp_fib_none.loc[idx_fib[6], ["open", "close"]] = [91.0, 92.0]
    setups_fib_none = generate_setups(
        fp_fib_none, structure_1h, structure_4h, daily_profile,
        require_fib_zone=True, swing_lookback=3,
    )
    check("tidak ada setup (belum ada pasangan swing low->high yang valid)", setups_fib_none.empty)

    print("\n== 27. require_fib_zone=False (default) -- perilaku lama TIDAK berubah ==")
    setups_no_fib = generate_setups(bar, structure_1h, structure_4h, daily_profile)
    check("tetap 1 setup seperti semula, tidak butuh swing sama sekali",
          len(setups_no_fib) == 1)

    print("\n" + "=" * 62)
    print("bias_mode='none' -- strategi mean-reversion, TANPA syarat bias sama sekali")
    print("=" * 62)

    print("\n== 28. LONG mean-reversion: discount+hijau, TANPA struktur 1H/4H apa pun ==")
    idx_mr = pd.date_range("2026-01-02", periods=1, freq="5min", tz="UTC")
    bar_mr_long = pd.DataFrame({
        "open": [93.0], "high": [94.5], "low": [92.0], "close": [94.0],
        "is_absorption": [True],
    }, index=idx_mr)
    empty_structure = pd.Series([], dtype=object, index=pd.DatetimeIndex([], tz="UTC"))  # SENGAJA kosong -- tidak ada bias
    setups_mr_long = generate_setups(
        bar_mr_long, empty_structure, empty_structure, daily_profile, bias_mode="none",
    )
    check("1 setup LONG dihasilkan walau structure_1h/4h KOSONG total",
          len(setups_mr_long) == 1 and setups_mr_long.iloc[0]["direction"] == "long",
          f"dapat {len(setups_mr_long)} setup")

    print("\n== 29. SHORT mean-reversion: premium+merah, TANPA struktur apa pun ==")
    bar_mr_short = pd.DataFrame({
        "open": [107.0], "high": [108.0], "low": [105.5], "close": [106.0],
        "is_absorption": [True],
    }, index=idx_mr)
    setups_mr_short = generate_setups(
        bar_mr_short, empty_structure, empty_structure, daily_profile, bias_mode="none",
    )
    check("1 setup SHORT dihasilkan walau structure_1h/4h KOSONG total",
          len(setups_mr_short) == 1 and setups_mr_short.iloc[0]["direction"] == "short",
          f"dapat {len(setups_mr_short)} setup")

    print("\n== 30. bias_mode='trend' (default) TETAP butuh struktur -- structure kosong -> tidak ada setup ==")
    setups_trend_empty = generate_setups(
        bar_mr_long, empty_structure, empty_structure, daily_profile,
    )  # bias_mode default = "trend"
    check("tidak ada setup (bias_mode='trend' butuh struktur, tidak ada di sini)",
          setups_trend_empty.empty)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())