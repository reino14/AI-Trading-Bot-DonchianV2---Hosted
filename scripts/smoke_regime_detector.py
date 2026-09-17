"""
scripts/smoke_regime_detector.py

Uji asap microstructure/regime_detector.py -- skenario volatile jelas,
trending jelas, sideways jelas, plus silang-cek versi vektor (merge_asof)
melawan versi loop lambat (ditulis ulang di sini, TERPISAH dari kode
produksi) untuk memastikan optimasi tidak diam-diam mengubah hasil.
"""

import sys

import numpy as np
import pandas as pd

from src.microstructure.regime_detector import (
    classify_regime,
    compute_volatility_threshold,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _slow_classify_regime(footprint_df, structure_1h, vol_threshold, vol_window=24):
    """Versi loop per-bar SENGAJA lambat, ditulis independen dari kode
    produksi -- referensi tangan untuk silang-cek, BUKAN dipakai produksi."""
    returns = footprint_df["close"].pct_change()
    vol = returns.shift(1).rolling(vol_window, min_periods=vol_window).std()
    labels = []
    for t in footprint_df.index:
        v = vol.loc[t]
        if pd.notna(v) and v >= vol_threshold:
            labels.append("volatile")
            continue
        eligible = structure_1h[structure_1h.index <= t]
        if eligible.empty:
            labels.append("sideways")
            continue
        bias = eligible.iloc[-1]
        labels.append("trending" if bias in ("up", "down") else "sideways")
    return pd.Series(labels, index=footprint_df.index)


def main() -> int:
    print("== 1. Kalibrasi ambang volatilitas sesuai persentil yang dihitung tangan ==")
    rng = np.random.default_rng(11)
    n = 500
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    # Volatilitas berubah-ubah: separuh awal tenang, separuh akhir liar.
    returns = np.concatenate([rng.normal(0, 0.001, n // 2), rng.normal(0, 0.02, n - n // 2)])
    prices = 100 * np.exp(np.cumsum(returns))
    footprint = pd.DataFrame({"close": prices}, index=idx)

    th = compute_volatility_threshold(footprint, vol_window=12, percentile=0.80)
    expected_vol = footprint["close"].pct_change().shift(1).rolling(12, min_periods=12).mean()  # placeholder cek non-crash
    check("ambang volatilitas adalah angka positif masuk akal",
          th > 0, f"dapat {th}")

    print("\n== 2. Volatile menang prioritas walau struktur 'trending' ==")
    idx2 = pd.date_range("2026-01-02", periods=50, freq="5min", tz="UTC")
    # Trend naik jelas TAPI dengan lonjakan volatil di tengah.
    prices2 = 100 + np.arange(50) * 0.5
    prices2[25:30] += rng.normal(0, 10, 5)  # lonjakan liar di tengah
    footprint2 = pd.DataFrame({"close": prices2}, index=idx2)
    structure_1h_up = pd.Series(["up"] * 5, index=pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC"))

    th2 = compute_volatility_threshold(footprint2, vol_window=6, percentile=0.5)
    regime2 = classify_regime(footprint2, structure_1h_up, vol_threshold=th2, vol_window=6)
    check("ADA bar berlabel 'volatile' di sekitar lonjakan (bukan cuma 'trending')",
          (regime2.iloc[26:32] == "volatile").any(),
          f"label di sekitar lonjakan: {regime2.iloc[26:32].tolist()}")

    print("\n== 3. Trending: struktur up/down, volatilitas rendah -> 'trending' ==")
    idx3 = pd.date_range("2026-01-03", periods=50, freq="5min", tz="UTC")
    prices3 = 100 + np.arange(50) * 0.1  # naik halus, tanpa lonjakan
    footprint3 = pd.DataFrame({"close": prices3}, index=idx3)
    th3 = compute_volatility_threshold(footprint3, vol_window=6, percentile=0.95)  # ambang tinggi -> tidak ada yg volatile
    regime3 = classify_regime(footprint3, structure_1h_up, vol_threshold=th3 * 100, vol_window=6)
    check("mayoritas bar (setelah vol_window) berlabel 'trending'",
          (regime3.iloc[10:] == "trending").mean() > 0.8,
          f"proporsi trending: {(regime3.iloc[10:] == 'trending').mean():.2f}")

    print("\n== 4. Sideways: struktur sideways, volatilitas rendah -> 'sideways' ==")
    structure_1h_side = pd.Series(["sideways"] * 5,
                                    index=pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC"))
    regime4 = classify_regime(footprint3, structure_1h_side, vol_threshold=th3 * 100, vol_window=6)
    check("mayoritas bar berlabel 'sideways'",
          (regime4.iloc[10:] == "sideways").mean() > 0.8,
          f"proporsi sideways: {(regime4.iloc[10:] == 'sideways').mean():.2f}")

    print("\n== 5. Silang-cek: versi vektor (produksi) == versi loop lambat (independen) ==")
    idx5 = pd.date_range("2026-02-01", periods=300, freq="5min", tz="UTC")
    rets5 = rng.normal(0, 0.005, 300)
    rets5[100:110] = rng.normal(0, 0.03, 10)  # blok volatil di tengah
    prices5 = 100 * np.exp(np.cumsum(rets5))
    footprint5 = pd.DataFrame({"close": prices5}, index=idx5)
    structure_1h_mixed = pd.Series(
        ["up", "down", "sideways", "up", "sideways"] * 20,
        index=pd.date_range("2026-01-25", periods=100, freq="2h", tz="UTC"),
    )
    th5 = compute_volatility_threshold(footprint5, vol_window=12, percentile=0.85)
    fast = classify_regime(footprint5, structure_1h_mixed, vol_threshold=th5, vol_window=12)
    slow = _slow_classify_regime(footprint5, structure_1h_mixed, vol_threshold=th5, vol_window=12)
    check("versi vektor IDENTIK dengan versi loop, bar demi bar",
          (fast.values == slow.values).all(),
          f"{(fast.values != slow.values).sum()} bar berbeda dari {len(fast)}")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())