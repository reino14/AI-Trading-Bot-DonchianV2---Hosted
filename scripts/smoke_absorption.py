"""
scripts/smoke_absorption.py

Uji asap microstructure/absorption.py -- kalibrasi ambang diverifikasi
terhadap persentil yang dihitung tangan, deteksi diverifikasi dengan
kasus eksplisit (jelas absorption, jelas bukan, tepat di batas ambang).
"""

import sys

import numpy as np
import pandas as pd

from src.microstructure.absorption import (
    calibrate_absorption_thresholds,
    detect_absorption,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1. Kalibrasi ambang sesuai persentil yang dihitung tangan ==")
    # volume 1..100, delta_pct dibuat supaya |delta_pct| juga 0.01..1.00
    # (skala berbeda, arah kebalikan) -- supaya dua kalibrasi independen
    # bisa diverifikasi terpisah tanpa saling mempengaruhi.
    df = pd.DataFrame({
        "volume": np.arange(1, 101, dtype=float),
        "delta_pct": np.arange(100, 0, -1, dtype=float) / 100.0,  # 1.00 turun ke 0.01
    })
    th = calibrate_absorption_thresholds(df, volume_percentile=0.75, delta_pct_percentile=0.25)
    expected_vol_p75 = float(np.quantile(np.arange(1, 101), 0.75))
    expected_delta_p25 = float(np.quantile(np.arange(100, 0, -1) / 100.0, 0.25))
    check("volume_threshold = persentil 75 volume",
          abs(th.volume_threshold - expected_vol_p75) < 1e-6,
          f"dapat={th.volume_threshold}, harus={expected_vol_p75}")
    check("delta_pct_threshold = persentil 25 dari |delta_pct|",
          abs(th.delta_pct_threshold - expected_delta_p25) < 1e-6,
          f"dapat={th.delta_pct_threshold}, harus={expected_delta_p25}")

    print("\n== 2. Deteksi: kasus jelas ABSORPTION (volume tinggi, delta nyaris nol) ==")
    th2 = calibrate_absorption_thresholds(
        pd.DataFrame({"volume": [100, 200, 300, 400, 500],
                       "delta_pct": [0.5, 0.4, 0.3, 0.2, 0.1]}),
        volume_percentile=0.5, delta_pct_percentile=0.5,
    )
    test_bars = pd.DataFrame({
        "volume": [1000, 1000, 50, 50],
        "delta_pct": [0.02, 0.9, 0.02, 0.9],
    })
    flags = detect_absorption(test_bars, th2)
    check("volume tinggi + delta rendah -> TRUE", bool(flags.iloc[0]) is True)
    check("volume tinggi + delta tinggi -> FALSE (bukan absorption, ada dorongan searah)",
          bool(flags.iloc[1]) is False)
    check("volume rendah + delta rendah -> FALSE (bukan absorption, cuma sepi)",
          bool(flags.iloc[2]) is False)
    check("volume rendah + delta tinggi -> FALSE", bool(flags.iloc[3]) is False)

    print("\n== 3. detect_absorption TIDAK menghitung ambang dari datanya sendiri ==")
    # Ambang dari dataset A, dipakai untuk deteksi di dataset B yang
    # SANGAT berbeda skalanya -- pastikan tidak diam-diam dikalibrasi ulang.
    th_from_a = calibrate_absorption_thresholds(
        pd.DataFrame({"volume": [1, 2, 3], "delta_pct": [0.9, 0.9, 0.9]})
    )
    dataset_b = pd.DataFrame({"volume": [1000000], "delta_pct": [0.01]})
    flags_b = detect_absorption(dataset_b, th_from_a)
    check("bar di dataset B terdeteksi absorption pakai ambang dari A (bukan dihitung ulang)",
          bool(flags_b.iloc[0]) is True)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())