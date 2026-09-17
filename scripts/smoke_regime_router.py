"""
scripts/smoke_regime_router.py

Uji asap microstructure/regime_router.py -- bar trending diarahkan ke
konfigurasi trend, bar sideways diarahkan ke konfigurasi mean-reversion,
regime yang tidak didaftarkan di regime_configs tidak pernah trading.
"""

import sys

import pandas as pd

from src.microstructure.regime_router import route_by_regime

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    daily_profile = pd.DataFrame({
        "session_id": ["2026-01-01"], "val_price": [95.0],
        "poc_price": [100.0], "vah_price": [105.0],
    })

    print("== 1. Bar TRENDING (bias up, structure ada) -> lolos lewat config 'trending' ==")
    idx = pd.date_range("2026-01-02 00:00:00", periods=1, freq="5min", tz="UTC")
    footprint_trend = pd.DataFrame({
        "open": [93.0], "high": [94.5], "low": [92.0], "close": [94.0],
        "is_absorption": [True],
    }, index=idx)
    structure_1h_up = pd.Series(["up"], index=pd.DatetimeIndex(["2026-01-01 00:00:00"], tz="UTC"))
    structure_4h_up = pd.Series(["up"], index=pd.DatetimeIndex(["2026-01-01 00:00:00"], tz="UTC"))
    regime_trending = pd.Series(["trending"], index=idx)

    configs = {
        "trending": {"target_mode": "prev_poc", "bias_mode": "trend"},
        "sideways": {"target_mode": "prev_poc", "bias_mode": "none"},
    }
    result = route_by_regime(
        footprint_trend, structure_1h_up, structure_4h_up, daily_profile,
        regime_trending, configs,
    )
    check("1 setup, diberi label regime='trending'",
          len(result) == 1 and result.iloc[0]["regime"] == "trending",
          f"dapat {len(result)} setup")

    print("\n== 2. Bar SIDEWAYS (structure kosong/sideways) -> lolos lewat config 'sideways' (mean-reversion) ==")
    structure_1h_empty = pd.Series([], dtype=object, index=pd.DatetimeIndex([], tz="UTC"))
    structure_4h_empty = pd.Series([], dtype=object, index=pd.DatetimeIndex([], tz="UTC"))
    regime_sideways = pd.Series(["sideways"], index=idx)

    result2 = route_by_regime(
        footprint_trend, structure_1h_empty, structure_4h_empty, daily_profile,
        regime_sideways, configs,
    )
    check("1 setup, diberi label regime='sideways' (via bias_mode=none)",
          len(result2) == 1 and result2.iloc[0]["regime"] == "sideways",
          f"dapat {len(result2)} setup")

    print("\n== 3. Bar TRENDING tapi structure sebenarnya KOSONG -> config 'trending' TIDAK menghasilkan apa pun ==")
    # Regime BILANG 'trending' (label eksternal), tapi structure_1h yang
    # SUNGGUHAN kosong -- config 'trending' pakai bias_mode='trend' yang
    # butuh structure, jadi TIDAK menghasilkan setup sama sekali.
    # Ini membuktikan router tidak "memaksakan" regime, cuma menyaring
    # dari apa yang benar-benar dihasilkan generate_setups().
    result3 = route_by_regime(
        footprint_trend, structure_1h_empty, structure_4h_empty, daily_profile,
        regime_trending, configs,
    )
    check("tidak ada setup (config trend butuh structure sungguhan, bukan cuma label regime)",
          result3.empty)

    print("\n== 4. Regime TIDAK terdaftar di regime_configs -> tidak pernah trading ==")
    regime_volatile = pd.Series(["volatile"], index=idx)  # "volatile" TIDAK ada di `configs`
    result4 = route_by_regime(
        footprint_trend, structure_1h_up, structure_4h_up, daily_profile,
        regime_volatile, configs,
    )
    check("tidak ada setup (regime 'volatile' tidak didaftarkan)", result4.empty)

    print("\n== 5. Gabungan dua regime berbeda waktu -> urut waktu, dua-duanya muncul ==")
    idx5 = pd.date_range("2026-01-02 00:00:00", periods=2, freq="1h", tz="UTC")
    footprint5 = pd.DataFrame({
        "open": [93.0, 93.0], "high": [94.5, 94.5], "low": [92.0, 92.0], "close": [94.0, 94.0],
        "is_absorption": [True, True],
    }, index=idx5)
    regime5 = pd.Series(["trending", "sideways"], index=idx5)
    result5 = route_by_regime(
        footprint5, structure_1h_up, structure_4h_up, daily_profile, regime5, configs,
    )
    check("2 setup, urut waktu, label regime benar masing-masing",
          len(result5) == 2
          and list(result5["regime"]) == ["trending", "sideways"]
          and result5.index.is_monotonic_increasing,
          f"dapat {len(result5)} setup, label={list(result5['regime']) if not result5.empty else []}")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())