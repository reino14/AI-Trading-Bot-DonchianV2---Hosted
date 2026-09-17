"""
scripts/smoke_donchian_close.py

Uji asap trend_following/donchian_close.py -- sinyal, penyelarasan
waktu return, penyelarasan waktu ongkos (paling rawan salah di jenis
backtest always-in-market begini), dan kenari look-ahead.
"""

import sys

import numpy as np
import pandas as pd

from src.strategy.donchian_close import (
    compute_donchian_signal,
    compute_strategy_returns,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1. Sinyal: breakout jelas, setara rolling_max(close,lookback).shift(1) ==")
    # Harga naik terus dari 100 ke 110 (10 bar), lookback=3.
    # channel_upper[t] = max(close[t-3:t]) (3 bar SEBELUM t, shift eksplisit).
    closes = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                        index=pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC"))
    signal = compute_donchian_signal(closes, lookback=3)
    # Verifikasi tangan bar index3 (close=103): channel_upper = max(close[0:3])=max(100,101,102)=102.
    # 103 > 102 -> breakout up -> signal=1.
    check("bar index3: breakout up terdeteksi (103 > channel 102)",
          signal.iloc[3] == 1.0, f"dapat {signal.iloc[3]}")
    check("posisi bertahan (forward-fill) di bar-bar setelahnya (harga terus naik)",
          (signal.iloc[3:] == 1.0).all(), f"dapat {signal.iloc[3:].tolist()}")
    check("bar sebelum breakout pertama -> NaN (belum ada posisi, bukan ditebak)",
          signal.iloc[:3].isna().all())

    print("\n== 2. Return: signal[t-1] dikali log_return[t] (posisi baru mulai untung DI PERIODE BERIKUTNYA) ==")
    closes2 = pd.Series([100.0, 110.0, 121.0],
                         index=pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC"))
    signal2 = pd.Series([1.0, 1.0, 1.0], index=closes2.index)  # asumsi sudah long dari awal
    result2 = compute_strategy_returns(closes2, signal2, cost_bps_per_unit=0.0)
    # log_return index1 = ln(110/100). strategy_return index1 = signal[0](=1) * log_return[1].
    expected_ret1 = np.log(110 / 100)
    check("strategy_return bar1 = signal[bar0] * log_return[bar1] (BUKAN signal[bar1])",
          abs(result2["strategy_return_gross"].iloc[1] - expected_ret1) < 1e-9,
          f"dapat {result2['strategy_return_gross'].iloc[1]:.6f}, harus {expected_ret1:.6f}")

    print("\n== 3. Ongkos: dikenakan pada PERIODE YANG SAMA dengan return posisi yang baru dibuka ==")
    closes3 = pd.Series([100.0, 100.0, 100.0, 100.0],
                         index=pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC"))
    # Sinyal: flat(0) -> long(1) di bar1 -> tetap long -> reversal ke short(-1) di bar3.
    signal3 = pd.Series([0.0, 1.0, 1.0, -1.0], index=closes3.index)
    result3 = compute_strategy_returns(closes3, signal3, cost_bps_per_unit=10.0)
    # Perubahan sinyal: bar1 (0->1, |diff|=1), bar3 (1->-1, |diff|=2).
    # Ongkos bar1 (masuk long) harus muncul di cost bar2 (periode return posisi itu).
    # Ongkos bar3 (reversal) harus muncul di cost bar... tunggu, index terakhir=bar3,
    # tidak ada bar4 -- jadi ongkos reversal bar3 TIDAK PERNAH muncul di data ini (wajar,
    # posisi barunya belum sempat menghasilkan return apa pun).
    check("ongkos MASUK POSISI (bar1, |diff|=1) muncul di cost BAR 2, bukan bar1",
          abs(result3["cost"].iloc[2] - 10.0 / 10_000) < 1e-9,
          f"cost bar2={result3['cost'].iloc[2]:.6f}, harus {10.0/10_000:.6f}")
    check("cost bar1 itu sendiri masih 0 (ongkos belum 'kena' return, posisi baru mulai)",
          abs(result3["cost"].iloc[1]) < 1e-12, f"dapat {result3['cost'].iloc[1]}")

    print("\n== 4. KENARI LOOK-AHEAD: ubah close SETELAH suatu bar -> sinyal SEBELUMNYA tidak berubah ==")
    closes4 = pd.Series([100, 101, 102, 103, 104, 105, 106, 90, 95, 99],
                         index=pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC"), dtype=float)
    signal4_before = compute_donchian_signal(closes4, lookback=3)
    cutoff = 5
    label_before = signal4_before.iloc[cutoff]

    closes4_future = closes4.copy()
    closes4_future.iloc[cutoff + 1:] = 0.001  # ubah drastis SEMUA bar setelah cutoff
    signal4_after = compute_donchian_signal(closes4_future, lookback=3)
    label_after = signal4_after.iloc[cutoff]

    check("sinyal di titik cutoff SAMA walau semua data SETELAHNYA diubah total",
          (pd.isna(label_before) and pd.isna(label_after)) or label_before == label_after,
          f"sebelum={label_before}, sesudah={label_after}")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())