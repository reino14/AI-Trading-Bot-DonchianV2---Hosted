"""
scripts/smoke_donchian_close_live.py

Bukti PALING PENTING: next_signal() (live, per-bar) menghasilkan
sinyal IDENTIK dengan compute_donchian_signal() (batch, sudah
divalidasi) di SETIAP bar -- bukan cuma "logikanya mirip di atas
kertas". Kalau tes ini gagal, JANGAN pakai versi live ini untuk
trading sungguhan/demo apa pun -- artinya ada penyimpangan halus dari
yang sudah diuji.
"""

import sys

import numpy as np
import pandas as pd

from src.strategy.donchian_close import compute_donchian_signal
from src.strategy.donchian_close_live import next_signal

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1. Live vs batch: identik bar-demi-bar di data acak, lookback=8 ==")
    rng = np.random.default_rng(7)
    n = 500
    closes = 100 + rng.normal(0, 1, n).cumsum()
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close_series = pd.Series(closes, index=idx)

    lookback = 8
    batch_signal = compute_donchian_signal(close_series, lookback=lookback)

    live_signal = [None] * n
    last = None
    for i in range(n):
        if i < lookback:
            continue  # belum cukup histori -- next_signal akan return None juga
        window = closes[max(0, i - lookback):i + 1]  # lookback+1 close terakhir sampai bar i
        result = next_signal(window, lookback=lookback, last_signal=last)
        live_signal[i] = result
        last = result

    mismatches = []
    for i in range(lookback, n):
        b = batch_signal.iloc[i]
        l = live_signal[i]
        b_val = None if pd.isna(b) else float(b)
        if b_val != l:
            mismatches.append((i, b_val, l))

    check("SEMUA bar (setelah lookback awal) identik antara live dan batch",
          len(mismatches) == 0,
          f"{len(mismatches)} bar berbeda dari {n-lookback}" +
          (f", contoh: {mismatches[:3]}" if mismatches else ""))

    print("\n== 2. Data kurang dari lookback+1 -> None, bukan tebakan ==")
    check("cuma 5 close, lookback=8 -> None", next_signal([100, 101, 102, 103, 104], lookback=8, last_signal=None) is None)

    print("\n== 3. Breakout naik terdeteksi dengan benar (dihitung tangan) ==")
    # 9 close: 8 bar channel [100..107], bar ke-9 (108) tembus ke atas.
    closes3 = [100, 101, 102, 103, 104, 105, 106, 107, 108]
    result3 = next_signal(closes3, lookback=8, last_signal=None)
    check("108 > max(100..107)=107 -> sinyal long (1.0)", result3 == 1.0, f"dapat {result3}")

    print("\n== 4. Tidak ada breakout -> posisi lama dipertahankan ==")
    closes4 = [100, 101, 102, 103, 104, 105, 106, 107, 103]  # 103 di tengah range, tidak breakout
    result4 = next_signal(closes4, lookback=8, last_signal=1.0)
    check("tidak breakout -> tetap 1.0 (posisi lama)", result4 == 1.0, f"dapat {result4}")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())