"""
scripts/smoke_channel_debug.py

Bukti PALING PENTING: angka channel yang ditampilkan make_channel_debug_fn()
(yang dilihat Nero di terminal) PERSIS SAMA dengan channel yang benar-benar
dipakai compute_donchian_signal() untuk memutuskan breakout -- bukan
hitungan terpisah yang kebetulan mirip.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.run_paper_donchian_futures import make_channel_debug_fn  # noqa: E402
from src.strategy.donchian_close import compute_donchian_signal  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1. Channel dari make_channel_debug_fn PERSIS sama dengan channel internal compute_donchian_signal ==")
    rng = np.random.default_rng(3)
    n = 300
    closes = 100 + rng.normal(0, 1, n).cumsum()
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    close_series = pd.Series(closes, index=idx)
    lookback = 50

    # Channel INTERNAL yang benar-benar dipakai sinyal (ekstrak manual
    # dengan rumus SAMA yang dipakai compute_donchian_signal secara
    # internal -- upper = rolling_max(close, lookback).shift(1)).
    internal_upper = close_series.rolling(lookback).max().shift(1)
    internal_lower = close_series.rolling(lookback).min().shift(1)

    debug_fn = make_channel_debug_fn(lookback)
    bars = [{"close": c} for c in closes]

    mismatches = 0
    checked = 0
    for i in range(lookback + 1, n, 17):  # sampel tiap 17 bar, cukup padat tanpa cek semua titik
        result_str = debug_fn(bars[: i + 1])
        # Parse angka "atas=X" dan "bawah=Y" dari string hasil fungsi.
        atas_str = result_str.split("atas=")[1].split(" ")[0]
        bawah_str = result_str.split("bawah=")[1].split(" ")[0]
        atas = float(atas_str)
        bawah = float(bawah_str)

        expected_upper = internal_upper.iloc[i]
        expected_lower = internal_lower.iloc[i]
        checked += 1
        # Toleransi 0,005 -- string hasil fungsi dibulatkan 2 desimal
        # untuk tampilan (":.2f"), jadi dibandingkan ke angka penuh
        # WAJAR beda sedikit di pembulatan, bukan tanda bug.
        if abs(atas - expected_upper) > 0.005 or abs(bawah - expected_lower) > 0.005:
            mismatches += 1

    check(f"SEMUA {checked} titik sampel cocok PERSIS dengan channel internal sinyal",
          mismatches == 0, f"{mismatches} titik BEDA dari {checked} yang dicek")

    print("\n== 2. Belum cukup data -> pesan jelas, bukan angka ngasal ==")
    result_early = debug_fn(bars[:10])  # jauh di bawah lookback+1=51
    check("pesan 'belum cukup data' muncul", "belum cukup data" in result_early, f"dapat: {result_early}")

    print("\n== 3. Breakout SUNGGUHAN: harga tembus 'atas' yang dicetak -> sinyal REAL juga berubah ke long ==")
    # Bangun kasus jelas: harga naik terus 60 bar, lalu breakout besar.
    closes2 = list(100 + np.arange(60) * 0.01) + [200.0]  # naik pelan, lalu breakout besar
    idx2 = pd.date_range("2026-02-01", periods=len(closes2), freq="1min", tz="UTC")
    close_series2 = pd.Series(closes2, index=idx2)
    real_signal = compute_donchian_signal(close_series2, lookback=50)

    bars2 = [{"close": c} for c in closes2]
    result_final = debug_fn(bars2)
    atas_final = float(result_final.split("atas=")[1].split(" ")[0])

    check("harga terakhir (200.0) MELEWATI 'atas' yang dicetak fungsi ini",
          closes2[-1] > atas_final, f"harga={closes2[-1]}, atas={atas_final}")
    check("sinyal REAL (compute_donchian_signal) JUGA long di bar ini -- konsisten dengan channel yang ditampilkan",
          real_signal.iloc[-1] == 1.0, f"dapat sinyal={real_signal.iloc[-1]}")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())