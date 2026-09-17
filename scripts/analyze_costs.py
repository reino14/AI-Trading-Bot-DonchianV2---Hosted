"""
scripts/analyze_costs.py

HARI 1 -- skrip yang menentukan apakah proyek ini masuk akal.

Menghitung, untuk tiap instrumen kandidat:
  - berapa besar harga bergerak dalam satu window (rata-rata dan sebarannya)
  - berapa ongkos satu transaksi bolak-balik
  - perbandingan keduanya

Kalau ongkos lebih besar dari pergerakan tipikal, tidak ada strategi yang bisa
menutupnya. Berhenti di sini lebih baik daripada tahu tiga bulan lagi.

Cara pakai:
    python -m scripts.analyze_costs BTC/USDT:USDT ETH/USDT:USDT
    python -m scripts.analyze_costs BTC/USDT:USDT --window 30
    python -m scripts.analyze_costs BTC/USDT ETH/USDT --preset crypto_spot --window 120
"""

import argparse

import numpy as np
import pandas as pd

from src.core.cost_model import PRESETS, CostConfig, round_trip_cost
from src.data.store import load_bars


def movement_stats(df: pd.DataFrame, window_minutes: int = 1, skip_resample: bool = False) -> dict:
    """
    Ukur seberapa jauh harga bergerak dari satu bar ke bar berikutnya,
    pada resolusi window_minutes (default 1 menit = data mentah, tanpa resample).

    skip_resample: True kalau df SUDAH di resolusi target (mis. diunduh
    langsung harian) -- lewati langkah resample, TAPI window_minutes
    TETAP harus mencerminkan durasi bar SEBENARNYA (mis. 1440 untuk
    harian), supaya skala vol_daily_pct tetap benar. Ini yang tadinya
    salah: memaksa window_minutes=1 sekaligus melewati resample bikin
    volatilitas ke-skala seolah bar-nya 1 menit, padahal 1440 menit.
    """
    if window_minutes > 1 and not skip_resample:
        df = (
            df.resample(f"{window_minutes}min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

    returns_bps = (df["close"].pct_change() * 10_000).dropna()
    abs_bps = returns_bps.abs()
    bar_range_bps = ((df["high"] - df["low"]) / df["close"] * 10_000).dropna()
    bars_per_day = (60 * 24) / window_minutes

    return {
        "bars": len(df),
        "window_minutes": window_minutes,
        "mean_abs_bps": abs_bps.mean(),
        "median_abs_bps": abs_bps.median(),
        "p75_abs_bps": abs_bps.quantile(0.75),
        "p90_abs_bps": abs_bps.quantile(0.90),
        "mean_range_bps": bar_range_bps.mean(),
        "vol_daily_pct": returns_bps.std() * np.sqrt(bars_per_day) / 100,
    }


#: Durasi satu bar dalam MENIT, per timeframe ccxt -- dipakai supaya
#: skala volatilitas tetap benar walau data sudah di resolusi target
#: (bukan hasil resample dari 1 menit).
_TIMEFRAME_TO_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440, "1w": 10_080,
}


def evaluate(symbol: str, cfg: CostConfig, window_minutes: int = 1, source_timeframe: str = "1m") -> dict:
    load_timeframe = None if source_timeframe == "1m" else source_timeframe
    df = load_bars(symbol, timeframe=load_timeframe)

    # Kalau data SUDAH diunduh langsung di resolusi target (mis. harian),
    # JANGAN resample lagi -- TAPI window_minutes yang dipakai untuk skala
    # (vol_daily_pct, hold_minutes) WAJIB durasi bar SEBENARNYA (1440 untuk
    # harian), bukan 1 -- kalau dipaksa 1, volatilitas ke-skala seolah bar
    # 1 menit, salah ~38x untuk kasus harian (bug yang ditemukan di lapangan).
    skip_resample = load_timeframe is not None
    effective_window = _TIMEFRAME_TO_MINUTES.get(source_timeframe, window_minutes) if skip_resample else window_minutes

    stats = movement_stats(df, window_minutes=effective_window, skip_resample=skip_resample)

    agresif = round_trip_cost(
        cfg, entry_is_maker=False, exit_is_maker=False, hold_minutes=effective_window
    )
    sabar = round_trip_cost(
        cfg, entry_is_maker=True, exit_is_maker=False, hold_minutes=effective_window
    )

    return {
        "symbol": symbol,
        **stats,
        "cost_taker_bps": agresif.total_bps,
        "cost_maker_entry_bps": sabar.total_bps,
        "ratio_taker": stats["p75_abs_bps"] / agresif.total_bps,
        "ratio_maker_entry": stats["p75_abs_bps"] / sabar.total_bps,
    }


def print_report(
    results: list[dict], cfg: CostConfig, window_minutes: int, preset_name: str, source_timeframe: str = "1m"
) -> None:
    print("\n" + "=" * 78)
    if source_timeframe != "1m":
        label = f"PER BAR ({source_timeframe})"
    else:
        label = "PER MENIT" if window_minutes == 1 else f"PER {window_minutes} MENIT"
    print(f"PERGERAKAN HARGA {label} vs ONGKOS TRANSAKSI  (preset: {preset_name})")
    print("=" * 78)
    print("\nSemua angka dalam bps (1 bps = 0,01%)\n")

    for r in results:
        print(f"  {r['symbol']}")
        print(f"    Jumlah bar dianalisis        : {r['bars']:,}")
        print(f"    Gerak per window, median     : {r['median_abs_bps']:.2f} bps")
        print(f"    Gerak per window, persentil 75: {r['p75_abs_bps']:.2f} bps")
        print(f"    Gerak per window, persentil 90: {r['p90_abs_bps']:.2f} bps")
        print(f"    Rentang dalam bar (high-low) : {r['mean_range_bps']:.2f} bps")
        print(f"    Volatilitas harian           : {r['vol_daily_pct']:.2f}%")
        print()
        print(f"    Ongkos bolak-balik, taker    : {r['cost_taker_bps']:.2f} bps")
        print(f"    Ongkos bolak-balik, limit    : {r['cost_maker_entry_bps']:.2f} bps")
        print()
        print(f"    RASIO gerak/ongkos (taker)   : {r['ratio_taker']:.2f}x")
        print(f"    RASIO gerak/ongkos (limit)   : {r['ratio_maker_entry']:.2f}x")

        best = max(r["ratio_taker"], r["ratio_maker_entry"])
        if best < 1.0:
            verdict = "MUSTAHIL - ongkos lebih besar dari pergerakan tipikal"
        elif best < 2.0:
            verdict = "SANGAT SULIT - margin terlalu tipis, cari instrumen lain"
        elif best < 4.0:
            verdict = "BISA DIUSAHAKAN - lanjut ke hari 2"
        else:
            verdict = "RUANG CUKUP LEBAR - periksa likuiditasnya, jangan-jangan tipis"
        print(f"    >>> {verdict}")
        print("-" * 78)

    print("\nCARA MEMBACA ANGKA INI\n")
    print("  Rasio adalah pergerakan persentil-75 dibagi ongkos bolak-balik.")
    print(f"  Artinya: pada 25% window {window_minutes} menit paling bergerak,")
    print("  seberapa besar pergerakannya dibanding ongkos yang harus dibayar.\n")
    print("  Rasio 2x berarti strategi harus benar cukup sering untuk")
    print("  menutup ongkos, dan itu belum memperhitungkan bahwa tebakan")
    print("  arahnya juga harus benar. Rasio di bawah 2 hampir selalu")
    print("  berujung rugi setelah biaya.\n")

    if not cfg.slippage_is_measured:
        print("  PERHATIAN: angka slippage di cost_model masih ASUMSI")
        print(f"  ({cfg.slippage_bps:.1f} bps), belum diukur dari transaksi nyata.")
        print("  Rasio di atas kemungkinan masih terlalu optimistis.")
        print("  Angka ini diperbarui di hari 7, setelah live pertama.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandingkan pergerakan harga vs ongkos")
    parser.add_argument("symbols", nargs="+", help="contoh: BTC/USDT:USDT ETH/USDT:USDT")
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="ukuran window dalam menit untuk mengukur pergerakan harga, default 1",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="crypto_perp",
        help="preset ongkos dari src/core/cost_model.py -- 'crypto_perp' (default, futures) "
        "atau 'crypto_spot' (fee spot Binance, tanpa funding)",
    )
    parser.add_argument(
        "--source-timeframe",
        default="1m",
        help="resolusi FILE yang dibaca (bukan resample) -- default '1m' (perilaku lama). "
        "Isi 'lain, mis. '1d', kalau sudah unduh data di resolusi itu lewat "
        "src/data/fetch_history.py --timeframe 1d -- --window diabaikan kalau ini bukan '1m' "
        "(data dipakai apa adanya, tidak di-resample lagi).",
    )
    args = parser.parse_args()

    cfg = PRESETS[args.preset]

    results = []
    for symbol in args.symbols:
        try:
            results.append(
                evaluate(symbol, cfg, window_minutes=args.window, source_timeframe=args.source_timeframe)
            )
        except FileNotFoundError as e:
            print(f"\n{e}\n")

    if results:
        label_window = 1 if args.source_timeframe != "1m" else args.window
        print_report(
            results, cfg, window_minutes=label_window, preset_name=args.preset,
            source_timeframe=args.source_timeframe,
        )
        if args.source_timeframe != "1m":
            print(f"  (Catatan: data dibaca langsung dari resolusi '{args.source_timeframe}', "
                  f"--window diabaikan.)\n")


if __name__ == "__main__":
    main()