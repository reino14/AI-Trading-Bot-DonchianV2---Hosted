"""
scripts/run_capital_backtest.py

CATATAN PENTING SEBELUM DIPAKAI
----------------------------------
Aturan paling longgar yang sudah kita uji sejauh ini (target=prev_poc,
tanpa dua-percobaan) cuma hasilkan 61 setup dalam SETAHUN PENUH -- itu
~0,17/hari. Untuk sampai 1-2 trade/hari (~365-730/tahun), skrip ini
HARUS melonggarkan aturan 6-12x lebih longgar dari yang paling longgar
sekalipun. INI BUKAN "strategi yang sama, cuma lebih sering" --
konfigurasi yang berhasil mencapai frekuensi target adalah HIPOTESIS
BARU, terpisah dari trial `prev_poc+2attempt` yang menarik perhatian
Anda (n=13, tidak bisa dipercaya sama sekali).

Pola yang sudah berulang kali terbukti di proyek ini: makin longgar
aturan -> makin banyak trade -> hasilnya makin mendekati nol atau
negatif, BUKAN makin bagus. Jangan berharap dua trade beruntung dari
`prev_poc+2attempt` (+223bps, +177bps) ikut terbawa ke sini -- itu
kebetulan sampel kecil, bukan sifat strategi yang bisa "diperbesar".

CARA KERJA
-----------
Coba serangkaian konfigurasi, makin longgar tiap langkah, sampai ketemu
yang frekuensinya masuk rentang target (--target-min sampai
--target-max trade/hari). Konfigurasi yang KETEMU dicetak LENGKAP --
supaya Anda tahu PERSIS parameter apa yang menghasilkan angka ini,
bukan kotak hitam.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.cost_model import CRYPTO_PERP
from src.microstructure.absorption import calibrate_absorption_thresholds, detect_absorption
from src.microstructure.market_structure import build_structure_series
from src.microstructure.signal_generator import generate_setups
from src.microstructure.trade_simulator import simulate_trades

# Tangga konfigurasi, DARI PALING KETAT KE PALING LONGGAR. Tiap baris
# adalah HIPOTESIS TERPISAH -- kalau nanti melacak n_trials untuk gate
# statistik manapun, seluruh tangga yang DICOBA (bukan cuma yang
# dipakai untuk hasil akhir) ikut dihitung.
LADDER_DEFAULT = [
    {
        "label": "baseline (prev_poc + 2attempt, bias 1H&4H, absorption p75/p25)",
        "absorption_pct": (0.75, 0.25),
        "gen_kwargs": {"target_mode": "prev_poc", "require_two_attempts": True},
    },
    {
        "label": "absorption dilonggarkan p65/p35",
        "absorption_pct": (0.65, 0.35),
        "gen_kwargs": {"target_mode": "prev_poc", "require_two_attempts": True},
    },
    {
        "label": "bias cukup 1H saja (4H diabaikan)",
        "absorption_pct": (0.75, 0.25),
        "gen_kwargs": {"target_mode": "prev_poc", "require_two_attempts": True,
                        "require_both_timeframes_agree": False},
    },
    {
        "label": "bias 1H saja + absorption p65/p35",
        "absorption_pct": (0.65, 0.35),
        "gen_kwargs": {"target_mode": "prev_poc", "require_two_attempts": True,
                        "require_both_timeframes_agree": False},
    },
    {
        "label": "bias 1H saja + absorption p65/p35 + TANPA dua-percobaan",
        "absorption_pct": (0.65, 0.35),
        "gen_kwargs": {"target_mode": "prev_poc", "require_two_attempts": False,
                        "require_both_timeframes_agree": False},
    },
    {
        "label": "bias 1H saja + absorption p55/p45 + TANPA dua-percobaan",
        "absorption_pct": (0.55, 0.45),
        "gen_kwargs": {"target_mode": "prev_poc", "require_two_attempts": False,
                        "require_both_timeframes_agree": False},
    },
    {
        "label": "bias 1H saja + absorption p50/p50 (median split) + TANPA dua-percobaan",
        "absorption_pct": (0.50, 0.50),
        "gen_kwargs": {"target_mode": "prev_poc", "require_two_attempts": False,
                        "require_both_timeframes_agree": False},
    },
]

# Varian KEDUA: require_fib_zone=True DIPERTAHANKAN di semua anak
# tangga (tidak pernah dibuang) -- bias dan absorption yang dilonggarkan,
# BUKAN syarat Fibonacci-nya. Ini supaya tetap bisa disebut "trial Fib
# yang sama, sampelnya diperbesar", bukan strategi lain yang kebetulan
# menyertakan kata "fib" di confignya.
LADDER_FIB = [
    {
        "label": "baseline (prev_poc + fib_zone, bias 1H&4H, absorption p75/p25)",
        "absorption_pct": (0.75, 0.25),
        "gen_kwargs": {"target_mode": "prev_poc", "require_fib_zone": True},
    },
    {
        "label": "fib_zone + absorption dilonggarkan p65/p35",
        "absorption_pct": (0.65, 0.35),
        "gen_kwargs": {"target_mode": "prev_poc", "require_fib_zone": True},
    },
    {
        "label": "fib_zone + bias cukup 1H saja",
        "absorption_pct": (0.75, 0.25),
        "gen_kwargs": {"target_mode": "prev_poc", "require_fib_zone": True,
                        "require_both_timeframes_agree": False},
    },
    {
        "label": "fib_zone + bias 1H saja + absorption p65/p35",
        "absorption_pct": (0.65, 0.35),
        "gen_kwargs": {"target_mode": "prev_poc", "require_fib_zone": True,
                        "require_both_timeframes_agree": False},
    },
    {
        "label": "fib_zone + bias 1H saja + absorption p55/p45",
        "absorption_pct": (0.55, 0.45),
        "gen_kwargs": {"target_mode": "prev_poc", "require_fib_zone": True,
                        "require_both_timeframes_agree": False},
    },
    {
        "label": "fib_zone + bias 1H saja + absorption p50/p50 (median split)",
        "absorption_pct": (0.50, 0.50),
        "gen_kwargs": {"target_mode": "prev_poc", "require_fib_zone": True,
                        "require_both_timeframes_agree": False},
    },
]


def compute_equity_curve(trades: pd.DataFrame, initial_capital: float, position_fraction: float):
    capital = initial_capital
    equity = [capital]
    for _, tr in trades.iterrows():
        pnl = capital * position_fraction * (tr["net_pnl_bps"] / 10_000)
        capital += pnl
        equity.append(capital)
    equity = pd.Series(equity)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return capital, equity, float(drawdown.min())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--target-min", type=float, default=1.0, help="trade/hari minimum yang dicari")
    p.add_argument("--target-max", type=float, default=2.0, help="trade/hari maksimum yang dicari")
    p.add_argument("--initial-capital", type=float, default=10_000_000)
    p.add_argument("--position-fraction", type=float, default=1.0,
                    help="fraksi modal per trade -- 1.0 = 100%%, SANGAT AGRESIF, lihat peringatan di output")
    p.add_argument("--ladder", choices=["default", "fib"], default="default",
                    help="'default' = longgarkan semua syarat; 'fib' = require_fib_zone=True "
                         "DIPERTAHANKAN di semua anak tangga, cuma bias/absorption yang dilonggarkan")
    args = p.parse_args()
    LADDER = LADDER_FIB if args.ladder == "fib" else LADDER_DEFAULT

    data_dir = Path(args.data_dir)
    prefix = args.symbol.lower().replace("usdt", "")

    print(f"Memuat data {args.symbol}...")
    footprint = pd.read_parquet(data_dir / "tick" / f"{prefix}_footprint_5min.parquet")
    df_1h = pd.read_parquet(data_dir / "klines" / args.symbol / "1h.parquet").set_index("open_time")
    df_1h.columns = [c.lower() for c in df_1h.columns]
    structure_1h = build_structure_series(df_1h, lookback=3)
    df_4h = pd.read_parquet(data_dir / "klines" / args.symbol / "4h.parquet").set_index("open_time")
    df_4h.columns = [c.lower() for c in df_4h.columns]
    structure_4h = build_structure_series(df_4h, lookback=3)
    daily_profile = pd.read_parquet(data_dir / "tick" / f"{prefix}_daily_profile.parquet")

    n_days = (footprint.index[-1] - footprint.index[0]).days
    print(f"Rentang data: {n_days} hari\n")

    print(f"Mencari konfigurasi dengan frekuensi {args.target_min}-{args.target_max} trade/hari...\n")
    print(f"{'Konfigurasi':70s} {'n_setup':>8s} {'n_trade':>8s} {'trade/hari':>11s}")
    print("-" * 100)

    chosen = None
    all_results = []
    for step in LADDER:
        vol_pct, delta_pct = step["absorption_pct"]
        th = calibrate_absorption_thresholds(
            footprint, volume_percentile=vol_pct, delta_pct_percentile=delta_pct
        )
        footprint["is_absorption"] = detect_absorption(footprint, th)

        setups = generate_setups(footprint, structure_1h, structure_4h, daily_profile, **step["gen_kwargs"])
        trades = simulate_trades(setups, footprint, CRYPTO_PERP, bar_minutes=5, max_holding_bars=288)

        freq = len(trades) / n_days if n_days > 0 else 0.0
        print(f"{step['label']:70s} {len(setups):>8d} {len(trades):>8d} {freq:>11.3f}")
        all_results.append((step, trades, freq))

        if chosen is None and args.target_min <= freq <= args.target_max:
            chosen = (step, trades, freq)

    if chosen is None:
        # Tidak ada yang persis kena target -- ambil yang PALING DEKAT,
        # jujurkan bahwa target tidak tercapai.
        chosen = min(all_results, key=lambda r: abs(r[2] - (args.target_min + args.target_max) / 2))
        print(f"\nPERHATIAN: tidak ada konfigurasi di tangga ini yang benar-benar mencapai "
              f"{args.target_min}-{args.target_max} trade/hari. Dipakai yang paling dekat.")

    step, trades, freq = chosen
    print(f"\n{'='*100}")
    print(f"KONFIGURASI TERPILIH: {step['label']}")
    print(f"  absorption_percentile = {step['absorption_pct']}")
    print(f"  generate_setups kwargs = {step['gen_kwargs']}")
    print(f"  Frekuensi = {freq:.3f} trade/hari ({len(trades)} trade dalam {n_days} hari)")
    print(f"{'='*100}\n")

    if trades.empty:
        print("Tidak ada trade sama sekali di konfigurasi ini -- tidak ada yang bisa disimulasikan.")
        return

    n = len(trades)
    mean = trades["net_pnl_bps"].mean()
    std = trades["net_pnl_bps"].std()
    t_stat = mean / (std / np.sqrt(n)) if std > 0 and n > 1 else float("nan")
    win_rate = (trades["net_pnl_bps"] > 0).mean()

    print("STATISTIK")
    print("-" * 60)
    print(f"n              = {n}")
    print(f"Win rate       = {win_rate:.1%}")
    print(f"Mean/trade     = {mean:.2f} bps")
    print(f"t-stat         = {t_stat:.3f}")
    print(f"(n>=30 di sini kemungkinan besar -- t-stat lebih bisa dipercaya "
          f"daripada trial n=13 sebelumnya, TAPI ini hipotesis BARU, bukan "
          f"prev_poc+2attempt yang lama)\n")

    for label, frac in [("MODAL (position_fraction diminta)", args.position_fraction),
                         ("Pembanding -- fraksi konservatif 20%", 0.2)]:
        final_capital, equity, max_dd = compute_equity_curve(trades, args.initial_capital, frac)
        print(f"{label} (fraksi={frac:.0%})")
        print("-" * 60)
        print(f"Modal awal      = Rp{args.initial_capital:,.0f}")
        print(f"Modal akhir     = Rp{final_capital:,.0f}")
        print(f"Return          = {(final_capital/args.initial_capital-1)*100:.2f}%")
        print(f"Max drawdown    = {max_dd:.1%}")
        print()

    if args.position_fraction >= 0.5:
        print("PERINGATAN: position_fraction >= 50% berarti SEBAGIAN BESAR/SELURUH modal "
              "dipertaruhkan di SATU trade setiap kali. Satu rangkaian rugi berturut-turut "
              "bisa menghabiskan modal jauh lebih cepat daripada yang terlihat dari rata-rata "
              "per-trade. Angka 'Pembanding 20%' di atas untuk perbandingan, bukan rekomendasi "
              "-- sizing yang aman itu keputusan risiko Anda sendiri, bukan sesuatu yang saya "
              "tentukan.")


if __name__ == "__main__":
    main()