"""
HARI 1, lanjutan - mencari lama pegang posisi yang masuk akal.

analyze_costs.py mengukur pergerakan harga dalam SATU bar 1 menit.
Hasilnya: ongkos jauh lebih besar dari pergerakannya.

Skrip ini menguji hal yang berbeda: bagaimana kalau posisi dipegang lebih lama?

Logikanya:
  - Ongkos bolak-balik TETAP, berapa pun lama posisi dipegang
    (fee dan spread dibayar sekali di awal dan sekali di akhir)
  - Pergerakan harga TUMBUH seiring waktu, kira-kira mengikuti akar waktu
    (pegang 4x lebih lama, pergerakannya sekitar 2x lebih besar)

Jadi ada satu titik di mana pergerakan akhirnya melampaui ongkos.
Skrip ini mencari titik itu.

Sinyal tetap dihitung dari bar 1 menit. Yang berubah hanya berapa lama
posisi dipegang sebelum ditutup.

Cara pakai:
    python -m scripts.analyze_holding "BTC/USDT:USDT" "ETH/USDT:USDT"
"""

import argparse

import pandas as pd

from src.core.cost_model import PRESETS, CostConfig, round_trip_cost
from src.data.store import load_bars

# Lama pegang posisi yang diuji, dalam menit.
HOLDING_PERIODS = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]

# Ambang minimum. Di bawah ini, ongkos memakan terlalu banyak.
MIN_RATIO = 2.0


def forward_move_bps(df: pd.DataFrame, hold_minutes: int) -> pd.Series:
    """
    Seberapa jauh harga bergerak dari sekarang sampai `hold_minutes` ke depan.

    Arah diabaikan (dipakai nilai absolut), karena strategi bisa untung dari
    naik maupun turun. Yang tidak bisa dilawan adalah pergerakan yang terlalu
    kecil dibanding ongkos.

    Ini adalah batas ATAS dari yang bisa ditangkap: mengasumsikan masuk dan
    keluar tepat di ujung periode. Strategi nyata hanya menangkap sebagiannya.
    """
    fwd = df["close"].shift(-hold_minutes) / df["close"] - 1
    return (fwd.abs() * 10_000).dropna()


def evaluate_holding(symbol: str, cfg: CostConfig) -> pd.DataFrame:
    df = load_bars(symbol)
    rows = []

    for hold in HOLDING_PERIODS:
        move = forward_move_bps(df, hold)

        taker = round_trip_cost(
            cfg, entry_is_maker=False, exit_is_maker=False, hold_minutes=hold
        ).total_bps
        maker_entry = round_trip_cost(
            cfg, entry_is_maker=True, exit_is_maker=False, hold_minutes=hold
        ).total_bps
        maker_both = round_trip_cost(
            cfg, entry_is_maker=True, exit_is_maker=True, hold_minutes=hold
        ).total_bps

        p75 = move.quantile(0.75)

        rows.append(
            {
                "hold_min": hold,
                "median_bps": move.median(),
                "p75_bps": p75,
                "p90_bps": move.quantile(0.90),
                "cost_taker": taker,
                "cost_maker_entry": maker_entry,
                "cost_maker_both": maker_both,
                "ratio_taker": p75 / taker,
                "ratio_maker_entry": p75 / maker_entry,
                "ratio_maker_both": p75 / maker_both,
            }
        )

    return pd.DataFrame(rows)


def print_table(symbol: str, tbl: pd.DataFrame) -> None:
    print(f"\n{symbol}")
    print("-" * 86)
    print(
        f"{'Pegang':>8} | {'Gerak p75':>10} | "
        f"{'Ongkos':>8} {'Ongkos':>8} {'Ongkos':>8} | "
        f"{'Rasio':>7} {'Rasio':>7} {'Rasio':>7}"
    )
    print(
        f"{'(menit)':>8} | {'(bps)':>10} | "
        f"{'taker':>8} {'mk-entry':>8} {'mk-both':>8} | "
        f"{'taker':>7} {'mk-ent':>7} {'mk-both':>7}"
    )
    print("-" * 86)

    for _, r in tbl.iterrows():
        best = max(r["ratio_taker"], r["ratio_maker_entry"], r["ratio_maker_both"])
        mark = "  <-- lolos" if best >= MIN_RATIO else ""
        print(
            f"{int(r['hold_min']):>8} | {r['p75_bps']:>10.2f} | "
            f"{r['cost_taker']:>8.2f} {r['cost_maker_entry']:>8.2f} "
            f"{r['cost_maker_both']:>8.2f} | "
            f"{r['ratio_taker']:>7.2f} {r['ratio_maker_entry']:>7.2f} "
            f"{r['ratio_maker_both']:>7.2f}{mark}"
        )


def summarize(symbol: str, tbl: pd.DataFrame) -> None:
    """Cari lama pegang paling pendek yang lolos ambang, per skenario eksekusi."""
    print(f"\n  Kesimpulan {symbol}:")

    for label, col in [
        ("taker penuh", "ratio_taker"),
        ("limit saat masuk", "ratio_maker_entry"),
        ("limit masuk & keluar", "ratio_maker_both"),
    ]:
        lolos = tbl[tbl[col] >= MIN_RATIO]
        if lolos.empty:
            print(f"    {label:<22}: tidak ada yang lolos sampai 4 jam")
        else:
            first = lolos.iloc[0]
            print(
                f"    {label:<22}: mulai lolos di {int(first['hold_min'])} menit "
                f"(rasio {first[col]:.2f}x)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cari lama pegang posisi di mana pergerakan melampaui ongkos"
    )
    parser.add_argument("symbols", nargs="+")
    parser.add_argument(
        "--market",
        choices=list(PRESETS),
        default="crypto_perp",
        help="struktur biaya yang dipakai (default: crypto_perp)",
    )
    args = parser.parse_args()

    cfg = PRESETS[args.market]

    print("\n" + "=" * 86)
    print("PERGERAKAN HARGA vs ONGKOS, PER LAMA PEGANG POSISI")
    print("=" * 86)
    print("\nOngkos tetap. Pergerakan tumbuh seiring waktu.")
    print(f"Struktur biaya : {args.market}")
    print(f"Ambang minimum : rasio {MIN_RATIO}x")

    for symbol in args.symbols:
        try:
            tbl = evaluate_holding(symbol, cfg)
        except FileNotFoundError as e:
            print(f"\n{e}\n")
            continue
        print_table(symbol, tbl)
        summarize(symbol, tbl)

    print("\n" + "=" * 86)
    print("CARA MEMBACA")
    print("=" * 86)
    print(
        """
  Kolom rasio = pergerakan persentil 75 dibagi ongkos bolak-balik.

  Angka ini adalah BATAS ATAS. Dia mengasumsikan masuk dan keluar tepat
  di ujung periode, yang tidak mungkin dicapai strategi nyata. Strategi
  yang bagus mungkin menangkap 30-50% dari angka ini.

  Karena itu ambang 2x adalah SYARAT MINIMUM, bukan target. Rasio 3-4x
  memberi ruang yang lebih masuk akal.

  Perhatikan juga: makin lama posisi dipegang, makin sedikit transaksi
  per hari, dan makin lama waktu yang dibutuhkan untuk mengumpulkan 200
  transaksi yang diperlukan gate Hari 2.

  Trade-off-nya nyata: pegang lebih lama = rasio lebih baik, tapi bukti
  statistik lebih lambat terkumpul.
"""
    )


if __name__ == "__main__":
    main()