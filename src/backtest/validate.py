"""
backtest/validate.py

Dua tanggung jawab, keduanya soal DISIPLIN menguji, bukan soal strategi:

  1. Pemisahan data: in-sample (buat menyetel parameter) vs
     out-of-sample (buat diuji jujur, tidak pernah dilihat saat
     menyetel). Kalau parameter disetel dan diuji di data yang sama,
     hasilnya nyaris pasti terlihat bagus tapi tidak berarti apa-apa --
     itu bukan mengukur edge asli, itu mengukur seberapa pas strategi
     "menghafal" data itu.

  2. Uji sensitivitas +-20%: kriteria lolos keempat Hari 2. Geser tiap
     parameter satu per satu, jalankan ulang backtest, lihat apakah
     hasilnya masih masuk akal atau langsung ambruk. Strategi yang
     cuma profit di satu titik parameter yang sangat spesifik itu
     tanda overfitting, bukan tanda strategi bagus.
"""

import dataclasses
from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import run_backtest, trades_to_dataframe
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.core.cost_model import CostConfig


def resample_bars(df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    """
    Gabungkan bar 1-menit jadi bar window_minutes-menit. Sama persis
    logikanya dengan yang dipakai scripts/analyze_costs.py di Hari 1 --
    supaya window yang diuji kelayakan ongkosnya (Hari 1) dan window
    yang dipakai strategi (Hari 2) selalu konsisten.
    """
    if window_minutes <= 1:
        return df
    return (
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


def split_out_of_sample(
    df: pd.DataFrame, test_months: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pisah data jadi (train_df, test_df) berdasarkan tanggal, BUKAN
    diacak. test_df = test_months bulan terakhir; train_df = sisanya.

    Diacak itu salah untuk data deret waktu -- kalau baris diacak,
    strategi bisa "menyetel" pakai data dari masa depan relatif
    terhadap data yang diuji, itu look-ahead bias lewat jalan belakang.
    """
    cutoff = df.index.max() - pd.DateOffset(months=test_months)
    train_df = df[df.index < cutoff]
    test_df = df[df.index >= cutoff]
    return train_df, test_df


def perturb_param_variants(
    params, pct: float = 0.2
) -> list[tuple[str, object]]:
    """
    Hasilkan varian parameter: tiap field numerik digeser -pct dan
    +pct satu per satu (field lain tetap). Generik -- bekerja untuk
    dataclass parameter strategi apa pun, tidak spesifik ke
    VwapReversionParams, supaya strategi lain nanti bisa pakai fungsi
    yang sama tanpa diubah.

    Return: list of (label, instance_parameter_baru)
    """
    variants = []
    for f in dataclasses.fields(params):
        value = getattr(params, f.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue  # lewati field non-numerik (kalau ada)

        for direction_label, sign in (("turun", -1), ("naik", 1)):
            new_value = value * (1 + sign * pct)
            if isinstance(value, int):
                new_value = max(1, round(new_value))  # tetap bilangan bulat, minimal 1
            new_params = dataclasses.replace(params, **{f.name: new_value})
            label = f"{f.name} {direction_label} {pct:.0%} ({value} -> {new_value})"
            variants.append((label, new_params))
    return variants


@dataclass
class RobustnessResult:
    base_metrics: BacktestMetrics
    variant_results: list[tuple[str, BacktestMetrics]]
    overall_robust: bool  # True kalau net expectancy tetap positif di SEMUA varian


def check_robustness(
    strategy_cls,
    base_params,
    df: pd.DataFrame,
    cost_cfg: CostConfig,
    bar_minutes: int = 30,
    entry_is_maker: bool = False,
    exit_is_maker: bool = False,
    pct: float = 0.2,
) -> RobustnessResult:
    """
    Jalankan backtest dengan base_params, lalu dengan tiap varian dari
    perturb_param_variants(), semuanya di df yang SAMA (biasanya
    out-of-sample test set, konsisten dengan run_backtest.py).

    overall_robust dinilai KETAT: net expectancy harus tetap positif
    di base_params MAUPUN di setiap varian geseran. Kalau ada satu saja
    yang jadi rugi, dianggap tidak robust -- ini standar tinggi
    disengaja, sesuai kriteria "hasil tidak runtuh" di panduan Hari 2.
    """

    def _run(p) -> BacktestMetrics:
        strat = strategy_cls(p)
        trades = run_backtest(
            df,
            strat,
            cost_cfg,
            bar_minutes=bar_minutes,
            entry_is_maker=entry_is_maker,
            exit_is_maker=exit_is_maker,
        )
        return compute_metrics(trades_to_dataframe(trades))

    base_metrics = _run(base_params)

    variant_results = [
        (label, _run(variant_params))
        for label, variant_params in perturb_param_variants(base_params, pct=pct)
    ]

    overall_robust = base_metrics.mean_net_pnl_bps > 0 and all(
        m.mean_net_pnl_bps > 0 for _, m in variant_results
    )

    return RobustnessResult(
        base_metrics=base_metrics,
        variant_results=variant_results,
        overall_robust=overall_robust,
    )