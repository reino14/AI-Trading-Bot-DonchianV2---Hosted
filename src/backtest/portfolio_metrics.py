"""
backtest/portfolio_metrics.py

Metrik dan GATE untuk hasil portfolio_engine.py.

KENAPA GATE "MIN 200 TRANSAKSI" TIDAK DIBAWA KE SINI
----------------------------------------------------
Ada identitas yang layak Anda hafal, karena ia menjelaskan kenapa empat
strategi kemarin sulit dinilai:

    t_stat = Sharpe_tahunan  x  sqrt(jumlah_TAHUN_data)

Turunannya sederhana. t = mean / (std / sqrt(n)) = SR_periodik x sqrt(n).
SR_tahunan = SR_periodik x sqrt(bar_per_tahun), dan n = tahun x
bar_per_tahun. Substitusikan, faktor bar_per_tahun saling menghapus.

Konsekuensinya keras:

  - MENAIKKAN FREKUENSI TIDAK MENAMBAH BUKTI. Menguji strategi yang sama
    di bar 5 menit bukannya 2 jam melipatgandakan jumlah transaksi 24
    kali, tapi t-stat-nya TIDAK berubah sama sekali. "Jumlah transaksi"
    terasa seperti ukuran sampel, padahal bukan. Ini sebabnya gate 200
    transaksi bisa lolos sekaligus tidak memberi tahu apa-apa.

  - HANYA DUA JALAN MENAIKKAN t. Tambah TAHUN data, atau naikkan Sharpe.
    Data 1500 hari Anda = 4,1 tahun, jadi t > 2 mensyaratkan Sharpe
    tahunan > 2/sqrt(4,1) = 0,99. Gate lama Anda, tanpa disadari,
    sebenarnya mensyaratkan Sharpe ~1,0. Itu ambang yang wajar -- tapi
    sebaiknya dinyatakan terang-terangan, bukan tersembunyi.

  - INI ALASAN MATEMATIS PINDAH KE CROSS-SECTIONAL. Anda tidak bisa
    menambah tahun (data tidak ada). Satu-satunya jalan tersisa adalah
    menaikkan Sharpe, dan hukum dasar manajemen aktif mengatakan
    IR ~ IC x sqrt(N) dengan N = jumlah taruhan independen. Dari N=2 aset
    ke N=100 aset, faktor sqrt(N) naik ~7x. Itu bukan trik statistik --
    itu satu-satunya sumber daya yang belum Anda pakai.
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_EULER_MASCHERONI = 0.5772156649015329


def bars_per_year(bar_minutes: int) -> float:
    return (365.0 * 1440.0) / bar_minutes


def max_drawdown(equity: pd.Series) -> float:
    """Drawdown terdalam sebagai fraksi positif (0.15 = 15%)."""
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(-dd.min())


def summarize(
    net_returns: pd.Series,
    bar_minutes: int,
    gross_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
) -> dict:
    r = net_returns.dropna()
    n = len(r)
    bpy = bars_per_year(bar_minutes)
    years = n / bpy

    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0

    sharpe_ann = (mean / std * math.sqrt(bpy)) if std > 0 else 0.0
    t_stat = (mean / std * math.sqrt(n)) if std > 0 else 0.0

    equity = (1.0 + r).cumprod()
    total_growth = float(equity.iloc[-1]) if n > 0 else 1.0
    cagr = (total_growth ** (1.0 / years) - 1.0) if years > 0 and total_growth > 0 else float("nan")

    out = {
        "n_bars": n,
        "years": years,
        "mean_bps_per_bar": mean * 10_000.0,
        "vol_annual": std * math.sqrt(bpy),
        "sharpe_annual": sharpe_ann,
        "t_stat": t_stat,
        "cagr": cagr,
        "max_drawdown": max_drawdown(equity),
    }

    if gross_returns is not None:
        g = gross_returns.dropna()
        gstd = float(g.std(ddof=1)) if len(g) > 1 else 0.0
        out["sharpe_annual_gross"] = (
            float(g.mean()) / gstd * math.sqrt(bpy) if gstd > 0 else 0.0
        )
        out["cost_drag_annual"] = (float(g.mean()) - mean) * bpy

    if turnover is not None:
        out["turnover_annual"] = float(turnover.mean()) * bpy

    return out


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """
    Nilai Sharpe tertinggi yang DIHARAPKAN muncul dari n_trials percobaan
    yang semuanya BERNILAI NOL sebenarnya -- murni kebetulan.

    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio":
        E[max SR] ~ sigma_SR x [ (1-gamma) Z^-1(1 - 1/N)
                                 + gamma  Z^-1(1 - 1/(N e)) ]

    sharpe_std = sebaran Sharpe antar percobaan yang Anda jalankan.
    Kalau Anda melakukan sweep parameter, Anda BISA menghitungnya
    langsung: std dari Sharpe seluruh varian di sweep itu.
    """
    if n_trials <= 1:
        return 0.0
    nd = NormalDist()
    z1 = nd.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = nd.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sharpe_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)


def deflated_t_threshold(n_trials: int, base_t: float = 2.0) -> float:
    """
    Ambang t-stat yang sudah dikoreksi untuk multiple testing (Sidak).

    Kenapa perlu: kalau Anda menguji 100 varian di data yang sama, Anda
    hampir pasti menemukan satu dengan t > 2 walau semuanya sebenarnya
    nol. Ambang t=2 itu benar untuk SATU hipotesis, bukan untuk yang
    ke-100.

    Angkanya: 1 percobaan -> 2,00. 10 -> 2,80. 100 -> 3,48. 1000 -> 4,06.

    HITUNG n_trials DENGAN JUJUR. Yang dihitung bukan cuma sweep terakhir,
    tapi SEMUA yang pernah Anda coba di dataset yang sama -- termasuk
    empat strategi kemarin dan setiap varian parameternya. Kalau ragu,
    bulatkan ke atas.
    """
    if n_trials <= 1:
        return base_t
    nd = NormalDist()
    alpha_single = 2.0 * (1.0 - nd.cdf(base_t))  # two-sided, ~0.0455 untuk t=2
    alpha_adj = 1.0 - (1.0 - alpha_single) ** (1.0 / n_trials)
    return nd.inv_cdf(1.0 - alpha_adj / 2.0)


def gate(
    summary: dict,
    n_trials: int = 1,
    min_years: float = 2.0,
    max_dd_allowed: float = 0.35,
) -> tuple[bool, list[str]]:
    """
    Pengganti gate lama. Kembalikan (lolos, daftar alasan).

    max_dd_allowed default 0.35, bukan 0.15 seperti gate lama. Alasannya
    jujur: 15% adalah ambang yang wajar untuk sistem BER-LEVERAGE RENDAH
    di aset tradisional, dan tidak pernah dilewati satu pun strategi crypto
    Anda -- termasuk yang Sharpe-nya akan dianggap bagus oleh dana quant
    manapun. Ambang yang tidak pernah bisa dilewati bukan disiplin, itu
    cuma cara menolak semuanya. Naikkan/turunkan sesuai toleransi Anda,
    tapi TETAPKAN DI DEPAN sebelum melihat hasil.
    """
    reasons: list[str] = []
    t_need = deflated_t_threshold(n_trials)

    if summary["years"] < min_years:
        reasons.append(
            f"data cuma {summary['years']:.2f} tahun, minimum {min_years:.1f}"
        )
    if summary["t_stat"] < t_need:
        reasons.append(
            f"t-stat {summary['t_stat']:.2f} < ambang terkoreksi {t_need:.2f} "
            f"(untuk {n_trials} percobaan)"
        )
    if summary["max_drawdown"] > max_dd_allowed:
        reasons.append(
            f"drawdown {summary['max_drawdown']:.1%} > batas {max_dd_allowed:.0%}"
        )

    return (len(reasons) == 0), reasons


def format_summary(summary: dict) -> str:
    lines = [
        f"  Bar            : {summary['n_bars']} ({summary['years']:.2f} tahun)",
        f"  Mean per bar   : {summary['mean_bps_per_bar']:.3f} bps",
        f"  Vol tahunan    : {summary['vol_annual']:.1%}",
        f"  Sharpe tahunan : {summary['sharpe_annual']:.2f}",
        f"  t-stat         : {summary['t_stat']:.2f}",
        f"  CAGR           : {summary['cagr']:.1%}",
        f"  Max drawdown   : {summary['max_drawdown']:.1%}",
    ]
    if "sharpe_annual_gross" in summary:
        lines.append(f"  Sharpe (kotor) : {summary['sharpe_annual_gross']:.2f}")
        lines.append(f"  Beban ongkos   : {summary['cost_drag_annual']:.1%} / tahun")
    if "turnover_annual" in summary:
        lines.append(f"  Turnover       : {summary['turnover_annual']:.1f}x / tahun")
    return "\n".join(lines)