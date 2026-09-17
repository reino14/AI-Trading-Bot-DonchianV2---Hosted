"""
microstructure/performance_metrics.py

Empat metrik institusional: Sharpe ratio, max drawdown, win/loss ratio,
recovery time. Dipakai untuk SMC nanti, tapi ditulis generik -- bisa
dipakai ulang untuk menilai ulang hasil Chris Strategy juga.

KENAPA SHARPE BUKAN SEKADAR "MEAN/STD PER TRADE" (beda dari t-stat
yang sudah kita pakai sepanjang proyek ini)
------------------------------------------------------------------------
t-stat yang kita pakai sejauh ini dihitung dari daftar trade -- tiap
trade satu observasi, terlepas dari kapan/berapa lama dipegang. Itu
valid untuk uji signifikansi ("apakah hasil ini kebetulan?"), TAPI
BUKAN Sharpe ratio yang sesungguhnya. Sharpe ratio didefinisikan atas
DERET WAKTU return kurva ekuitas (biasanya harian), diannualisasi --
dua trade yang sama-sama +50bps tapi satu dipegang 5 menit dan satu
dipegang 2 hari punya kontribusi SANGAT BEDA ke Sharpe tahunan, tapi
kontribusi SAMA ke t-stat per-trade. Modul ini membangun kurva ekuitas
BERBASIS WAKTU KALENDER dulu (build_daily_equity_curve), baru
menghitung Sharpe darinya -- bukan langsung dari daftar trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_daily_equity_curve(
    trades: pd.DataFrame,
    initial_capital: float,
    position_fraction: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """
    trades: WAJIB punya kolom exit_time (datetime) dan net_pnl_bps,
        terurut waktu. P&L diterapkan SAAT EXIT (bukan saat entry) --
        itu titik waktu ketika untung/rugi benar-benar terealisasi.

    Return: Series modal, index = grid HARIAN dari `start` sampai
        `end` (bukan cuma titik-titik exit trade) -- kalau beberapa
        trade keluar di hari yang sama, urut sesuai waktu keluarnya
        dulu (posisi tunggal, exit_time sudah unik per desain
        trade_simulator.py), baru snapshot akhir hari itu yang dicatat
        ke grid harian.
    """
    capital = initial_capital
    points = [(start - pd.Timedelta(days=1), capital)]  # titik awal sebelum hari pertama

    for _, tr in trades.sort_values("exit_time").iterrows():
        pnl = capital * position_fraction * (tr["net_pnl_bps"] / 10_000)
        capital += pnl
        points.append((tr["exit_time"], capital))

    equity_at_trades = pd.Series(
        [p[1] for p in points], index=pd.DatetimeIndex([p[0] for p in points])
    )
    equity_at_trades = equity_at_trades[~equity_at_trades.index.duplicated(keep="last")]

    daily_grid = pd.date_range(start, end, freq="1D", tz=equity_at_trades.index.tz)
    combined_index = equity_at_trades.index.union(daily_grid)
    full = equity_at_trades.reindex(combined_index).sort_index().ffill()
    return full.reindex(daily_grid).ffill()


def sharpe_ratio(daily_equity: pd.Series, periods_per_year: float = 365.0) -> float:
    """
    Sharpe TAHUNAN dari kurva ekuitas harian. Asumsi risk-free rate = 0
    (lazim untuk backtest kripto jangka pendek -- bedanya kecil
    dibanding sumber ketidakpastian lain di sini).
    """
    daily_returns = daily_equity.pct_change().dropna()
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return float("nan")
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(periods_per_year))


def max_drawdown_pct(equity: pd.Series) -> float:
    """Drawdown terdalam, sebagai fraksi POSITIF (0.314 = -31.4%)."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(-drawdown.min())


@dataclass
class DrawdownEpisode:
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None  # None kalau BELUM pulih sampai akhir data
    depth_pct: float
    recovery_days: int | None  # None kalau belum pulih


def find_drawdown_episodes(equity: pd.Series) -> list[DrawdownEpisode]:
    """
    Pecah kurva ekuitas jadi episode drawdown: puncak baru -> lembah
    -> pulih ke atas puncak lama (atau BELUM pulih sampai data habis).

    Episode dengan recovery_date=None (belum pulih) SENGAJA tetap
    dimasukkan -- membuangnya akan membuat "waktu pulih rata-rata"
    terlihat lebih bagus dari kenyataan (survivorship bias di dalam
    data kita sendiri).
    """
    running_max = equity.cummax()
    is_at_peak = equity >= running_max

    episodes: list[DrawdownEpisode] = []
    peak_date = equity.index[0]
    peak_value = equity.iloc[0]
    in_drawdown = False
    trough_date = None
    trough_value = None

    for date, value in equity.items():
        if is_at_peak.loc[date]:
            if in_drawdown:
                episodes.append(DrawdownEpisode(
                    peak_date=peak_date, trough_date=trough_date, recovery_date=date,
                    depth_pct=float((peak_value - trough_value) / peak_value),
                    recovery_days=(date - trough_date).days,
                ))
                in_drawdown = False
            peak_date, peak_value = date, value
        else:
            if not in_drawdown or value < trough_value:
                trough_date, trough_value = date, value
            in_drawdown = True

    if in_drawdown:
        episodes.append(DrawdownEpisode(
            peak_date=peak_date, trough_date=trough_date, recovery_date=None,
            depth_pct=float((peak_value - trough_value) / peak_value),
            recovery_days=None,
        ))

    return episodes


def win_loss_ratio(trades: pd.DataFrame) -> float:
    """Rata-rata untung (bps) dibagi rata-rata rugi absolut (bps). NaN kalau tidak ada salah satunya."""
    wins = trades.loc[trades["net_pnl_bps"] > 0, "net_pnl_bps"]
    losses = trades.loc[trades["net_pnl_bps"] < 0, "net_pnl_bps"]
    if wins.empty or losses.empty:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def summarize_performance(
    trades: pd.DataFrame,
    initial_capital: float,
    position_fraction: float,
) -> dict:
    """Satu fungsi ringkasan -- gabungkan semua metrik di atas jadi satu dict siap cetak."""
    if trades.empty:
        return {"n_trades": 0}

    start = trades["exit_time"].min().normalize()
    end = trades["exit_time"].max().normalize()
    daily_equity = build_daily_equity_curve(trades, initial_capital, position_fraction, start, end)

    episodes = find_drawdown_episodes(daily_equity)
    recovered = [e.recovery_days for e in episodes if e.recovery_days is not None]
    still_in_drawdown = any(e.recovery_date is None for e in episodes)

    return {
        "n_trades": len(trades),
        "win_rate": float((trades["net_pnl_bps"] > 0).mean()),
        "win_loss_ratio": win_loss_ratio(trades),
        "sharpe_ratio": sharpe_ratio(daily_equity),
        "max_drawdown_pct": max_drawdown_pct(daily_equity),
        "n_drawdown_episodes": len(episodes),
        "avg_recovery_days": float(np.mean(recovered)) if recovered else float("nan"),
        "max_recovery_days": float(np.max(recovered)) if recovered else float("nan"),
        "still_in_drawdown_at_end": still_in_drawdown,
        "final_capital": float(daily_equity.iloc[-1]),
        "total_return_pct": float(daily_equity.iloc[-1] / initial_capital - 1),
    }