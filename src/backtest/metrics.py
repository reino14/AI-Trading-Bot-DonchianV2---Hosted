"""
backtest/metrics.py

Mengubah daftar transaksi (dari backtest/engine.py) jadi angka yang
dipakai menilai apakah strategi lolos kriteria Hari 2:

  - Minimal 200 transaksi out-of-sample
  - Expectancy positif setelah biaya, t-stat > 2
  - Drawdown maksimum di bawah 15%
  - (Robustness ±20% parameter -- dicek terpisah di scripts/run_sweep.py,
    karena itu perlu menjalankan backtest berkali-kali, bukan tugas
    file ini)

PENTING: semua angka di sini dihitung dari net_pnl_bps -- P&L yang
SUDAH dipotong ongkos transaksi lewat cost_model.py. Kalau Anda pernah
melihat laporan strategi yang terlihat bagus tapi memakai P&L kotor
(gross), itu laporan yang menyesatkan; disiplin proyek ini sejak
Hari 1 adalah tidak pernah melihat angka tanpa ongkos.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    n_trades: int
    win_rate: float

    mean_net_pnl_bps: float
    std_net_pnl_bps: float
    t_stat: float  # seberapa yakin rata-rata net P&L beda dari nol, BUKAN kebetulan

    total_return_pct: float
    max_drawdown_pct: float

    def __str__(self) -> str:
        return (
            f"Jumlah transaksi   : {self.n_trades}\n"
            f"Win rate           : {self.win_rate:.1%}\n"
            f"Rata-rata net P&L  : {self.mean_net_pnl_bps:.2f} bps/transaksi\n"
            f"Std net P&L        : {self.std_net_pnl_bps:.2f} bps\n"
            f"t-stat             : {self.t_stat:.2f}\n"
            f"Return total       : {self.total_return_pct:.2f}%\n"
            f"Drawdown maksimum  : {self.max_drawdown_pct:.2f}%"
        )


def position_fraction_from_risk(
    risk_pct: float, stop_loss_bps: float, allow_leverage: bool = True
) -> float:
    """
    Hitung porsi modal (position_fraction) yang dipertaruhkan per
    transaksi, berdasarkan konvensi manajemen risiko standar
    (fixed-fractional risk sizing): ukuran posisi diatur supaya KALAU
    stop-loss penuh kena, kerugian yang terjadi persis sama dengan
    risk_pct dari modal saat itu -- bukan seluruh modal (position
    fraction = 1.0, yang jadi default lama compute_metrics kalau
    parameter ini tidak dipakai).

    risk_pct: pecahan modal yang mau dipertaruhkan per transaksi,
        mis. 0.01 untuk 1%. Ini ANGKA KONVENSI UMUM di manajemen
        risiko trading, bukan yang dicari-cari sampai hasil bagus.
    stop_loss_bps: jarak stop-loss strategi, dalam bps.
    allow_leverage: kalau False (WAJIB untuk SPOT -- tidak ada
        leverage/pinjaman), hasil DIBATASI maksimal 1.0 (seluruh modal,
        tidak lebih). Rumus fixed-fractional di atas bisa saja
        menghasilkan angka > 1.0 kalau stop_loss_bps sempit -- itu
        implisit berarti leverage, yang TIDAK VALID untuk spot trading
        (tidak bisa pegang aset melebihi modal sendiri tanpa pinjam).
        Default True supaya perilaku lama (futures) tidak berubah.
    """
    fraction = risk_pct / (stop_loss_bps / 10_000)
    if not allow_leverage:
        fraction = min(fraction, 1.0)
    return fraction


def compute_metrics(trades_df: pd.DataFrame, position_fraction: float = 1.0) -> BacktestMetrics:
    """
    Hitung semua metrik dari DataFrame transaksi (lihat
    backtest.engine.trades_to_dataframe).

    position_fraction: porsi modal yang dipertaruhkan per transaksi,
        dipakai HANYA untuk kurva ekuitas (total_return_pct dan
        max_drawdown_pct). Default 1.0 (seluruh modal per transaksi)
        -- SENGAJA dipertahankan sebagai default lama untuk kompatibel
        ke belakang, tapi ini asumsi yang TIDAK REALISTIS untuk trading
        sungguhan. Pakai position_fraction_from_risk() di atas untuk
        angka yang lebih masuk akal.

        win_rate, mean_net_pnl_bps, std_net_pnl_bps, dan t_stat TIDAK
        dipengaruhi position_fraction -- itu murni ukuran seberapa
        besar tiap transaksi bergerak secara harga (bps), sama sekali
        tidak bergantung berapa modal yang dipertaruhkan. Cuma kurva
        ekuitas (dan karenanya drawdown/return total) yang berubah
        kalau ukuran posisi diubah.

    Kurva ekuitas dihitung dengan COMPOUNDING: tiap transaksi
    mengalikan modal dengan (1 + net_pnl_bps * position_fraction / 10000).
    """
    if trades_df.empty:
        return BacktestMetrics(
            n_trades=0,
            win_rate=0.0,
            mean_net_pnl_bps=0.0,
            std_net_pnl_bps=0.0,
            t_stat=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
        )

    net = trades_df["net_pnl_bps"]
    n = len(net)

    win_rate = (net > 0).mean()
    mean_net = net.mean()
    std_net = net.std(ddof=1) if n > 1 else 0.0

    # t-stat: rata-rata dibagi galat-standar rata-rata. Mengukur apakah
    # mean_net "cukup jauh" dari nol dibanding sebarannya sendiri --
    # BUKAN cuma soal mean_net > 0, tapi seberapa yakin itu bukan
    # kebetulan dari sampel kecil / sebaran yang liar. TIDAK dipengaruhi
    # position_fraction.
    t_stat = (mean_net / (std_net / np.sqrt(n))) if std_net > 0 and n > 1 else 0.0

    # Kurva ekuitas, compounding, DISKALAKAN oleh position_fraction --
    # ini satu-satunya bagian yang berubah kalau ukuran posisi diubah.
    scaled_net = net * position_fraction
    equity = (1 + scaled_net / 10_000).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max  # selalu <= 0
    max_drawdown_pct = -drawdown.min() * 100  # dibalik jadi angka positif

    total_return_pct = (equity.iloc[-1] - 1) * 100

    return BacktestMetrics(
        n_trades=n,
        win_rate=win_rate,
        mean_net_pnl_bps=mean_net,
        std_net_pnl_bps=std_net,
        t_stat=t_stat,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
    )


@dataclass
class CapitalSimulation:
    """Hasil simulasi modal dalam satuan uang -- lihat simulate_capital()."""

    starting_capital: float
    ending_capital: float
    total_profit_loss: float
    best_trade_amount: float
    worst_trade_amount: float
    mean_trade_amount: float
    max_drawdown_amount: float  # negatif atau 0

    def __str__(self) -> str:
        return (
            f"Modal awal          : {self.starting_capital:,.2f}\n"
            f"Modal akhir         : {self.ending_capital:,.2f}\n"
            f"Untung/rugi total   : {self.total_profit_loss:+,.2f}\n"
            f"Transaksi terbaik   : {self.best_trade_amount:+,.2f}\n"
            f"Transaksi terburuk  : {self.worst_trade_amount:+,.2f}\n"
            f"Rata-rata/transaksi : {self.mean_trade_amount:+,.2f}\n"
            f"Drawdown (uang)     : {self.max_drawdown_amount:+,.2f} dari puncak modal tertinggi"
        )


def simulate_capital(
    trades_df: pd.DataFrame, position_fraction: float, starting_capital: float
) -> CapitalSimulation:
    """
    Simulasikan modal dalam SATUAN UANG (bukan bps abstrak) -- supaya
    hasil backtest lebih kebayang secara intuitif: modal awal berapa,
    jadi berapa di akhir, untung/rugi tiap transaksi berapa dalam uang
    sungguhan.

    PENTING: fungsi ini pakai LOGIKA COMPOUNDING YANG SAMA PERSIS dengan
    compute_metrics() (position_fraction yang sama, urutan transaksi
    yang sama) -- BUKAN simulasi baru dengan asumsi berbeda. ending_capital
    di sini akan selalu konsisten dengan total_return_pct dari
    compute_metrics(): ending_capital == starting_capital *
    (1 + total_return_pct/100), cuma direpresentasikan dalam satuan
    uang alih-alih persentase.

    max_drawdown_amount dihitung dari PUNCAK MODAL TERTINGGI yang pernah
    tercapai di sepanjang simulasi (bukan cuma dari modal awal) -- sama
    seperti max_drawdown_pct di compute_metrics().
    """
    if trades_df.empty:
        return CapitalSimulation(
            starting_capital=starting_capital,
            ending_capital=starting_capital,
            total_profit_loss=0.0,
            best_trade_amount=0.0,
            worst_trade_amount=0.0,
            mean_trade_amount=0.0,
            max_drawdown_amount=0.0,
        )

    scaled_net = trades_df["net_pnl_bps"] * position_fraction

    capital = starting_capital
    capital_curve = [capital]
    trade_amounts = []

    for r in scaled_net:
        pnl_amount = capital * (r / 10_000)
        trade_amounts.append(pnl_amount)
        capital += pnl_amount
        capital_curve.append(capital)

    capital_series = pd.Series(capital_curve)
    running_max = capital_series.cummax()
    max_drawdown_amount = (capital_series - running_max).min()  # <= 0

    return CapitalSimulation(
        starting_capital=starting_capital,
        ending_capital=capital,
        total_profit_loss=capital - starting_capital,
        best_trade_amount=max(trade_amounts),
        worst_trade_amount=min(trade_amounts),
        mean_trade_amount=sum(trade_amounts) / len(trade_amounts),
        max_drawdown_amount=max_drawdown_amount,
    )


@dataclass
class GateResult:
    """Hasil cek empat kriteria lolos Hari 2 (yang ke-4, robustness, dicek terpisah)."""

    passes_min_trades: bool
    passes_expectancy: bool
    passes_drawdown: bool
    overall_pass: bool  # gabungan tiga di atas -- BELUM termasuk uji sweep ±20%

    def __str__(self) -> str:
        def mark(b: bool) -> str:
            return "LULUS" if b else "TIDAK LULUS"

        return (
            f"Minimal 200 transaksi           : {mark(self.passes_min_trades)}\n"
            f"Expectancy positif & t-stat > 2  : {mark(self.passes_expectancy)}\n"
            f"Drawdown maksimum di bawah 15%   : {mark(self.passes_drawdown)}\n"
            f"--> Hasil gabungan (belum termasuk uji sensitivitas ±20%): "
            f"{mark(self.overall_pass)}"
        )


def evaluate_gate(
    metrics: BacktestMetrics,
    min_trades: int = 200,
    t_stat_threshold: float = 2.0,
    max_drawdown_threshold_pct: float = 15.0,
) -> GateResult:
    """
    Cek tiga dari empat kriteria Hari 2. Kriteria keempat (robustness,
    hasil tidak runtuh saat parameter digeser +-20%) SENGAJA tidak
    dicek di sini -- itu butuh menjalankan backtest berkali-kali dengan
    parameter berbeda, urusan scripts/run_sweep.py, bukan satu hasil
    backtest tunggal seperti yang dipegang fungsi ini.
    """
    passes_min_trades = metrics.n_trades >= min_trades
    passes_expectancy = metrics.mean_net_pnl_bps > 0 and metrics.t_stat > t_stat_threshold
    passes_drawdown = metrics.max_drawdown_pct < max_drawdown_threshold_pct

    overall_pass = passes_min_trades and passes_expectancy and passes_drawdown

    return GateResult(
        passes_min_trades=passes_min_trades,
        passes_expectancy=passes_expectancy,
        passes_drawdown=passes_drawdown,
        overall_pass=overall_pass,
    )


if __name__ == "__main__":
    # Uji asap: sambungkan ke engine.py, pakai data acak yang sama
    # seperti smoke test engine.py, cuma untuk memastikan compute_metrics
    # dan evaluate_gate tidak error dan menghasilkan angka yang bentuknya
    # masuk akal. INI BUKAN HASIL YANG BERMAKNA -- data acak murni,
    # tidak ada pola asli untuk ditangkap strategi apa pun.
    import numpy as np

    from src.backtest.engine import run_backtest, trades_to_dataframe
    from src.core.cost_model import CostConfig
    from src.strategy.vwap_reversion import VwapReversionStrategy

    rng = np.random.default_rng(42)
    n = 2000
    idx = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
    price = 100_000 + np.cumsum(rng.normal(0, 50, n))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.uniform(0, 30, n),
            "low": price - rng.uniform(0, 30, n),
            "close": price + rng.normal(0, 10, n),
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )

    strat = VwapReversionStrategy()
    cfg = CostConfig()
    trades = run_backtest(df, strat, cfg, bar_minutes=30)
    trades_df = trades_to_dataframe(trades)

    metrics = compute_metrics(trades_df)
    gate = evaluate_gate(metrics)

    print("=== METRIK ===")
    print(metrics)
    print("\n=== GATE HARI 2 ===")
    print(gate)