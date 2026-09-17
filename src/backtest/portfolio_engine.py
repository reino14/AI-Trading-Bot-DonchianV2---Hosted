"""
backtest/portfolio_engine.py

Mesin simulasi PORTOFOLIO (banyak aset sekaligus). Pendamping
backtest/engine.py yang lama -- BUKAN penggantinya. engine.py tetap
dipakai src/runner/paper.py yang sedang live, jangan disentuh.

TIGA PERBEDAAN MENDASAR DARI engine.py
--------------------------------------
1. UNIT AKUNTANSI. engine.py mencatat TRANSAKSI (buka -> tutup, P&L
   dalam bps per transaksi). Di portofolio cross-sectional, "transaksi"
   bukan unit yang bermakna: tiap rebalance Anda menggeser 100 posisi
   sedikit-sedikit, tidak ada yang jelas "buka" atau "tutup". Unitnya
   jadi RETURN PER PERIODE. Semua metrik dihitung dari deret return itu.

2. ONGKOS. engine.py memanggil round_trip_cost() sekali per transaksi.
   Di sini ongkos proporsional terhadap TURNOVER: sum(|w_t - w_{t-1}|)
   di tiap rebalance, dikali ongkos satu sisi. Menggeser bobot dari
   0.04 ke 0.05 itu turnover 0.01, bukan satu transaksi penuh.

3. GATE. Gate lama "min 200 transaksi" tidak punya padanan di sini, dan
   memang tidak perlu -- lihat catatan panjang di portfolio_metrics.py
   soal kenapa t-stat hanya bergantung pada Sharpe tahunan dan JUMLAH
   TAHUN data, bukan pada jumlah transaksi.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.cross_sectional_base import CrossSectionalStrategy


@dataclass(frozen=True)
class CostSpec:
    """
    Ongkos dalam bentuk yang dibutuhkan akuntansi berbasis turnover.

    per_side_bps:
        Ongkos SATU SISI (beli saja, atau jual saja) dalam bps: fee +
        separuh spread + slippage. Bukan bolak-balik. Ini berbeda dari
        round_trip_cost() di src/core/cost_model.py yang mengembalikan
        ongkos bolak-balik -- lihat cost_spec_from_config() di bawah,
        yang PERLU Anda verifikasi sebelum dipakai untuk angka sungguhan.

    financing_bps_per_day:
        Ongkos memegang eksposur per hari (funding perp, bunga margin).
        Dikenakan ke eksposur bruto. Set 0 untuk spot tanpa leverage.
        CATATAN: untuk strategi funding carry, funding justru PEMASUKAN
        dan harus masuk sebagai sinyal/return lewat panel `extra`, bukan
        lewat field ini.
    """

    per_side_bps: float
    financing_bps_per_day: float = 0.0


@dataclass
class PortfolioResult:
    """Hasil mentah satu backtest portofolio. Metrik dihitung terpisah."""

    gross_returns: pd.Series  # return per bar SEBELUM ongkos
    net_returns: pd.Series  # return per bar SESUDAH ongkos -- yang penting
    turnover: pd.Series  # sum(|dw|) per bar
    cost_drag: pd.Series  # ongkos per bar, dalam fraksi return
    weights_executed: pd.DataFrame  # bobot yang benar-benar dipegang
    equity: pd.Series  # kurva ekuitas dari net_returns
    bar_minutes: int
    strategy_name: str


def _align_and_sanitize(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    strategy: CrossSectionalStrategy,
    max_gross_exposure: float,
) -> pd.DataFrame:
    """
    Jaring pengaman struktural, dijalankan SEBELUM bobot dipakai.
    Filosofinya sama dengan clip SHORT di engine.py lama: jangan percaya
    begitu saja pada apa yang dikembalikan strategi.
    """
    if list(weights.index) != list(close.index):
        raise ValueError(
            f"{strategy.name}: index bobot tidak sama dengan index harga "
            f"({len(weights.index)} vs {len(close.index)} baris). "
            f"generate_weights() WAJIB mengembalikan index yang sama persis."
        )
    if list(weights.columns) != list(close.columns):
        raise ValueError(
            f"{strategy.name}: kolom bobot tidak sama dengan kolom harga. "
            f"generate_weights() WAJIB mengembalikan kolom yang sama persis."
        )

    w = weights.astype(float).fillna(0.0)
    w = w.replace([np.inf, -np.inf], 0.0)

    # Pengaman 1: strategi spot tidak boleh short.
    if not strategy.ALLOWS_SHORT:
        n_neg = int((w < 0).to_numpy().sum())
        if n_neg > 0:
            print(
                f"  PERINGATAN: {strategy.name} mendeklarasikan ALLOWS_SHORT=False "
                f"tapi menghasilkan {n_neg} bobot negatif -- dipotong jadi 0. "
                f"Ini kemungkinan BUG di generate_weights()."
            )
            w = w.clip(lower=0.0)

    # Pengaman 2: aset yang harganya tidak tersedia (belum listing / data
    # bolong) tidak bisa dipegang. Ini mencegah bias survivorship yang
    # paling sering terlewat: strategi memberi bobot ke aset yang pada
    # tanggal itu belum ada di bursa.
    tradable = close.notna() & close.shift(1).notna()
    w = w.where(tradable, 0.0)

    # Pengaman 3: batas eksposur bruto. Kalau strategi menghasilkan
    # gross > batas, seluruh baris diskalakan turun secara proporsional
    # (arah relatif antar aset dipertahankan, ukurannya saja yang dipotong).
    gross = w.abs().sum(axis=1)
    scale = np.where(gross > max_gross_exposure, max_gross_exposure / gross.replace(0, np.nan), 1.0)
    scale = pd.Series(scale, index=w.index).fillna(1.0)
    w = w.mul(scale, axis=0)

    return w


def run_portfolio_backtest(
    close: pd.DataFrame,
    strategy: CrossSectionalStrategy,
    cost: CostSpec,
    bar_minutes: int,
    extra: dict[str, pd.DataFrame] | None = None,
    execution_lag_bars: int = 1,
    max_gross_exposure: float = 1.0,
) -> PortfolioResult:
    """
    close: DataFrame harga penutupan, index datetime UTC, kolom = simbol.

    execution_lag_bars:
        Jeda antara bobot DIPUTUSKAN dan bobot DIEKSEKUSI, dalam bar.
        1 = minimum yang jujur: bobot dihitung dari penutupan bar t,
        dieksekusi di penutupan bar t, memperoleh return bar t+1.
        Naikkan ke 2 kalau ingin pesimis soal kecepatan eksekusi.
        JANGAN set 0 -- itu artinya memakai harga penutupan bar t untuk
        memutuskan lalu menerima return bar t juga, yaitu look-ahead.
    """
    if execution_lag_bars < 1:
        raise ValueError(
            "execution_lag_bars < 1 berarti look-ahead: bobot memperoleh "
            "return dari bar yang harganya dipakai untuk memutuskan bobot itu."
        )
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("close.index harus DatetimeIndex.")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close.index harus urut naik.")

    raw_weights = strategy.generate_weights(close, extra)
    w = _align_and_sanitize(raw_weights, close, strategy, max_gross_exposure)

    # Bobot yang BENAR-BENAR dipegang selama bar t adalah bobot yang
    # diputuskan execution_lag_bars bar sebelumnya. Penggeseran ini
    # dilakukan DI SINI, di engine -- bukan diserahkan ke strategi --
    # supaya tidak ada strategi yang bisa "lupa" menggesernya.
    w_exec = w.shift(execution_lag_bars).fillna(0.0)

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    gross_returns = (w_exec * returns).sum(axis=1)

    # Turnover: perubahan bobot yang dieksekusi dari bar ke bar.
    turnover = (w_exec - w_exec.shift(1).fillna(0.0)).abs().sum(axis=1)

    trading_cost = turnover * (cost.per_side_bps / 10_000.0)

    bars_per_day = 1440.0 / bar_minutes
    financing = (
        w_exec.abs().sum(axis=1)
        * (cost.financing_bps_per_day / 10_000.0)
        / bars_per_day
    )

    cost_drag = trading_cost + financing
    net_returns = gross_returns - cost_drag

    equity = (1.0 + net_returns).cumprod()

    return PortfolioResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turnover,
        cost_drag=cost_drag,
        weights_executed=w_exec,
        equity=equity,
        bar_minutes=bar_minutes,
        strategy_name=strategy.name,
    )


def apply_vol_target(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    target_annual_vol: float = 0.20,
    lookback_bars: int = 60,
    bar_minutes: int = 1440,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """
    Lapisan L5 (portfolio construction) -- OPSIONAL dan terpisah dari
    engine, supaya bisa diuji sendiri dan dimatikan sendiri.

    Skalakan seluruh portofolio naik/turun supaya volatilitas realisasinya
    mendekati target. Ini TIDAK menambah alpha sama sekali -- ia hanya
    membuat ukuran risiko konsisten sepanjang waktu, yang menaikkan Sharpe
    lewat penurunan varians, bukan lewat kenaikan return.

    Memakai volatilitas MASA LALU (rolling, digeser 1 bar) -- tidak boleh
    memakai volatilitas periode yang sedang dinilai.
    """
    bars_per_year = (365.0 * 1440.0) / bar_minutes
    port_ret = (weights.shift(1).fillna(0.0) * close.pct_change().fillna(0.0)).sum(axis=1)
    realized = port_ret.rolling(lookback_bars, min_periods=lookback_bars // 2).std()
    realized_ann = realized * np.sqrt(bars_per_year)
    scale = (target_annual_vol / realized_ann.replace(0, np.nan)).shift(1)
    scale = scale.clip(upper=max_leverage).fillna(0.0)
    return weights.mul(scale, axis=0)


def cost_spec_from_config(round_trip_bps_no_hold: float) -> CostSpec:
    """
    PERLU VERIFIKASI SEBELUM DIPAKAI UNTUK ANGKA SUNGGUHAN.

    Saya belum melihat isi src/core/cost_model.py, jadi saya TIDAK
    menebak API-nya. Jembatan sementara: jalankan round_trip_cost() Anda
    dengan hold_minutes=0 (supaya komponen funding nol dan yang tersisa
    murni fee + spread bolak-balik), lalu masukkan .total_bps ke sini.
    Fungsi ini membaginya dua untuk mendapat ongkos satu sisi.

    Pembagian dua ini SAH HANYA KALAU komponen entry dan exit di
    cost_model.py Anda simetris untuk skenario eksekusi yang sama.
    Kalau tidak, kirim saya isi cost_model.py dan saya buatkan adapter
    yang benar. Jangan pakai ini untuk keputusan go-live tanpa dicek.
    """
    return CostSpec(per_side_bps=round_trip_bps_no_hold / 2.0, financing_bps_per_day=0.0)