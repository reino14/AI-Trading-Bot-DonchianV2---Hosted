"""
scripts/smoke_portfolio_engine.py

Uji asap portfolio_engine.py. BUKAN backtest sungguhan -- ini memastikan
mesinnya sendiri benar sebelum dipakai menilai strategi apa pun.

Tes terpenting di sini adalah dua KENARI look-ahead (tes 1 dan 2).
Keduanya menguji hal yang tidak bisa diperiksa dengan mata: apakah bobot
baris t benar-benar dipasangkan dengan return bar t+1, bukan bar t.
Kalau pemasangannya meleset satu bar, SETIAP backtest setelah ini akan
terlihat bagus dan SEMUANYA salah. Jalankan ini tiap kali engine diubah.
"""

import sys

import numpy as np
import pandas as pd

from src.backtest.portfolio_engine import (
    CostSpec,
    apply_vol_target,
    run_portfolio_backtest,
)
from src.backtest.portfolio_metrics import (
    deflated_t_threshold,
    format_summary,
    gate,
    summarize,
)
from src.strategy.cross_sectional_base import (
    CrossSectionalParams,
    CrossSectionalStrategy,
)
from src.strategy.xs_momentum import XsMomentumParams, XsMomentumStrategy

try:
    from src.backtest.cost_adapter import build_cost_spec
    from src.core.cost_model import CRYPTO_SPOT

    _REAL_SPOT_COST = build_cost_spec(CRYPTO_SPOT, entry_is_maker=False, exit_is_maker=False)
except ImportError:
    _REAL_SPOT_COST = None  # cost_model.py belum ditempatkan di src/core/

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_panel(n_bars: int, n_assets: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_bars, freq="1D", tz="UTC")
    cols = [f"SYM{i:02d}" for i in range(n_assets)]
    rets = rng.normal(0, 0.03, (n_bars, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=cols)


# --------------------------------------------------------------------------
# Strategi kenari -- hanya untuk pengujian, tidak untuk dipakai
# --------------------------------------------------------------------------


class _FutureOracle(CrossSectionalStrategy):
    """Bobot baris t = tanda return bar t+1. Sengaja curang."""

    def __init__(self):
        super().__init__(CrossSectionalParams())

    def generate_weights(self, close, extra=None):
        future = close.pct_change().shift(-1)
        w = np.sign(future).fillna(0.0)
        return w / max(1, len(close.columns))


class _PastOnly(CrossSectionalStrategy):
    """Bobot baris t = tanda return bar t. Tidak curang."""

    def __init__(self):
        super().__init__(CrossSectionalParams())

    def generate_weights(self, close, extra=None):
        w = np.sign(close.pct_change()).fillna(0.0)
        return w / max(1, len(close.columns))


class _ConstantLong(CrossSectionalStrategy):
    """Beli semuanya bobot rata di bar pertama, tahan selamanya."""

    def __init__(self):
        super().__init__(CrossSectionalParams())

    def generate_weights(self, close, extra=None):
        return pd.DataFrame(
            1.0 / len(close.columns), index=close.index, columns=close.columns
        )


class _RandomNoise(CrossSectionalStrategy):
    """Bobot acak tiap bar. Ekspektansi kotor nol, ongkos besar."""

    def __init__(self, seed=3):
        super().__init__(CrossSectionalParams())
        self.seed = seed

    def generate_weights(self, close, extra=None):
        rng = np.random.default_rng(self.seed)
        raw = rng.normal(0, 1, close.shape)
        w = pd.DataFrame(raw, index=close.index, columns=close.columns)
        return w.div(w.abs().sum(axis=1), axis=0)


class _IllegalShort(CrossSectionalStrategy):
    """Mendeklarasikan spot-only tapi menghasilkan bobot negatif."""

    ALLOWS_SHORT = False

    def __init__(self):
        super().__init__(CrossSectionalParams())

    def generate_weights(self, close, extra=None):
        w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        w.iloc[:, 0] = 0.5
        w.iloc[:, 1] = -0.5
        return w


class _Oversized(CrossSectionalStrategy):
    """Meminta eksposur bruto 5x."""

    def __init__(self):
        super().__init__(CrossSectionalParams())

    def generate_weights(self, close, extra=None):
        return pd.DataFrame(
            5.0 / len(close.columns), index=close.index, columns=close.columns
        )


# --------------------------------------------------------------------------

FREE = CostSpec(per_side_bps=0.0)


def main() -> int:
    close = make_panel(n_bars=1200, n_assets=40)

    print("\n== 1. Kenari look-ahead: oracle masa depan HARUS menang telak ==")
    r = run_portfolio_backtest(close, _FutureOracle(), FREE, bar_minutes=1440)
    s = summarize(r.net_returns, 1440)
    check(
        "oracle Sharpe sangat tinggi",
        s["sharpe_annual"] > 10,
        f"Sharpe={s['sharpe_annual']:.1f}",
    )
    print("    (kalau ini GAGAL, pemasangan bobot-vs-return meleset satu bar)")

    print("\n== 2. Kenari look-ahead: sinyal murni masa lalu HARUS ~nol ==")
    r = run_portfolio_backtest(close, _PastOnly(), FREE, bar_minutes=1440)
    s = summarize(r.net_returns, 1440)
    check(
        "sinyal lagged tidak punya edge di random walk",
        abs(s["t_stat"]) < 2.5,
        f"t={s['t_stat']:.2f}",
    )
    print("    (kalau ini GAGAL, engine membocorkan informasi masa depan)")

    print("\n== 3. Akuntansi ongkos: turnover x ongkos, persis ==")
    cost = CostSpec(per_side_bps=10.0)
    r = run_portfolio_backtest(close, _ConstantLong(), cost, bar_minutes=1440)
    total_turnover = float(r.turnover.sum())
    total_cost = float(r.cost_drag.sum())
    check(
        "buy-and-hold hanya bayar sekali",
        abs(total_turnover - 1.0) < 1e-9,
        f"turnover total={total_turnover:.6f} (harus 1.0)",
    )
    check(
        "ongkos = turnover x 10bps",
        abs(total_cost - 1.0 * 10.0 / 10_000) < 1e-12,
        f"ongkos total={total_cost:.8f}",
    )

    print("\n== 4. Sinyal nol: kotor ~nol, bersih HARUS negatif ==")
    r = run_portfolio_backtest(close, _RandomNoise(), cost, bar_minutes=1440)
    sg = summarize(r.gross_returns, 1440)
    sn = summarize(r.net_returns, 1440)
    check("kotor tidak punya edge", abs(sg["t_stat"]) < 2.5, f"t_kotor={sg['t_stat']:.2f}")
    check(
        "bersih rugi karena ongkos",
        sn["mean_bps_per_bar"] < sg["mean_bps_per_bar"] - 1.0,
        f"kotor={sg['mean_bps_per_bar']:.2f}bps, bersih={sn['mean_bps_per_bar']:.2f}bps",
    )

    print("\n== 5. Jaring pengaman ALLOWS_SHORT ==")
    r = run_portfolio_backtest(close, _IllegalShort(), FREE, bar_minutes=1440)
    check(
        "bobot negatif dipotong jadi nol",
        float(r.weights_executed.min().min()) >= 0.0,
        f"bobot minimum={r.weights_executed.min().min():.4f}",
    )

    print("\n== 6. Batas eksposur bruto ==")
    r = run_portfolio_backtest(
        close, _Oversized(), FREE, bar_minutes=1440, max_gross_exposure=1.0
    )
    peak = float(r.weights_executed.abs().sum(axis=1).max())
    check("bruto dipotong ke 1.0", abs(peak - 1.0) < 1e-9, f"bruto puncak={peak:.6f}")

    print("\n== 7. Aset belum listing tidak bisa dipegang ==")
    holed = close.copy()
    holed.iloc[:300, 0] = np.nan
    r = run_portfolio_backtest(holed, _ConstantLong(), FREE, bar_minutes=1440)
    held_early = float(r.weights_executed.iloc[:300, 0].abs().sum())
    check("bobot nol selama data bolong", held_early == 0.0, f"jumlah bobot={held_early}")
    check("tidak ada NaN di return", not r.net_returns.isna().any())

    print("\n== 8. execution_lag_bars=0 harus ditolak ==")
    try:
        run_portfolio_backtest(close, _PastOnly(), FREE, bar_minutes=1440, execution_lag_bars=0)
        check("lag 0 ditolak", False, "tidak melempar error")
    except ValueError:
        check("lag 0 ditolak", True)

    print("\n== 9. Identitas t = Sharpe x sqrt(tahun) ==")
    r = run_portfolio_backtest(close, _RandomNoise(seed=11), FREE, bar_minutes=1440)
    s = summarize(r.net_returns, 1440)
    implied = s["sharpe_annual"] * np.sqrt(s["years"])
    check(
        "identitas terverifikasi secara numerik",
        abs(implied - s["t_stat"]) < 1e-6,
        f"SR x sqrt(th)={implied:.4f} vs t={s['t_stat']:.4f}",
    )

    print("\n== 10. Ambang t terkoreksi multiple testing ==")
    for n in (1, 10, 100, 1000):
        print(f"    {n:>4} percobaan -> butuh t > {deflated_t_threshold(n):.2f}")
    check("ambang naik seiring jumlah percobaan", deflated_t_threshold(100) > 3.0)

    print("\n== 11. Vol targeting menstabilkan vol, tidak menambah alpha ==")
    strat = XsMomentumStrategy(XsMomentumParams(lookback_bars=20, rebalance_bars=5))
    base_w = strat.generate_weights(close)
    vt_w = apply_vol_target(base_w, close, target_annual_vol=0.20, bar_minutes=1440)
    check("vol targeting mengubah ukuran bobot", not np.allclose(base_w.values, vt_w.values))
    check("tidak ada NaN setelah vol targeting", not vt_w.isna().to_numpy().any())

    print("\n== 12. Strategi referensi jalan ujung ke ujung ==")
    r = run_portfolio_backtest(
        close, strat, CostSpec(per_side_bps=5.0), bar_minutes=1440
    )
    s = summarize(r.net_returns, 1440, r.gross_returns, r.turnover)
    print(format_summary(s))
    passed, reasons = gate(s, n_trials=25)
    print(f"  GATE: {'LOLOS' if passed else 'GAGAL'}")
    for reason in reasons:
        print(f"    - {reason}")
    check(
        "momentum tidak menemukan edge di data acak",
        not passed,
        "benar -- data ini random walk, tidak ada edge untuk ditemukan",
    )

    print("\n== 13. KONTROL POSITIF: engine HARUS menemukan edge yang memang ada ==")
    # Panel dengan momentum cross-sectional sungguhan yang ditanam:
    # tiap aset punya drift tersembunyi yang bergerak lambat (AR(1)),
    # jadi return masa lalu benar-benar memprediksi return masa depan.
    rng = np.random.default_rng(99)
    n_bars, n_assets = 1200, 40
    drift = np.zeros((n_bars, n_assets))
    for t in range(1, n_bars):
        drift[t] = 0.97 * drift[t - 1] + rng.normal(0, 0.00035, n_assets)
    rets = drift + rng.normal(0, 0.02, (n_bars, n_assets))
    edge_panel = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)),
        index=pd.date_range("2022-01-01", periods=n_bars, freq="1D", tz="UTC"),
        columns=[f"SYM{i:02d}" for i in range(n_assets)],
    )
    r = run_portfolio_backtest(
        edge_panel, strat, CostSpec(per_side_bps=5.0), bar_minutes=1440
    )
    s = summarize(r.net_returns, 1440, r.gross_returns, r.turnover)
    print(format_summary(s))
    check(
        "edge yang ditanam terdeteksi",
        s["t_stat"] > 3.0,
        f"t={s['t_stat']:.2f}, Sharpe={s['sharpe_annual']:.2f}",
    )
    print("    (kalau ini GAGAL sementara tes 1-12 lulus, engine-nya terlalu")
    print("     tumpul -- ia akan menolak strategi bagus, bukan cuma yang jelek)")

    if _REAL_SPOT_COST is not None:
        print("\n== 14. Baseline momentum dengan ongkos CRYPTO_SPOT sungguhan ==")
        print(f"    (per_side={_REAL_SPOT_COST.per_side_bps:.2f}bps, "
              f"financing={_REAL_SPOT_COST.financing_bps_per_day:.3f}bps/hari -- "
              f"dari cost_model.py Anda, bukan angka bebas)")
        r = run_portfolio_backtest(close, strat, _REAL_SPOT_COST, bar_minutes=1440)
        s = summarize(r.net_returns, 1440, r.gross_returns, r.turnover)
        print(format_summary(s))
        check(
            "adapter ongkos sungguhan menyambung tanpa error",
            True,
        )
    else:
        print("\n== 14. Dilewati -- src/core/cost_model.py belum ditemukan di sini ==")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS. Engine siap dipakai menilai strategi sungguhan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())