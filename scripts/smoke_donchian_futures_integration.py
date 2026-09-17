"""
scripts/smoke_donchian_futures_integration.py

Uji integrasi PALING PENTING: jalankan DonchianCloseFuturesStrategy
lewat run_backtest() ASLI milik Nero (src_realbot/backtest/engine.py,
salinan persis dari file yang dia kirim) -- BUKAN mesin backtest kita
sendiri. Kalau tes ini lulus, strategi ini TERBUKTI nyambung ke
infrastruktur Nero, bukan cuma "kelihatannya cocok di atas kertas".
"""

import sys

import numpy as np
import pandas as pd

from src.strategy.donchian_close import compute_donchian_signal
from src.strategy.donchian_close_futures import DonchianCloseFuturesParams, DonchianCloseFuturesStrategy
from src.strategy.trade_dependence import extract_trades
from src.backtest.engine import run_backtest, trades_to_dataframe
from src.core.cost_model import CostConfig
from src.strategy.base import Position

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1. ALLOWS_SHORT=True dideklarasikan eksplisit ==")
    strat = DonchianCloseFuturesStrategy(DonchianCloseFuturesParams(lookback=8))
    check("ALLOWS_SHORT True", strat.ALLOWS_SHORT is True)
    check("nama & describe() jalan (dipakai engine untuk pesan)",
          "lookback" in strat.describe(), f"dapat: {strat.describe()}")

    print("\n== 2. generate_signals: pemetaan Position benar dibanding compute_donchian_signal mentah ==")
    rng = np.random.default_rng(9)
    n = 300
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    prices = 100 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({
        "open": prices, "high": prices + 0.5, "low": prices - 0.5,
        "close": prices, "volume": rng.uniform(10, 100, n),
    }, index=idx)

    raw_signal = compute_donchian_signal(df["close"], lookback=8)
    mapped_signal = strat.generate_signals(df)

    mismatches = 0
    for i in range(len(df)):
        raw = raw_signal.iloc[i]
        mapped = mapped_signal.iloc[i]
        if pd.isna(raw):
            expected = Position.FLAT
        elif raw == 1.0:
            expected = Position.LONG
        else:
            expected = Position.SHORT
        if mapped != expected:
            mismatches += 1
    check("SEMUA bar terpetakan benar (NaN->FLAT, 1->LONG, -1->SHORT)",
          mismatches == 0, f"{mismatches} bar salah peta dari {len(df)}")

    print("\n== 3. run_backtest() ASLI milik Nero -- tidak crash, hasilkan trade ==")
    cost_cfg = CostConfig()  # default milik Nero, cukup untuk uji sambung
    trades = run_backtest(df, strat, cost_cfg, bar_minutes=60, entry_is_maker=False, exit_is_maker=False)
    trades_df = trades_to_dataframe(trades)
    check("Return berupa list Trade, tidak crash", isinstance(trades, list))
    check("ADA trade dihasilkan (lookback=8 di data acak -- wajar sering trading)",
          len(trades) > 0, f"dapat {len(trades)} trade")

    print("\n== 4. ALLOWS_SHORT=True -> ADA trade SHORT (tidak ditahan jadi FLAT oleh jaring pengaman) ==")
    if not trades_df.empty:
        n_short = (trades_df["direction"] == Position.SHORT).sum()
        check("ada trade SHORT di antara hasil (jaring pengaman TIDAK menahannya)",
              n_short > 0, f"dapat {n_short} trade short dari {len(trades_df)} total")

    print("\n== 5. Silang-cek jumlah trade: engine asli Nero vs extract_trades kita sendiri ==")
    # Dua implementasi INDEPENDEN (engine.py Nero vs trade_dependence.py
    # kita) yang memproses sinyal YANG SAMA -- jumlah trade seharusnya
    # SANGAT DEKAT (bisa beda 1 karena konvensi trade terakhir yang
    # masih terbuka, tapi tidak boleh jomplang jauh).
    our_trades = extract_trades(raw_signal, df["close"], cost_bps_per_unit=0.0)
    check("jumlah trade dari dua implementasi independen dekat (selisih <= 2)",
          abs(len(trades_df) - len(our_trades)) <= 2,
          f"engine Nero={len(trades_df)}, extract_trades kita={len(our_trades)}")

    print("\n== 6. PERINGATAN clipping TIDAK muncul (karena ALLOWS_SHORT=True) ==")
    # Ini dicek MANUAL dengan membaca output -- tidak ada assertion
    # otomatis di sini, cuma catatan supaya Anda lihat langsung: kalau
    # ALLOWS_SHORT sengaja diset False lalu strategi tetap hasilkan
    # SHORT, baris "PERINGATAN: strategi ... ALLOWS_SHORT=False tapi..."
    # akan tercetak di atas. Kalau TIDAK tercetak sama sekali di run
    # ini, itu konfirmasi jaring pengaman memang tidak aktif (benar,
    # sesuai desain untuk strategi futures ini).
    print("  (lihat apakah baris 'PERINGATAN: strategi ... ALLOWS_SHORT=False' muncul")
    print("   di atas -- SEHARUSNYA TIDAK, karena ALLOWS_SHORT=True di strategi ini)")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())