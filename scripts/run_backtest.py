"""
scripts/run_backtest.py

HARI 2 -- jalankan satu backtest penuh di data instrumen asli, dengan
parameter strategi TERKUNCI.

Mendukung lebih dari satu pola strategi (lihat STRATEGIES di bawah).
Pola VWAP reversion (kandidat #1) sudah terbukti gagal lewat pengujian
menyeluruh -- filenya TETAP disimpan sebagai catatan, bukan dihapus.
Pola Donchian breakout (kandidat #2) ditambahkan di window waktu yang
berbeda (4 jam, bukan 30 menit) berdasarkan bukti bahwa window terlalu
pendek kemungkinan jadi akar masalah, bukan cuma pilihan pola.

Alur:
  1. Muat data mentah 1-menit lewat src.data.store.load_bars
  2. Resample ke window strategi (default 240 menit = 4 jam)
  3. Pisah data: N bulan terakhir jadi OUT-OF-SAMPLE, sisanya train.
     Backtest ini dijalankan di data OUT-OF-SAMPLE SAJA secara default.
  4. Hitung metrik, cek 3 dari 4 kriteria lolos (kriteria ke-4,
     robustness, ada di scripts/run_sweep.py)

Cara pakai:
    python -m scripts.run_backtest "BTC/USDT:USDT" "ETH/USDT:USDT"
    python -m scripts.run_backtest "BTC/USDT:USDT" --strategy donchian_breakout --window 240
    python -m scripts.run_backtest "BTC/USDT:USDT" --eval-on train --param entry_window_bars=30
    python -m scripts.run_backtest "BTC/USDT:USDT" --strategy vwap_reversion --window 30
    python -m scripts.run_backtest "BTC/USDT:USDT" "ETH/USDT:USDT" --pool
"""

import argparse
import dataclasses

import pandas as pd

from src.backtest.engine import run_backtest, trades_to_dataframe
from src.backtest.metrics import compute_metrics, evaluate_gate, position_fraction_from_risk, simulate_capital
from src.backtest.validate import resample_bars, split_out_of_sample
from src.core.cost_model import PRESETS, CostConfig
from src.data.store import load_bars
from src.strategy.donchian_breakout import DonchianBreakoutParams, DonchianBreakoutStrategy
from src.strategy.mean_reversion import MeanReversionParams, MeanReversionStrategy
from src.strategy.time_series_momentum import TimeSeriesMomentumParams, TimeSeriesMomentumStrategy
from src.strategy.vwap_reversion import VwapReversionParams, VwapReversionStrategy

# Daftar pola yang tersedia. Tambah baris baru di sini kalau ada
# kandidat pola berikutnya -- CLI (--strategy, --param) otomatis
# mendukungnya tanpa perlu edit kode lain.
STRATEGIES = {
    "vwap_reversion": (VwapReversionStrategy, VwapReversionParams),
    "donchian_breakout": (DonchianBreakoutStrategy, DonchianBreakoutParams),
    "time_series_momentum": (TimeSeriesMomentumStrategy, TimeSeriesMomentumParams),
    "mean_reversion": (MeanReversionStrategy, MeanReversionParams),
}


def build_params(params_cls, overrides: list[str]):
    """
    Bangun instance parameter dari default + override CLI.

    overrides: list string "key=value", mis. ["entry_window_bars=30"].
    Tipe data (int/float) diambil otomatis dari tipe nilai default field
    itu -- supaya tidak perlu daftar --nama-param khusus per strategi.
    """
    default_params = params_cls()
    kwargs = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"Format --param salah: '{item}' (harus key=value)")
        key, _, raw_value = item.partition("=")
        if not hasattr(default_params, key):
            valid = ", ".join(f.name for f in dataclasses.fields(default_params))
            raise SystemExit(f"Parameter '{key}' tidak dikenal. Pilihan: {valid}")
        caster = type(getattr(default_params, key))
        kwargs[key] = caster(raw_value)
    return dataclasses.replace(default_params, **kwargs)


def resolve_position_fraction(params, risk_pct: float, allow_leverage: bool = True) -> float:
    """
    Hitung position_fraction dari risk_pct, kalau strategi punya field
    stop_loss_bps. Kalau strategi masa depan tidak punya stop-loss
    eksplisit, fallback ke position_fraction=1.0 (asumsi lama) dengan
    peringatan tercetak -- bukan gagal diam-diam.

    allow_leverage: teruskan False untuk preset spot -- lihat
    src/backtest/metrics.py:position_fraction_from_risk soal kenapa
    ini WAJIB untuk instrumen tanpa leverage.
    """
    stop_loss_bps = getattr(params, "stop_loss_bps", None)
    if not stop_loss_bps or stop_loss_bps <= 0:
        print("  (Strategi ini tidak punya stop_loss_bps -- position sizing risk-based")
        print("  tidak bisa dihitung, pakai position_fraction=1.0 -- ini sudah sesuai")
        print("  untuk spot/no-leverage secara kebetulan, tapi dicetak eksplisit biar jelas.)")
        return 1.0
    return position_fraction_from_risk(risk_pct, stop_loss_bps, allow_leverage=allow_leverage)


#: Durasi satu bar dalam MENIT per timeframe ccxt -- dipakai supaya
#: bar_minutes (dan karenanya perhitungan funding/hold time di
#: cost_model) tetap benar walau data dibaca langsung dari resolusi
#: target (bukan hasil resample dari 1 menit). Sama seperti
#: scripts/analyze_costs.py -- lihat bug yang ditemukan di sana kalau
#: durasi bar dipaksa jadi 1 padahal aslinya 1440 (harian).
_TIMEFRAME_TO_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440, "1w": 10_080,
}


def run_one(
    symbol: str,
    window: int,
    test_months: int,
    entry_is_maker: bool,
    exit_is_maker: bool,
    strategy_name: str,
    param_overrides: list[str],
    eval_on: str = "test",
    risk_pct: float = 0.01,
    preset_name: str = "crypto_perp",
    starting_capital: float | None = None,
    source_timeframe: str = "1m",
) -> None:
    load_timeframe = None if source_timeframe == "1m" else source_timeframe
    skip_resample = load_timeframe is not None
    effective_window = _TIMEFRAME_TO_MINUTES.get(source_timeframe, window) if skip_resample else window

    print(f"\n{'=' * 78}")
    print(f"  {symbol}  (pola: {strategy_name}, "
          f"{'resolusi native ' + source_timeframe if skip_resample else f'window {window} menit'}, "
          f"{test_months} bulan terakhir = out-of-sample, preset: {preset_name})")
    print("=" * 78)

    raw = load_bars(symbol, timeframe=load_timeframe)
    bars = raw if skip_resample else resample_bars(raw, window)
    train_df, test_df = split_out_of_sample(bars, test_months=test_months)

    print(f"  Data terpakai        : {len(bars):,} bar total")
    print(
        f"  In-sample (train)    : {len(train_df):,} bar"
        + (f", {train_df.index.min():%Y-%m-%d} s/d {train_df.index.max():%Y-%m-%d}" if len(train_df) else " (kosong)")
    )
    print(
        f"  Out-of-sample (test) : {len(test_df):,} bar"
        + (f", {test_df.index.min():%Y-%m-%d} s/d {test_df.index.max():%Y-%m-%d}" if len(test_df) else " (kosong)")
    )

    eval_df = train_df if eval_on == "train" else test_df
    if len(eval_df) == 0:
        print(f"\n  Data {eval_on} kosong. Cek --test-months atau jumlah data yang diunduh.")
        return

    if eval_on == "train":
        print("\n  *** MODE EKSPLORASI -- dievaluasi di data IN-SAMPLE (train). ***")
        print("  *** Angka ini BUKAN hasil resmi Hari 2. Untuk keputusan lolos/tidak, ***")
        print("  *** wajib jalankan ulang TANPA --eval-on train (default: test).      ***")

    strategy_cls, params_cls = STRATEGIES[strategy_name]
    params = build_params(params_cls, param_overrides)
    strat = strategy_cls(params)
    cfg = PRESETS[preset_name]
    allow_leverage = preset_name != "crypto_spot"  # WAJIB no-leverage untuk spot

    trades = run_backtest(
        eval_df,
        strat,
        cfg,
        bar_minutes=effective_window,
        entry_is_maker=entry_is_maker,
        exit_is_maker=exit_is_maker,
    )
    trades_df = trades_to_dataframe(trades)

    position_fraction = resolve_position_fraction(params, risk_pct, allow_leverage=allow_leverage)
    metrics = compute_metrics(trades_df, position_fraction=position_fraction)
    gate = evaluate_gate(metrics)

    print(f"\n  Strategi : {strat.describe()}")
    print(f"  Eksekusi : entry {'maker' if entry_is_maker else 'taker'}, exit {'maker' if exit_is_maker else 'taker'}")
    print(f"  Sizing   : risiko {risk_pct:.1%} modal/transaksi -> posisi {position_fraction:.2f}x modal")
    print()
    print("  " + str(metrics).replace("\n", "\n  "))
    print()
    print("  " + str(gate).replace("\n", "\n  "))

    if starting_capital is not None:
        capital_sim = simulate_capital(trades_df, position_fraction, starting_capital)
        print("\n  --- Simulasi dalam satuan uang (konsisten dengan angka bps di atas) ---")
        print("  " + str(capital_sim).replace("\n", "\n  "))

    if eval_on == "train":
        print("\n  (Ingat: ini angka EKSPLORASI di train. Jalankan lagi tanpa --eval-on")
        print("   train untuk angka resmi di out-of-sample sebelum menyimpulkan apa pun.)")
    elif not gate.overall_pass:
        print("\n  Catatan: ini hasil yang SAH, bukan kegagalan proyek. Kalau tidak")
        print("  lolos, jangan menambah parameter supaya lolos -- itu overfitting.")
        print("  Coba pola strategi lain kalau ini sudah dieksplorasi tuntas.")
    else:
        print("\n  Lolos 3 dari 4 kriteria. Lanjut ke uji sensitivitas:")
        print(
            f"      python -m scripts.run_sweep \"{symbol}\" --strategy {strategy_name} "
            f"--window {window} --test-months {test_months}"
        )


def run_pooled(
    symbols: list[str],
    window: int,
    test_months: int,
    entry_is_maker: bool,
    exit_is_maker: bool,
    strategy_name: str,
    param_overrides: list[str],
    eval_on: str = "test",
    risk_pct: float = 0.01,
    preset_name: str = "crypto_perp",
    starting_capital: float | None = None,
    source_timeframe: str = "1m",
) -> None:
    """
    Jalankan backtest per instrumen, lalu GABUNGKAN semua transaksinya
    jadi satu sampel sebelum dihitung metrik/gate.

    Ini menguji hipotesis berbeda dari run_one(): bukan "apakah strategi
    ini profit di BTC" dan "di ETH" masing-masing, tapi "apakah strategi
    TERKUNCI ini, diterapkan ke kripto mayor pada umumnya, punya edge".
    Sah secara statistik selama parameter strategi SAMA PERSIS dipakai
    ke semua instrumen (tidak disetel ulang per instrumen) -- kalau
    disetel beda-beda per instrumen baru itu jadi masalah (overfitting
    per instrumen).

    Kegunaan utamanya: menambah ukuran sampel tanpa perlu data historis
    baru, dengan menggabungkan instrumen yang berkorelasi tapi berbeda,
    bukan mengulang-ulang melihat periode data yang sama.
    """
    load_timeframe = None if source_timeframe == "1m" else source_timeframe
    skip_resample = load_timeframe is not None
    effective_window = _TIMEFRAME_TO_MINUTES.get(source_timeframe, window) if skip_resample else window

    print(f"\n{'=' * 78}")
    print(f"  GABUNGAN {' + '.join(symbols)}  (pola: {strategy_name}, "
          f"{'resolusi native ' + source_timeframe if skip_resample else f'window {window} menit'}, "
          f"preset: {preset_name})")
    print("=" * 78)

    strategy_cls, params_cls = STRATEGIES[strategy_name]
    params = build_params(params_cls, param_overrides)
    cfg = PRESETS[preset_name]
    allow_leverage = preset_name != "crypto_spot"

    all_trades = []
    for symbol in symbols:
        raw = load_bars(symbol, timeframe=load_timeframe)
        bars = raw if skip_resample else resample_bars(raw, window)
        train_df, test_df = split_out_of_sample(bars, test_months=test_months)
        eval_df = train_df if eval_on == "train" else test_df

        if len(eval_df) == 0:
            print(f"\n  {symbol}: data {eval_on} kosong, dilewati.")
            continue

        strat = strategy_cls(params)
        trades = run_backtest(
            eval_df, strat, cfg, bar_minutes=effective_window,
            entry_is_maker=entry_is_maker, exit_is_maker=exit_is_maker,
        )
        trades_df = trades_to_dataframe(trades)
        print(f"  {symbol}: {len(trades_df)} transaksi")
        all_trades.append(trades_df)

    if not all_trades:
        print("\n  Tidak ada data untuk digabung.")
        return

    pooled_df = pd.concat(all_trades, ignore_index=True)

    if eval_on == "train":
        print("\n  *** MODE EKSPLORASI -- dievaluasi di data IN-SAMPLE (train). ***")
        print("  *** Angka ini BUKAN hasil resmi Hari 2. ***")

    position_fraction = resolve_position_fraction(params, risk_pct, allow_leverage=allow_leverage)
    metrics = compute_metrics(pooled_df, position_fraction=position_fraction)
    gate = evaluate_gate(metrics)

    print(f"\n  Strategi (terkunci sama untuk semua instrumen): {strategy_cls.__name__}({params})")
    print(f"  Total transaksi gabungan: {len(pooled_df)}")
    print()
    print("  " + str(metrics).replace("\n", "\n  "))
    print()
    print("  " + str(gate).replace("\n", "\n  "))

    if starting_capital is not None:
        capital_sim = simulate_capital(pooled_df, position_fraction, starting_capital)
        print("\n  --- Simulasi dalam satuan uang (konsisten dengan angka bps di atas) ---")
        print("  " + str(capital_sim).replace("\n", "\n  "))

    if eval_on == "train":
        print("\n  (Ingat: ini angka EKSPLORASI di train.)")
    elif not gate.overall_pass:
        print("\n  Catatan: hasil gabungan ini yang paling representatif untuk")
        print("  Hari 2 mengingat keterbatasan panjang data yang tersedia saat ini.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jalankan backtest Hari 2 di data instrumen asli")
    parser.add_argument("symbols", nargs="+", help="contoh: BTC/USDT:USDT ETH/USDT:USDT")
    parser.add_argument(
        "--strategy", choices=list(STRATEGIES), default="donchian_breakout", help="pola strategi yang diuji"
    )
    parser.add_argument(
        "--window", type=int, default=240, help="ukuran bar dalam menit, default 240 (4 jam)"
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=6,
        help="berapa bulan terakhir dipakai sebagai out-of-sample, default 6",
    )
    parser.add_argument(
        "--maker-entry", action="store_true", help="pakai limit order saat entry (default: taker)"
    )
    parser.add_argument(
        "--maker-exit", action="store_true", help="pakai limit order saat exit (default: taker)"
    )
    parser.add_argument(
        "--eval-on",
        choices=["train", "test"],
        default="test",
        help="'test' (default, RESMI) = out-of-sample; 'train' = EKSPLORASI di in-sample",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="override parameter strategi, format key=value, bisa diulang "
        "(mis. --param entry_window_bars=30 --param stop_loss_bps=80)",
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help="gabungkan transaksi dari SEMUA symbols jadi satu sampel sebelum "
        "dihitung metrik/gate, alih-alih dihitung terpisah per instrumen",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=0.01,
        help="pecahan modal yang dipertaruhkan per transaksi (fixed-fractional risk "
        "sizing, dihitung dari stop_loss_bps strategi), default 0.01 (=1%%). "
        "Pakai --risk-pct 1.0 untuk kembali ke asumsi lama (100%% modal/transaksi).",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="crypto_perp",
        help="preset ongkos dari src/core/cost_model.py -- 'crypto_perp' (default) atau "
        "'crypto_spot' (fee spot, TANPA funding, position sizing OTOMATIS dibatasi "
        "maksimal 1.0x modal karena spot tidak boleh leverage)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="kalau diisi (mis. 1000), tampilkan simulasi tambahan dalam SATUAN UANG "
        "(modal awal/akhir, untung/rugi tiap transaksi) -- angkanya dijamin konsisten "
        "dengan angka bps yang sudah ada, cuma direpresentasikan lebih intuitif",
    )
    parser.add_argument(
        "--source-timeframe",
        default="1m",
        help="resolusi FILE yang dibaca (bukan resample) -- default '1m' (perilaku lama). "
        "Isi mis. '1d' kalau sudah unduh data di resolusi itu lewat "
        "src/data/fetch_history.py --timeframe 1d -- --window diabaikan kalau ini bukan '1m'.",
    )
    args = parser.parse_args()

    if args.pool:
        try:
            run_pooled(
                args.symbols, args.window, args.test_months, args.maker_entry,
                args.maker_exit, args.strategy, args.param, args.eval_on, args.risk_pct,
                args.preset, args.capital, args.source_timeframe,
            )
        except FileNotFoundError as e:
            print(f"\n{e}\n")
        return

    for symbol in args.symbols:
        try:
            run_one(
                symbol,
                args.window,
                args.test_months,
                args.maker_entry,
                args.maker_exit,
                args.strategy,
                args.param,
                args.eval_on,
                args.risk_pct,
                args.preset,
                args.capital,
                args.source_timeframe,
            )
        except FileNotFoundError as e:
            print(f"\n{e}\n")


if __name__ == "__main__":
    main()