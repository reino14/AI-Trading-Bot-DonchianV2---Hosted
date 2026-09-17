"""
scripts/run_paper_donchian_futures.py

Launcher untuk PaperRunner dengan DonchianCloseFuturesStrategy --
MENGIKUTI PERSIS pola yang sudah ada di blok `--live` runner/paper.py
Anda sendiri (untuk DonchianBreakoutStrategy), cuma strategi dan
parameternya diganti. TIDAK menimpa runner/paper.py -- file terpisah,
supaya strategi lama tetap bisa dipakai lewat file aslinya.

BELUM DIUJI END-TO-END oleh saya -- saya tidak punya source
data/stream.py, execution/order_manager.py, execution/slippage_tracker.py
di sandbox saya, jadi tidak bisa membangun PaperRunner untuk tes
langsung seperti integrasi engine.py tadi. WAJIB dites dulu lewat jalur
MockBroker (tanpa --live) sebelum --live sungguhan -- persis anjuran
di paper.py Anda sendiri.

DEFAULT lookback=238 (tengah plateau tervalidasi dari scan 12-672),
BUKAN 8 -- kalau Anda mau override ke 8 atau nilai lain, itu keputusan
eksplisit Anda lewat --lookback, bukan default diam-diam yang mengarah
ke wilayah yang sudah terbukti buruk di scan sebelumnya.
"""

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from src.execution.broker import Broker, MockBroker
from src.strategy.donchian_close_futures import DonchianCloseFuturesParams, DonchianCloseFuturesStrategy
from src.runner.paper import PaperRunner


async def replay_historical(runner: PaperRunner, n_bars: int, data_dir: str, symbol_file: str) -> None:
    """
    Suapkan bar 1H BTC SUNGGUHAN (yang sudah ditarik sebelumnya untuk
    Chris Strategy) lewat runner.process_bar() SATU PER SATU, urut
    waktu -- validasi PALING KUAT sebelum --live: bukan cuma "berhasil
    dibuat objeknya", tapi benar-benar melihat sinyal berubah dan order
    (lewat MockBroker) benar-benar terkirim di data harga nyata.
    """
    path = Path(data_dir) / "klines" / symbol_file / "1h.parquet"
    df = pd.read_parquet(path).set_index("open_time")
    df.columns = [c.lower() for c in df.columns]
    df = df.tail(n_bars)

    print(f"\nMemutar ulang {len(df)} bar 1H terakhir dari {path} lewat process_bar()...\n")

    n_signal_changes = 0
    last_position = runner._current_position

    for ts, row in df.iterrows():
        bar = {
            "timestamp": int(ts.timestamp() * 1000),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0)),
        }
        await runner.process_bar(bar)
        if runner._current_position != last_position:
            n_signal_changes += 1
            last_position = runner._current_position

    print(f"\n=== Selesai memutar {len(df)} bar ===")
    print(f"  Perubahan posisi terjadi: {n_signal_changes} kali")
    print(f"  Posisi akhir: {runner._current_position}")
    if n_signal_changes == 0:
        print("  PERHATIAN: TIDAK ADA perubahan posisi sama sekali selama replay ini --")
        print("  wajar kalau n_bars < lookback+1, atau kebetulan tidak ada breakout di")
        print("  jendela ini. Coba n_bars lebih besar sebelum menyimpulkan pipa tidak jalan.")


def build_runner(args: argparse.Namespace) -> PaperRunner:
    params = DonchianCloseFuturesParams(lookback=args.lookback)
    strategy = DonchianCloseFuturesStrategy(params)
    print(f"  (Strategi: {strategy.describe()})")
    print(f"  (ALLOWS_SHORT={strategy.ALLOWS_SHORT} -- wajib True untuk futures selalu-di-pasar ini)")

    if args.mock:
        broker = MockBroker(fill_immediately=True)
        print("  (Broker: MockBroker -- TIDAK menyentuh jaringan sama sekali)")
    else:
        broker = Broker(exchange_id="binanceusdm", testnet=True)
        print("  (Broker: Binance Demo Trading, futures -- demo-fapi.binance.com)")

    return PaperRunner(
        strategy, broker, symbol=args.symbol, timeframe=args.timeframe,
        order_amount=args.amount, use_reduce_only=True,  # futures -- reduceOnly relevan, beda dari spot
        session_hours=args.session_hours, min_entry_buffer_hours=args.min_entry_buffer_hours,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true",
                    help="WAJIB diisi eksplisit untuk koneksi Demo Trading sungguhan. "
                         "Tanpa ini, dan tanpa --mock, program akan MENOLAK jalan -- "
                         "supaya tidak ada kombinasi ambigu yang diam-diam nembak jaringan.")
    p.add_argument("--mock", action="store_true",
                    help="Jalankan dengan MockBroker (tanpa jaringan sama sekali) -- "
                         "WAJIB dicoba dulu sebelum --live, persis pola paper.py asli Anda.")
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--timeframe", default="1h", help="WAJIB 1h -- lookback yang divalidasi dihitung dalam jam")
    p.add_argument("--lookback", type=int, default=238,
                    help="default 238 = tengah plateau tervalidasi. WAJIB override eksplisit "
                         "kalau mau nilai lain -- lihat peringatan soal lookback pendek sebelumnya.")
    p.add_argument("--amount", type=float, default=0.001, help="ukuran order dalam BTC")
    p.add_argument("--session-hours", type=float, default=None)
    p.add_argument("--min-entry-buffer-hours", type=float, default=None)
    p.add_argument("--replay-historical", type=int, default=None,
                    help="jumlah bar 1H BTC SUNGGUHAN terakhir untuk diputar ulang lewat "
                         "process_bar() di mode --mock -- validasi kuat sebelum --live. "
                         "Isi minimal lookback+50 supaya sempat lewati masa pemanasan.")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--symbol-file", default="BTCUSDT", help="nama folder di data-dir/klines/")
    args = p.parse_args()

    if not args.live and not args.mock:
        raise SystemExit(
            "Wajib pilih salah satu eksplisit: --mock (uji tanpa jaringan, WAJIB dicoba dulu) "
            "atau --live (Demo Trading sungguhan). Tidak ada default diam-diam."
        )
    if args.live and args.mock:
        raise SystemExit("--live dan --mock tidak bisa dipakai bersamaan -- pilih salah satu.")

    if args.lookback < 100:
        print(f"\nPERINGATAN: lookback={args.lookback} jauh di bawah plateau tervalidasi (148-328).")
        print("Scan sebelumnya menunjukkan wilayah ini konsisten buruk (lookback=72 dan 150")
        print("keduanya rugi bersih). Anda tetap bisa lanjut, tapi ini bukan parameter yang")
        print("sudah tervalidasi -- ini eksperimen terpisah, bukan strategi yang sudah teruji.\n")

    runner = build_runner(args)

    if args.mock:
        print("\n=== Mode MOCK -- tidak menyentuh jaringan, cuma verifikasi pipa ===")
        if args.replay_historical:
            asyncio.run(replay_historical(runner, args.replay_historical, args.data_dir, args.symbol_file))
        else:
            print("Panggil runner.process_bar(bar) manual dengan bar tiruan untuk uji,")
            print("persis pola __main__ non---live di runner/paper.py Anda sendiri.")
            print("Atau tambahkan --replay-historical N untuk memutar N bar BTC sungguhan")
            print("lewat pipa ini secara otomatis (rekomendasi: coba ini dulu).")
    else:
        print(f"\n=== Mode LIVE (Demo Trading) -- {args.symbol} @ {args.timeframe} ===")
        asyncio.run(runner.run())


if __name__ == "__main__":
    main()