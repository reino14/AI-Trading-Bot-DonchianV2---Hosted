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
import signal
import time
from pathlib import Path

import pandas as pd

from src.execution.broker import Broker, MockBroker
from src.strategy.base import Position
from src.strategy.donchian_close_futures import DonchianCloseFuturesParams, DonchianCloseFuturesStrategy
from src.strategy.regime_detector import detect_regime
from src.strategy.regime_filtered import RegimeFilteredStrategy
import src.runner.paper as paper_module
from src.runner.paper import PaperRunner


def make_channel_debug_fn(lookback: int):
    """
    Bangun fungsi debug_info_fn untuk PaperRunner -- menghitung channel
    Donchian (atas/bawah) dan jaraknya ke harga sekarang, PERSIS rumus
    yang sama dipakai compute_donchian_signal() (rolling_max/min close
    `lookback` bar SEBELUM bar ini, TIDAK termasuk bar ini sendiri --
    lihat src/strategy/donchian_close.py).

    Dipakai untuk tracking manual: kalau harga sudah lewat "atas" atau
    "bawah" yang dicetak fungsi ini tapi sinyal TIDAK berubah di bar
    berikutnya, itu petunjuk kuat ada bug -- channel di sini dihitung
    dari BUFFER YANG SAMA PERSIS yang dipakai sinyal sungguhan, bukan
    hitungan terpisah yang bisa diam-diam beda.
    """
    def fn(bars: list[dict]) -> str:
        if len(bars) < lookback + 1:
            return f"[channel] belum cukup data ({len(bars)}/{lookback + 1} bar)"
        closes = [b["close"] for b in bars]
        window = closes[-(lookback + 1):-1]  # `lookback` bar SEBELUM bar terakhir
        upper = max(window)
        lower = min(window)
        current = closes[-1]
        dist_upper = (upper - current) / current
        dist_lower = (current - lower) / current
        return (f"[channel lookback={lookback}] atas={upper:.2f} (jarak {dist_upper:+.3%})  "
                f"bawah={lower:.2f} (jarak {dist_lower:+.3%})")
    return fn


def make_channel_fn(lookback: int):
    """
    (atas, bawah) channel Donchian untuk mode masuk-lagi "midline" --
    RUMUS SAMA PERSIS dengan make_channel_debug_fn() di atas dan dengan
    sinyal: max/min close `lookback` bar SEBELUM bar terakhir.
    None kalau data belum cukup.
    """
    def fn(bars: list[dict]):
        if len(bars) < lookback + 1:
            return None
        window = [b["close"] for b in bars[-(lookback + 1):-1]]
        return max(window), min(window)
    return fn


#: buffer PaperRunner saat filter regime aktif (default runner 500) -- lihat
#: BATASAN di src/strategy/regime_filtered.py. 1500 bar 1h = ~62 hari.
REGIME_BUFFER_BARS = 1500


def make_debug_fn(lookback: int, regime_strategy: RegimeFilteredStrategy | None = None):
    """Debug channel seperti biasa, plus regime terkini kalau filter aktif --
    dihitung dari buffer YANG SAMA dengan sinyal sungguhan."""
    channel_fn = make_channel_debug_fn(lookback)
    if regime_strategy is None:
        return channel_fn

    def fn(bars: list[dict]) -> str:
        text = channel_fn(bars)
        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        r = detect_regime(df.set_index("timestamp"), regime_strategy.cfg).iloc[-1]
        return (f"{text}  [regime] {r['regime']} "
                f"(ADX={r['adx']:.1f} CHOP={r['chop']:.1f} ER={r['er']:.2f})")
    return fn


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


def use_price_pct_brackets() -> None:
    """
    Ubah arti --tp-roi-pct/--sl-roi-pct jadi % PERGERAKAN HARGA (= % dari nilai
    posisi), TIDAK bergantung leverage. Caranya: angka dikali leverage posisi
    SEBENARNYA saat entry sebelum masuk ke compute_bracket_roi() asli, yang
    lalu membaginya lagi dengan leverage yang sama -> harga SL/TP tetap.
    paper.py tidak diubah sama sekali.
    """
    asli = paper_module.compute_bracket_roi

    def by_price(entry, direction, leverage, tp_roi, sl_roi, fee_side):
        lev = leverage if leverage and leverage > 0 else 1.0
        return asli(entry, direction, leverage, tp_roi * lev, sl_roi * lev, fee_side)

    paper_module.compute_bracket_roi = by_price


def _sigterm_to_keyboard_interrupt(signum, frame):
    # Dashboard (Linux) menghentikan bot dengan SIGTERM. Default Python
    # langsung mati tanpa beres-beres; di sini diperlakukan sama dengan Ctrl+C.
    raise KeyboardInterrupt


async def close_position_on_stop(runner: PaperRunner, attempts: int = 5, wait_seconds: float = 3.0) -> None:
    """
    Dipanggil SETELAH loop utama berhenti (tombol Hentikan / Ctrl+C / SIGTERM).
    Posisi di bursa ditutup lewat jalur order yang SAMA dengan bot
    (_handle_signal_change), dicek ulang ke bursa tiap percobaan. Kalau tetap
    gagal, SL dipasang ulang supaya posisi tidak tertinggal tanpa pelindung.
    """
    print("\n=== Bot dihentikan -- menutup posisi di bursa (--close-on-stop) ===")
    for i in range(1, attempts + 1):
        try:
            runner._sync_position_from_exchange()
        except Exception as e:
            print(f"  [stop] gagal membaca posisi bursa ({e})")
        if runner._current_position == Position.FLAT:
            n = runner._cancel_conditional_orders_safely("bot dihentikan, posisi sudah flat")
            print(f"  [stop] posisi FLAT di bursa{' -- sisa SL/TP dibersihkan' if n and n > 0 else ''}. Selesai.")
            return
        try:
            price = runner.broker.fetch_current_price(runner.symbol)
        except Exception as e:
            print(f"  [stop] gagal ambil harga ({e}) -- coba lagi")
            time.sleep(wait_seconds)
            continue
        print(f"  [stop] percobaan {i}/{attempts}: tutup posisi {runner._current_position} di sekitar {price:.2f}")
        try:
            await runner._handle_signal_change(Position.FLAT, override_price=price)
        except Exception as e:
            print(f"  [stop] order penutup gagal ({e})")
        time.sleep(wait_seconds)

    try:
        runner._sync_position_from_exchange()
    except Exception:
        pass
    if runner._current_position != Position.FLAT:
        print("!" * 70)
        print(f"  [stop] POSISI MASIH TERBUKA setelah {attempts} percobaan -- TUTUP MANUAL di Binance.")
        if runner._bracket:
            runner._restore_stop_loss()
        print("!" * 70)
    else:
        runner._cancel_conditional_orders_safely("bot dihentikan, posisi sudah flat")
        print("  [stop] posisi FLAT di bursa. Selesai.")


def build_runner(args: argparse.Namespace) -> PaperRunner:
    # getattr: pemanggil lama yang menyusun Namespace sendiri (tanpa field
    # baru ini) tetap jalan dengan perilaku lama, bukan crash.
    reentry_mode = getattr(args, "reentry_mode", "reversal")
    params = DonchianCloseFuturesParams(lookback=args.lookback)
    strategy = DonchianCloseFuturesStrategy(params)
    regime_strategy = None
    if getattr(args, "regime_filter", False):
        regime_strategy = RegimeFilteredStrategy(strategy)
        strategy = regime_strategy
    print(f"  (Strategi: {strategy.describe()})")
    print(f"  (ALLOWS_SHORT={strategy.ALLOWS_SHORT} -- wajib True untuk futures selalu-di-pasar ini)")

    if args.mock:
        broker = MockBroker(fill_immediately=True)
        print("  (Broker: MockBroker -- TIDAK menyentuh jaringan sama sekali)")
    else:
        broker = Broker(exchange_id="binanceusdm", testnet=True)
        print("  (Broker: Binance Demo Trading, futures -- demo-fapi.binance.com)")

    if args.take_profit_pct is not None:
        print(f"  (Take-profit: {args.take_profit_pct:.1%}, "
              f"{'BERHENTI TOTAL setelah kena' if args.stop_after_take_profit else 'tahan arah sama sampai sinyal berbalik'})")
    if getattr(args, "tp_sl_price_pct", False) and getattr(args, "tp_roi_pct", None) is not None:
        use_price_pct_brackets()
        print(f"  (TP/SL dalam % PERGERAKAN HARGA = % dari nilai posisi: TP +{args.tp_roi_pct:g}%, "
              f"SL -{args.sl_roi_pct:g}% -- tidak bergantung leverage.)")
    if getattr(args, "tp_roi_pct", None) is not None:
        tahan_roi = ("masuk lagi setelah harga kembali ke tengah channel lalu breakout baru searah"
                     if reentry_mode == "midline" else "tahan arah sampai sinyal berbalik")
        print(f"  (SL/TP dari ROI KOTOR terhadap margin, dititipkan ke BURSA: TP +{args.tp_roi_pct:g}%, "
              f"SL -{args.sl_roi_pct:g}%. Harga dihitung dari leverage posisi yang sebenarnya.")
        print(f"   Setelah TP: {'BERHENTI TOTAL' if args.stop_after_take_profit else tahan_roi}. "
              f"Setelah SL: {tahan_roi}.)")
    if args.risk_reward is not None:
        print(f"  (SL + TP rasio {args.risk_reward:g}:1 BERSIH setelah fee, dititipkan ke BURSA:")
        print(f"   SL = maks({args.sl_atr_mult:g} x ATR{args.sl_atr_period}, {args.sl_min_fee_mult:g} x fee bolak-balik), "
              f"TP = {args.risk_reward:g} x SL + {args.risk_reward + 1:g} x fee.")
        tahan = ("masuk lagi setelah harga kembali ke tengah channel lalu breakout baru searah"
                 if reentry_mode == "midline" else "tahan arah sampai sinyal berbalik")
        print(f"   Setelah TP: {'BERHENTI TOTAL' if args.stop_after_take_profit else tahan}. "
              f"Setelah SL: {tahan}.)")

    return PaperRunner(
        strategy, broker, symbol=args.symbol, timeframe=args.timeframe,
        order_amount=args.amount, use_reduce_only=True,  # futures -- reduceOnly relevan, beda dari spot
        session_hours=args.session_hours, min_entry_buffer_hours=args.min_entry_buffer_hours,
        take_profit_pct=args.take_profit_pct, stop_after_take_profit=args.stop_after_take_profit,
        backfill_bars=args.backfill_bars, live_take_profit_poll_seconds=args.live_take_profit_poll_seconds,
        debug_info_fn=make_debug_fn(args.lookback, regime_strategy),
        buffer_size=REGIME_BUFFER_BARS if regime_strategy is not None else 500,
        risk_reward=args.risk_reward, sl_atr_mult=args.sl_atr_mult, sl_atr_period=args.sl_atr_period,
        sl_min_fee_mult=args.sl_min_fee_mult, bracket_poll_seconds=args.bracket_poll_seconds,
        reentry_mode=reentry_mode,
        tp_roi=(args.tp_roi_pct / 100) if getattr(args, "tp_roi_pct", None) is not None else None,
        sl_roi=(args.sl_roi_pct / 100) if getattr(args, "sl_roi_pct", None) is not None else None,
        reentry_channel_fn=make_channel_fn(args.lookback) if reentry_mode == "midline" else None,
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
    p.add_argument("--take-profit-pct", type=float, default=None,
                    help="mis. 0.02 untuk 2%% -- posisi ditutup paksa begitu floating PnL "
                         "mencapai ini, tidak menunggu sinyal strategi berbalik")
    p.add_argument("--stop-after-take-profit", action="store_true",
                    help="setelah take-profit kena SEKALI, program BERHENTI trading total "
                         "(bukan cuma menahan arah yang sama -- benar-benar berhenti permanen "
                         "sampai direstart manual)")
    p.add_argument("--backfill-bars", type=int, default=None,
                    help="mis. 200 -- isi buffer dari histori SEBELUM live mulai, supaya sinyal "
                         "pertama bisa langsung dihitung, tidak perlu menunggu lookback bar baru "
                         "satu-satu dari stream")
    p.add_argument("--live-take-profit-poll-seconds", type=float, default=None,
                    help="mis. 5.0 -- pantau take-profit lewat harga LIVE tiap sekian detik, "
                         "TERPISAH dari evaluasi candle (yang cuma sekali per candle tutup). "
                         "Tanpa ini, take-profit tetap jalan tapi cuma dievaluasi sekali per candle.")
    p.add_argument("--risk-reward", type=float, default=None,
                    help="mis. 2 -- pasang STOP LOSS dan TAKE PROFIT di BURSA dengan rasio UANG BERSIH "
                         "(setelah fee) R:1. Tidak bisa digabung dengan --take-profit-pct.")
    p.add_argument("--sl-atr-mult", type=float, default=2.0,
                    help="jarak SL = kelipatan ATR (default 2, aturan 2N sistem Turtle)")
    p.add_argument("--sl-atr-period", type=int, default=20,
                    help="jumlah bar untuk ATR, di timeframe bot (default 20)")
    p.add_argument("--sl-min-fee-mult", type=float, default=2.0,
                    help="batas bawah jarak SL = kelipatan fee bolak-balik (default 2), supaya "
                         "fee paling banyak sepertiga dari kerugian per SL")
    p.add_argument("--bracket-poll-seconds", type=float, default=5.0,
                    help="seberapa sering bot mengecek apakah SL/TP sudah kena, untuk membereskan "
                         "sisa order (eksekusi SL/TP sendiri oleh BURSA, tidak bergantung angka ini)")
    p.add_argument("--tp-roi-pct", type=float, default=None,
                    help="TAKE PROFIT sebagai ROI KOTOR terhadap margin, dalam PERSEN (mis. 5 = +5%%, sama "
                         "dengan ROI di aplikasi Binance). Wajib berpasangan dengan --sl-roi-pct.")
    p.add_argument("--sl-roi-pct", type=float, default=None,
                    help="STOP LOSS sebagai ROI KOTOR terhadap margin, dalam PERSEN (mis. 1.5 = -1,5%%). "
                         "Fee bolak-balik DITAMBAHKAN ke kerugian ini saat SL kena.")
    p.add_argument("--tp-sl-price-pct", action="store_true",
                    help="artikan --tp-roi-pct/--sl-roi-pct sebagai %% PERGERAKAN HARGA (= %% dari nilai "
                         "posisi), bukan ROI terhadap margin. Mis. --tp-roi-pct 5 = harga +5%%.")
    p.add_argument("--close-on-stop", action="store_true",
                    help="saat bot dihentikan (Ctrl+C / tombol Hentikan / SIGTERM), TUTUP posisi di bursa "
                         "dan bersihkan SL/TP. Tanpa ini posisi dibiarkan terbuka dengan SL/TP di bursa.")
    p.add_argument("--reentry-mode", choices=["reversal", "midline"], default="reversal",
                    help="setelah posisi ditutup TP/SL, kapan boleh masuk lagi ke arah YANG SAMA: "
                         "'reversal' (default) = tunggu sinyal berbalik; 'midline' = siap begitu harga "
                         "kembali ke tengah channel, lalu masuk saat ada breakout baru searah")
    p.add_argument("--regime-filter", action="store_true",
                    help="tahan ENTRY baru sampai market regime TREND searah sinyal "
                         "(ADX/Choppiness/Efficiency Ratio, lihat src/strategy/regime_detector.py). "
                         "Posisi yang sudah terbuka TIDAK ditutup paksa karena regime.")
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
    if (args.tp_roi_pct is None) != (args.sl_roi_pct is None):
        raise SystemExit("--tp-roi-pct dan --sl-roi-pct harus diisi berdua.")
    if args.tp_roi_pct is not None:
        if args.risk_reward is not None or args.take_profit_pct is not None:
            raise SystemExit("--tp-roi-pct/--sl-roi-pct tidak bisa digabung dengan --risk-reward atau "
                             "--take-profit-pct: semuanya memasang TP di bursa. Pilih satu cara.")
        if args.tp_roi_pct <= 0 or args.sl_roi_pct <= 0:
            raise SystemExit("--tp-roi-pct dan --sl-roi-pct harus > 0.")
    if args.risk_reward is not None:
        if args.take_profit_pct is not None:
            raise SystemExit("--risk-reward dan --take-profit-pct tidak bisa dipakai bersamaan: "
                             "keduanya memasang TP di bursa. Pilih salah satu.")
        if args.risk_reward <= 0 or args.sl_atr_mult <= 0 or args.sl_atr_period <= 0 or args.sl_min_fee_mult < 0:
            raise SystemExit("--risk-reward, --sl-atr-mult, --sl-atr-period harus > 0, "
                             "dan --sl-min-fee-mult tidak boleh negatif.")
        if args.live_take_profit_poll_seconds is not None:
            print("  (Catatan: --live-take-profit-poll-seconds tidak berpengaruh di mode --risk-reward; "
                  "SL/TP dieksekusi bursa.)")

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
        signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)
        try:
            asyncio.run(runner.run())
        except KeyboardInterrupt:
            if args.close_on_stop:
                # Abaikan sinyal stop berikutnya selama menutup -- jangan
                # sampai proses penutupan sendiri terpotong di tengah jalan.
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                asyncio.run(close_position_on_stop(runner))
            else:
                print("\n=== Bot dihentikan -- posisi & SL/TP di bursa DIBIARKAN (tanpa --close-on-stop) ===")


if __name__ == "__main__":
    main()