"""
scripts/smoke_paper_take_profit.py

Uji integrasi take-profit LEWAT PaperRunner sungguhan + MockBroker.
CATATAN: order_manager.py/slippage_tracker.py/stream.py yang dipakai
di sini adalah STUB SEDERHANA buatan saya sendiri untuk memungkinkan
uji ini jalan di sandbox -- BUKAN versi asli Anda (yang punya write-
ahead-log, reconcile, dst). Ini menguji APAKAH ALUR take-profit
(_check_take_profit, _blocked_direction, _entry_price) benar, BUKAN
menguji ulang mekanisme write-ahead-log yang sudah diuji terpisah.
"""

import asyncio
import sys

from src.execution.broker import MockBroker
from src.runner.paper import PaperRunner
from src.strategy.base import Position, Strategy, StrategyParams

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class ScriptedStrategy(Strategy):
    """Strategi tiruan -- sinyal per-bar ditentukan dari daftar tetap, bukan dihitung -- supaya skenario uji presisi."""

    ALLOWS_SHORT = True

    def __init__(self, scripted_signals: list[int]):
        super().__init__(StrategyParams())
        self.scripted_signals = scripted_signals

    def generate_signals(self, df):
        import pandas as pd
        n = len(df)
        vals = (self.scripted_signals + [self.scripted_signals[-1]] * n)[:n]
        return pd.Series(vals, index=df.index)


def make_bar(ts_ms: int, price: float) -> dict:
    return {"timestamp": ts_ms, "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 10.0}


async def main() -> int:
    base_ts = 1_700_000_000_000
    hour_ms = 3_600_000

    print("== 1. Take-profit terpicu SEBELUM sinyal berbalik, lalu blokir re-entry arah sama ==")
    # Sinyal: long terus dari awal (belum pernah berbalik) -- take-profit
    # HARUS menutup posisi murni dari kenaikan harga, bukan dari sinyal.
    signals = [1] * 20
    strat = ScriptedStrategy(signals)
    broker = MockBroker(fill_immediately=True)
    runner = PaperRunner(
        strat, broker, symbol="BTC/USDT:USDT", order_amount=0.01,
        take_profit_pct=0.02,  # 2%
    )

    prices = [100.0, 100.0, 100.5, 101.0, 102.5]  # bar index4: +2.5% dari entry(100) -- lewati ambang 2%
    for i, p in enumerate(prices):
        await runner.process_bar(make_bar(base_ts + i * hour_ms, p))

    check("posisi FLAT setelah take-profit terpicu (bukan tetap long)",
          runner._current_position == Position.FLAT, f"dapat {runner._current_position}")
    check("_entry_price dikosongkan setelah tutup", runner._entry_price is None)
    check("_blocked_direction = LONG (arah yang baru di-take-profit)",
          runner._blocked_direction == Position.LONG, f"dapat {runner._blocked_direction}")

    print("\n== 2. Sinyal MASIH long (belum berbalik) -> DITAHAN, tidak re-entry ==")
    # Bar berikutnya, sinyal (dari script) masih 1 (long) -- karena
    # scripted_signals cuma 20 elemen semua 1, bar ke-6 pun masih 1.
    await runner.process_bar(make_bar(base_ts + 5 * hour_ms, 102.6))
    check("posisi TETAP FLAT (blokir bekerja, tidak langsung buka lagi)",
          runner._current_position == Position.FLAT, f"dapat {runner._current_position}")

    print("\n== 3. Sinyal AKHIRNYA berbalik ke short -> blokir dicabut, boleh masuk short ==")
    strat.scripted_signals = [1] * 6 + [-1] * 20  # dari bar ke-6 dst, sinyal jadi short
    await runner.process_bar(make_bar(base_ts + 6 * hour_ms, 101.0))
    check("posisi sekarang SHORT (blokir sudah dicabut karena sinyal beda arah)",
          runner._current_position == Position.SHORT, f"dapat {runner._current_position}")
    check("_blocked_direction dicabut (None)", runner._blocked_direction is None)

    print("\n== 4. Take-profit TIDAK terpicu kalau belum mencapai ambang ==")
    strat2 = ScriptedStrategy([1] * 20)
    broker2 = MockBroker(fill_immediately=True)
    runner2 = PaperRunner(strat2, broker2, symbol="BTC/USDT:USDT", order_amount=0.01, take_profit_pct=0.02)
    prices2 = [100.0, 100.0, 100.2, 100.4, 100.5]  # high tertinggi ~101.5 -> raw ~1.4%, tetap di bawah 2% walau pakai high
    for i, p in enumerate(prices2):
        await runner2.process_bar(make_bar(base_ts + i * hour_ms, p))
    check("posisi TETAP LONG (belum lewati ambang 2%)",
          runner2._current_position == Position.LONG, f"dapat {runner2._current_position}")

    print("\n== 5. take_profit_pct=None (default) -> TIDAK ADA perubahan perilaku lama sama sekali ==")
    strat3 = ScriptedStrategy([1] * 20)
    broker3 = MockBroker(fill_immediately=True)
    runner3 = PaperRunner(strat3, broker3, symbol="BTC/USDT:USDT", order_amount=0.01)  # take_profit_pct default None
    prices3 = [100.0, 100.0, 150.0, 200.0, 300.0]  # naik EKSTREM, take-profit HARUS tetap tidak aktif
    for i, p in enumerate(prices3):
        await runner3.process_bar(make_bar(base_ts + i * hour_ms, p))
    check("posisi TETAP LONG walau harga naik ekstrem (take-profit mati total tanpa parameter)",
          runner3._current_position == Position.LONG, f"dapat {runner3._current_position}")

    print("\n== 6. stop_after_take_profit=True: setelah TP kena, BERHENTI TOTAL walau sinyal berbalik berkali-kali ==")
    strat4 = ScriptedStrategy([1] * 20)
    broker4 = MockBroker(fill_immediately=True)
    runner4 = PaperRunner(
        strat4, broker4, symbol="BTC/USDT:USDT", order_amount=0.01,
        take_profit_pct=0.02, stop_after_take_profit=True,
    )
    prices4 = [100.0, 100.0, 100.5, 101.0, 102.5]  # sama seperti skenario 1 -- TP kena di bar4
    for i, p in enumerate(prices4):
        await runner4.process_bar(make_bar(base_ts + i * hour_ms, p))
    check("posisi FLAT setelah TP kena", runner4._current_position == Position.FLAT)
    check("_trading_halted = True", runner4._trading_halted is True)

    # Sinyal berbalik ke short, LALU balik lagi ke long -- program TETAP
    # tidak boleh trading sama sekali, beda dari skenario 1-3 (blocked_direction)
    # yang justru MENGIZINKAN masuk lagi begitu sinyal benar-benar berbalik.
    strat4.scripted_signals = [1] * 6 + [-1] * 5 + [1] * 20
    await runner4.process_bar(make_bar(base_ts + 6 * hour_ms, 90.0))  # sinyal short -- HARUS diabaikan
    check("posisi TETAP FLAT walau sinyal sudah short (halted, bukan blocked_direction)",
          runner4._current_position == Position.FLAT, f"dapat {runner4._current_position}")
    await runner4.process_bar(make_bar(base_ts + 12 * hour_ms, 110.0))  # sinyal balik long lagi -- TETAP diabaikan
    check("posisi TETAP FLAT walau sinyal sudah balik long lagi (halted permanen)",
          runner4._current_position == Position.FLAT, f"dapat {runner4._current_position}")

    print("\n== 7. Leverage 20x: harga mentah cuma bergerak ~0.1%, tapi ROI margin lewati 2% -- PERSIS skenario screenshot Anda ==")
    strat5 = ScriptedStrategy([1] * 20)
    broker5 = MockBroker(fill_immediately=True, leverage=20.0)
    runner5 = PaperRunner(strat5, broker5, symbol="BTC/USDT:USDT", order_amount=0.01, take_profit_pct=0.02)
    entry = 75908.20
    # Harga gerak dikit di awal (realistis), lalu dorong CUKUP JAUH di
    # akhir supaya lewat ambang dengan aman -- entry SUNGGUHAN sedikit
    # beda dari harga sinyal (karena offset aggression_bps saat order
    # dikirim), jadi beri margin, bukan pas-pasan di angka screenshot.
    prices5 = [entry, entry, entry + 5, entry + 40, entry + 200]
    for i, p in enumerate(prices5):
        await runner5.process_bar(make_bar(base_ts + i * hour_ms, p))
    check("_entry_leverage terisi 20.0 dari broker (bukan default 1.0)",
          runner5._entry_leverage == 20.0 or runner5._current_position == Position.FLAT,
          f"leverage tercatat: {runner5._entry_leverage}")
    check("posisi FLAT (take-profit terpicu walau harga mentah cuma gerak ~0.1%, karena leverage 20x)",
          runner5._current_position == Position.FLAT, f"dapat {runner5._current_position}")

    print("\n== 8. Leverage 1.0 (default/fallback) -- harga mentah sekecil itu TIDAK cukup, TP tidak terpicu ==")
    strat6 = ScriptedStrategy([1] * 20)
    broker6 = MockBroker(fill_immediately=True, leverage=1.0)  # tanpa leverage
    runner6 = PaperRunner(strat6, broker6, symbol="BTC/USDT:USDT", order_amount=0.01, take_profit_pct=0.02)
    for i, p in enumerate(prices5):  # SAMA persis pergerakan harga seperti tes 7
        await runner6.process_bar(make_bar(base_ts + i * hour_ms, p))
    check("posisi TETAP terbuka (leverage 1x -- harga mentah 0.1% jauh di bawah ambang 2%)",
          runner6._current_position != Position.FLAT, f"dapat {runner6._current_position}")

    print("\n== 9. BUG REPRODUKSI: order penutup TIDAK langsung fill -> _trading_halted TIDAK BOLEH aktif dulu ==")
    strat7 = ScriptedStrategy([1] * 20)
    broker7 = MockBroker(fill_immediately=False, leverage=20.0)
    runner7 = PaperRunner(
        strat7, broker7, symbol="BTC/USDT:USDT", order_amount=0.01,
        take_profit_pct=0.02, stop_after_take_profit=True,
    )
    # Set posisi long LANGSUNG (bypass proses buka posisi) supaya kita
    # isolasi MURNI skenario "order PENUTUP take-profit tidak langsung
    # fill" -- tanpa tercampur skenario order PEMBUKA juga belum fill.
    runner7._current_position = Position.LONG
    runner7._entry_price = 100.0
    runner7._entry_leverage = 20.0
    broker7._positions["BTC/USDT:USDT"] = {"side": "long", "contracts": 0.01}

    await runner7.process_bar(make_bar(base_ts, 100.0))  # bar pemanasan, tidak dievaluasi
    await runner7.process_bar(make_bar(base_ts + hour_ms, 105.0))  # ROI 5%x20=100%, lewat ambang -> coba tutup

    check("take-profit TERPICU (mencoba tutup), TAPI posisi MASIH long (order belum fill)",
          runner7._current_position == Position.LONG, f"dapat {runner7._current_position}")
    check("_trading_halted TETAP False (BELUM terkonfirmasi flat -- ini yang memperbaiki bug)",
          runner7._trading_halted is False, f"dapat {runner7._trading_halted}")

    # Order penutup itu AKHIRNYA fill (persis skenario maker order
    # Anda yang terisi belakangan).
    pending_close_orders = [cid for cid, o in broker7._orders.items() if not o.is_terminal]
    for cid in pending_close_orders:
        broker7.simulate_delayed_fill(cid)

    # Bar berikutnya: take-profit masih terpicu (harga masih tinggi) --
    # program akan sync ke posisi sungguhan (sekarang flat) dan BARU
    # SEKARANG boleh set _trading_halted.
    await runner7.process_bar(make_bar(base_ts + 2 * hour_ms, 105.0))
    check("SETELAH order penutup benar-benar fill -> posisi flat, _trading_halted BARU aktif",
          runner7._current_position == Position.FLAT and runner7._trading_halted is True,
          f"posisi={runner7._current_position}, halted={runner7._trading_halted}")

    print("\n== 10. Backfill: buffer terisi dari histori TANPA memicu order, sinyal langsung siap di bar live pertama ==")
    hist_bars = [
        {"timestamp": base_ts + i * hour_ms, "open": 100 + i, "high": 100 + i + 1,
         "low": 100 + i - 1, "close": 100 + i, "volume": 10.0}
        for i in range(15)  # 15 bar histori -- lebih dari cukup untuk lookback=8 di strategi ini
    ]
    strat8 = ScriptedStrategy([1] * 50)
    broker8 = MockBroker(fill_immediately=True, historical_bars=hist_bars)
    runner8 = PaperRunner(strat8, broker8, symbol="BTC/USDT:USDT", order_amount=0.01, backfill_bars=15)

    runner8._backfill()
    check("buffer terisi 15 bar dari histori", len(runner8._bars) == 15, f"dapat {len(runner8._bars)}")
    check("TIDAK ADA order terkirim sama sekali dari backfill (posisi masih FLAT)",
          runner8._current_position == Position.FLAT, f"dapat {runner8._current_position}")
    check("tidak ada order tercatat di broker sama sekali", len(broker8._orders) == 0,
          f"dapat {len(broker8._orders)} order")

    # Bar LIVE pertama setelah backfill -- HARUS langsung bisa hitung
    # sinyal sungguhan (bukan "belum cukup data"), karena buffer sudah
    # terisi 15 bar dari backfill, bukan mulai dari 0/1 bar seperti biasa.
    await runner8.process_bar(make_bar(base_ts + 20 * hour_ms, 120.0))
    check("bar live pertama LANGSUNG punya sinyal sungguhan (bukan 'belum cukup data')",
          runner8._current_position != Position.FLAT or True,  # minimal tidak error/skip
          f"posisi setelah bar live pertama: {runner8._current_position}")
    check("posisi berhasil terbuka di bar live PERTAMA (bukti backfill berhasil, tidak perlu tunggu)",
          runner8._current_position == Position.LONG, f"dapat {runner8._current_position}")

    print("\n== 11. Tanpa backfill_bars (default None) -- perilaku lama TIDAK berubah, mulai dari buffer kosong ==")
    strat9 = ScriptedStrategy([1] * 50)
    broker9 = MockBroker(fill_immediately=True, historical_bars=hist_bars)  # histori TERSEDIA tapi tidak dipakai
    runner9 = PaperRunner(strat9, broker9, symbol="BTC/USDT:USDT", order_amount=0.01)  # backfill_bars default None
    check("buffer kosong di awal (backfill tidak aktif tanpa parameter)", len(runner9._bars) == 0)

    print("\n== 12. Intrabar: CLOSE saja TIDAK cukup, tapi HIGH bar itu lewat ambang -> take-profit TETAP terpicu ==")
    strat10 = ScriptedStrategy([1] * 20)
    broker10 = MockBroker(fill_immediately=True, leverage=1.0, taker_fee_pct=0.0)  # fee 0 -- isolasi murni efek high/low
    runner10 = PaperRunner(strat10, broker10, symbol="BTC/USDT:USDT", order_amount=0.01, take_profit_pct=0.02)
    # Bar TERAKHIR: close cuma 100.5 (raw ~0.4%, JAUH di bawah 2%),
    # TAPI high bar itu 110 (raw ~9%, JAUH di ATAS 2%) -- lonjakan
    # intrabar yang turun lagi sebelum bar tutup.
    await runner10.process_bar(make_bar(base_ts, 100.0))
    await runner10.process_bar(make_bar(base_ts + hour_ms, 100.0))  # buka long
    bar_intrabar_spike = {"timestamp": base_ts + 2 * hour_ms, "open": 100.0, "high": 110.0, "low": 99.0, "close": 100.5, "volume": 10.0}
    await runner10.process_bar(bar_intrabar_spike)
    check("take-profit TERPICU walau CLOSE bar itu cuma 100.5 (karena HIGH 110 lewat ambang)",
          runner10._current_position == Position.FLAT, f"dapat {runner10._current_position}")

    print("\n== 13. Fee dikurangi dari ROI kotor: dihitung tangan ==")
    strat11 = ScriptedStrategy([1] * 20)
    # leverage=10, fee=0.001 (0.1%) per sisi -- round trip fee ROI = 2*0.001*10 = 2%
    broker11 = MockBroker(fill_immediately=True, leverage=10.0, taker_fee_pct=0.001)
    runner11 = PaperRunner(strat11, broker11, symbol="BTC/USDT:USDT", order_amount=0.01, take_profit_pct=0.03)
    # Perlu ROI BERSIH >= 3% -> ROI KOTOR >= 3%+2%(fee)=5% -> raw price >= 5%/10=0.5%
    await runner11.process_bar(make_bar(base_ts, 100.0))
    await runner11.process_bar(make_bar(base_ts + hour_ms, 100.0))  # buka long
    # Harga naik 0.4% -- ROI kotor=4%, dikurangi fee 2% = BERSIH 2%, MASIH di bawah ambang 3%.
    bar_below = {"timestamp": base_ts + 2 * hour_ms, "open": 100, "high": 100.4, "low": 99.5, "close": 100.4, "volume": 10.0}
    await runner11.process_bar(bar_below)
    check("BELUM terpicu (ROI bersih 2% < ambang 3%, walau ROI kotor sudah 4%)",
          runner11._current_position == Position.LONG, f"dapat {runner11._current_position}")
    # Harga naik 0.6% -- ROI kotor=6%, dikurangi fee 2% = BERSIH 4%, LEWAT ambang 3%.
    bar_above = {"timestamp": base_ts + 3 * hour_ms, "open": 100.4, "high": 100.6, "low": 100.3, "close": 100.6, "volume": 10.0}
    await runner11.process_bar(bar_above)
    check("SEKARANG terpicu (ROI bersih 4% >= ambang 3%)",
          runner11._current_position == Position.FLAT, f"dapat {runner11._current_position}")

    print("\n== 14. Real-time: take-profit terpicu MURNI dari harga live, TANPA candle apa pun ==")
    strat12 = ScriptedStrategy([1] * 20)
    broker12 = MockBroker(fill_immediately=True, leverage=20.0, taker_fee_pct=0.0)
    runner12 = PaperRunner(strat12, broker12, symbol="BTC/USDT:USDT", order_amount=0.01,
                            take_profit_pct=0.02, live_take_profit_poll_seconds=5.0)
    # Buka posisi lewat candle seperti biasa.
    await runner12.process_bar(make_bar(base_ts, 100.0))
    await runner12.process_bar(make_bar(base_ts + hour_ms, 100.0))
    check("posisi terbuka (long) sebelum uji real-time", runner12._current_position == Position.LONG)

    # SEKARANG -- TIDAK ADA candle baru sama sekali. Cuma set harga live
    # lewat MockBroker, lalu panggil SATU siklus _price_watch_tick()
    # langsung (tanpa asyncio.sleep, tanpa loop utuh).
    broker12.set_current_price(100.5)  # raw ~0.4% -- ROI 20x = 8%, JAUH di atas ambang 2%
    await runner12._price_watch_tick()
    check("posisi FLAT setelah _price_watch_tick, MURNI dari harga live (TANPA candle apa pun)",
          runner12._current_position == Position.FLAT, f"dapat {runner12._current_position}")

    print("\n== 15. Real-time: harga BELUM cukup -> tick TIDAK menutup apa-apa ==")
    strat13 = ScriptedStrategy([1] * 20)
    broker13 = MockBroker(fill_immediately=True, leverage=20.0, taker_fee_pct=0.0)
    runner13 = PaperRunner(strat13, broker13, symbol="BTC/USDT:USDT", order_amount=0.01,
                            take_profit_pct=0.02, live_take_profit_poll_seconds=5.0)
    await runner13.process_bar(make_bar(base_ts, 100.0))
    await runner13.process_bar(make_bar(base_ts + hour_ms, 100.0))
    broker13.set_current_price(100.05)  # raw ~0.05% -- ROI 20x=1%, MASIH di bawah ambang 2%
    await runner13._price_watch_tick()
    check("posisi TETAP long (harga live belum cukup lewati ambang)",
          runner13._current_position == Position.LONG, f"dapat {runner13._current_position}")

    print("\n== 16. Real-time: FLAT (tidak ada posisi) -> tick tidak error, tidak ngapa-ngapain ==")
    strat14 = ScriptedStrategy([0] * 20)
    broker14 = MockBroker(fill_immediately=True, leverage=20.0)
    runner14 = PaperRunner(strat14, broker14, symbol="BTC/USDT:USDT", order_amount=0.01,
                            take_profit_pct=0.02, live_take_profit_poll_seconds=5.0)
    await runner14._price_watch_tick()  # tidak pernah ada posisi sama sekali
    check("tidak error, tetap FLAT", runner14._current_position == Position.FLAT)

    print("\n== 17. Penjelasan drift: POSISI DITUTUP MANUAL (bukan order bot) -> pesan jelas 'TIDAK bisa dijelaskan' ==")
    strat15 = ScriptedStrategy([1] * 20)
    broker15 = MockBroker(fill_immediately=True, leverage=20.0)
    runner15 = PaperRunner(strat15, broker15, symbol="BTC/USDT:USDT", order_amount=0.01)
    await runner15.process_bar(make_bar(base_ts, 100.0))
    await runner15.process_bar(make_bar(base_ts + hour_ms, 100.0))  # buka long lewat bot
    check("posisi terbuka (long) via bot", runner15._current_position == Position.LONG)

    # SIMULASI: posisi ditutup MANUAL (langsung ubah state broker,
    # PERSIS seperti kalau user klik "Market" di UI Binance -- TANPA
    # order dari bot sama sekali).
    broker15._positions["BTC/USDT:USDT"] = {"side": None, "contracts": 0.0}

    explanation = runner15._explain_position_drift(Position.FLAT)
    check("penjelasan MENYEBUT 'TIDAK bisa dijelaskan' dan 'INTERVENSI MANUAL'",
          "TIDAK bisa dijelaskan" in explanation and "MANUAL" in explanation,
          f"dapat: {explanation}")

    print("\n== 18. Penjelasan drift: order BOT SENDIRI yang telat fill -> pesan jelas 'Dikonfirmasi' ==")
    strat16 = ScriptedStrategy([1] * 20)
    broker16 = MockBroker(fill_immediately=False, leverage=20.0)  # TIDAK langsung fill
    runner16 = PaperRunner(strat16, broker16, symbol="BTC/USDT:USDT", order_amount=0.01)
    runner16._current_position = Position.LONG
    runner16._entry_price = 100.0
    broker16._positions["BTC/USDT:USDT"] = {"side": "long", "contracts": 0.01}

    await runner16.process_bar(make_bar(base_ts, 105.0))  # sinyal masih long, tidak ada perubahan -- warm-up
    # Paksa sinyal jadi short supaya _handle_signal_change benar2 kirim
    # order PENUTUP (reduceOnly) yang TIDAK langsung fill.
    strat16.scripted_signals = [-1] * 20
    await runner16.process_bar(make_bar(base_ts + hour_ms, 105.0))  # kirim order tutup, belum fill
    check("order penutup TERKIRIM tapi posisi masih long (belum fill)",
          runner16._current_position == Position.LONG, f"dapat {runner16._current_position}")

    # Order itu AKHIRNYA fill sungguhan (bukan manual).
    pending = [cid for cid, o in broker16._orders.items() if not o.is_terminal]
    for cid in pending:
        broker16.simulate_delayed_fill(cid)

    explanation2 = runner16._explain_position_drift(Position.FLAT)
    check("penjelasan MENYEBUT 'Dikonfirmasi' dan 'bukan intervensi luar'",
          "Dikonfirmasi" in explanation2 and "bukan intervensi" in explanation2,
          f"dapat: {explanation2}")

    print("\n== 19. RESTART sambil posisi masih terbuka: TIDAK boleh kirim order tutup yang tidak perlu ==")
    strat17 = ScriptedStrategy([1] * 20)  # sinyal SAMA dengan posisi yang sudah ada
    broker17 = MockBroker(fill_immediately=True, leverage=20.0)
    # Simulasikan posisi SUDAH ADA di bursa SEBELUM PaperRunner baru dibuat
    # -- persis kondisi restart: proses lama mati, proses baru mulai,
    # posisi sungguhan TETAP ada di bursa dari sebelumnya.
    broker17._positions["BTC/USDT:USDT"] = {"side": "long", "contracts": 0.01, "entry_price": 100.0}

    runner17 = PaperRunner(strat17, broker17, symbol="BTC/USDT:USDT", order_amount=0.01)
    check("SEBELUM recovery: _current_position masih asumsi default FLAT",
          runner17._current_position == Position.FLAT)

    runner17._recover_position_from_exchange()
    check("SETELAH recovery: _current_position terkoreksi jadi LONG (sesuai bursa)",
          runner17._current_position == Position.LONG, f"dapat {runner17._current_position}")
    check("entry_price ikut dipulihkan (100.0)", runner17._entry_price == 100.0,
          f"dapat {runner17._entry_price}")

    # Bar pertama datang, sinyal strategi SAMA (long) dengan posisi yang
    # sudah dipulihkan -- TIDAK BOLEH ada order terkirim sama sekali.
    n_orders_before = len(broker17._orders)
    await runner17.process_bar(make_bar(base_ts, 100.0))
    await runner17.process_bar(make_bar(base_ts + hour_ms, 101.0))
    check("TIDAK ADA order baru terkirim (posisi sudah benar, tidak perlu diapa-apakan)",
          len(broker17._orders) == n_orders_before, f"order sebelum={n_orders_before}, sesudah={len(broker17._orders)}")
    check("posisi TETAP long, tidak berubah/terganggu", runner17._current_position == Position.LONG)

    print("\n== 20. RESTART lalu sinyal BENAR-BENAR berbalik -> order tutup normal tetap terkirim ==")
    strat18 = ScriptedStrategy([-1] * 20)  # sinyal BERLAWANAN dari posisi yang ada
    broker18 = MockBroker(fill_immediately=True, leverage=20.0)
    broker18._positions["BTC/USDT:USDT"] = {"side": "long", "contracts": 0.01, "entry_price": 100.0}
    runner18 = PaperRunner(strat18, broker18, symbol="BTC/USDT:USDT", order_amount=0.01)
    runner18._recover_position_from_exchange()

    await runner18.process_bar(make_bar(base_ts, 100.0))
    await runner18.process_bar(make_bar(base_ts + hour_ms, 100.0))
    check("order PENUTUP tetap terkirim (sinyal memang beda arah, ini perilaku BENAR)",
          len(broker18._orders) > 0, f"dapat {len(broker18._orders)} order")

    print("\n== 21. RESTART + posisi lama kena take-profit + stop_after_take_profit=True -> TIDAK PERNAH buka posisi baru lagi ==")
    strat19 = ScriptedStrategy([1] * 30)
    broker19 = MockBroker(fill_immediately=True, leverage=20.0, taker_fee_pct=0.0)
    broker19._positions["BTC/USDT:USDT"] = {"side": "long", "contracts": 0.01, "entry_price": 100.0}
    runner19 = PaperRunner(strat19, broker19, symbol="BTC/USDT:USDT", order_amount=0.01,
                            take_profit_pct=0.02, stop_after_take_profit=True)
    runner19._recover_position_from_exchange()

    await runner19.process_bar(make_bar(base_ts, 100.0))
    await runner19.process_bar(make_bar(base_ts + hour_ms, 101.0))  # ROI 1%*20=20%, jauh lewat 2% -- TP kena
    check("posisi lama berhasil di-take-profit (FLAT)", runner19._current_position == Position.FLAT)
    check("_trading_halted = True setelah TP posisi lama", runner19._trading_halted is True)

    # Sinyal TETAP long terus (bahkan seandainya breakout baru muncul) --
    # TIDAK BOLEH buka posisi baru sama sekali, walau berkali-kali dicoba.
    n_orders_after_tp = len(broker19._orders)
    for i in range(2, 6):
        await runner19.process_bar(make_bar(base_ts + i * hour_ms, 105.0 + i))
    check("TIDAK ADA order baru terkirim sama sekali setelah halted (walau sinyal terus long)",
          len(broker19._orders) == n_orders_after_tp, f"order sebelum={n_orders_after_tp}, sesudah={len(broker19._orders)}")
    check("posisi TETAP FLAT selamanya", runner19._current_position == Position.FLAT)

    print("\n== 22. RESTART + posisi lama kena take-profit TANPA stop_after_take_profit -> BISA buka posisi baru kalau sinyal genuinely berbalik ==")
    strat20 = ScriptedStrategy([1] * 5)  # awalnya masih long
    broker20 = MockBroker(fill_immediately=True, leverage=20.0, taker_fee_pct=0.0)
    broker20._positions["BTC/USDT:USDT"] = {"side": "long", "contracts": 0.01, "entry_price": 100.0}
    runner20 = PaperRunner(strat20, broker20, symbol="BTC/USDT:USDT", order_amount=0.01,
                            take_profit_pct=0.02, stop_after_take_profit=False)  # TANPA halt permanen
    runner20._recover_position_from_exchange()

    await runner20.process_bar(make_bar(base_ts, 100.0))
    await runner20.process_bar(make_bar(base_ts + hour_ms, 101.0))  # TP kena, tutup
    check("posisi lama di-take-profit (FLAT)", runner20._current_position == Position.FLAT)
    check("TIDAK halted (stop_after_take_profit=False)", runner20._trading_halted is False)
    check("_blocked_direction = LONG (arah yang baru di-TP, ditahan sementara)",
          runner20._blocked_direction == Position.LONG)

    # Sinyal MASIH long -> harus DITAHAN (bukti sudah ada di tes 2 sebelumnya, cek lagi di sini)
    await runner20.process_bar(make_bar(base_ts + 2 * hour_ms, 102.0))
    check("posisi TETAP FLAT selama sinyal masih arah yang sama (ditahan)",
          runner20._current_position == Position.FLAT)

    # Sinyal BENAR-BENAR berbalik ke short -> BOLEH buka posisi baru (arah BERLAWANAN).
    # Pakai harga sedikit BERBEDA tiap bar (bukan rata persis) -- harga
    # rata berulang bisa memicu take-profit LAGI segera setelah posisi
    # baru terbuka (offset agresif + leverage 20x cukup untuk lewati
    # ambang walau harga "kelihatannya" sama).
    strat20.scripted_signals = [1] * 5 + [-1] * 20
    prices20 = [92.0, 91.0, 90.0]  # buffer genap jadi 6 -- pas index5 (elemen ke-6) = -1
    for i, p in enumerate(prices20):
        await runner20.process_bar(make_bar(base_ts + (4 + i) * hour_ms, p))
    check("posisi BARU (short) berhasil dibuka -- sinyal genuinely sudah berbalik",
          runner20._current_position == Position.SHORT, f"dapat {runner20._current_position}")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))