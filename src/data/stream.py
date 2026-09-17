"""
data/stream.py

Sambungan WebSocket ke harga live bursa, dengan penyambungan ulang
OTOMATIS kalau koneksi putus. Dipakai runner/paper.py untuk dapat bar
harga real-time -- bukan polling REST API berulang-ulang (lebih
lambat, lebih gampang kena rate limit).

Pakai ccxt.pro (modul async ccxt khusus WebSocket) -- BUKAN ccxt biasa
(dipakai execution/broker.py, REST-only). INI SATU-SATUNYA FILE yang
boleh mengimpor ccxt.pro -- pemisahan yang sama dan alasan yang sama
seperti broker.py: satu titik yang perlu diperbaiki kalau ada
perubahan di sisi bursa.
"""

import asyncio
from collections.abc import AsyncIterator

#: Jeda sebelum mencoba sambung ulang, naik bertahap (exponential
#: backoff) -- supaya tidak membanjiri bursa dengan percobaan beruntun
#: kalau jaringan/bursa memang lagi bermasalah lama. Nilai terakhir
#: diulang terus kalau masih gagal juga (bukan naik tanpa batas).
RECONNECT_DELAYS_SECONDS = [1, 2, 5, 10, 30, 60]


class PriceStream:
    """
    Sambungan WebSocket untuk satu simbol + satu timeframe.

    Dipakai lewat watch_bars() -- async generator yang menghasilkan
    bar baru begitu tersedia, dan menyambung ulang otomatis kalau
    koneksi putus, TANPA melempar exception ke pemanggil kecuali
    dihentikan manual lewat stop(). Pemanggil (runner/paper.py) cukup:

        async for bar in stream.watch_bars():
            ...proses bar...

    -- tidak perlu tahu-menahu soal reconnect sama sekali, itu
    tanggung jawab kelas ini.
    """

    def __init__(self, exchange_id: str = "binanceusdm", symbol: str = "BTC/USDT:USDT", timeframe: str = "2h"):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self._stopped = False
        self._reconnect_attempt = 0

    def stop(self) -> None:
        """Hentikan loop watch_bars() dengan bersih di iterasi berikutnya."""
        self._stopped = True

    def _reconnect_delay(self) -> float:
        """
        Fungsi murni (tidak menyentuh jaringan/waktu sungguhan) --
        sengaja dipisah dari watch_bars() supaya progresi backoff-nya
        bisa dites tanpa perlu simulasi koneksi WebSocket sungguhan.
        """
        idx = min(self._reconnect_attempt, len(RECONNECT_DELAYS_SECONDS) - 1)
        return RECONNECT_DELAYS_SECONDS[idx]

    async def watch_bars(self) -> AsyncIterator[dict]:
        """
        PENTING: watch_ohlcv() dari ccxt.pro mengirim update TERUS-
        MENERUS ke candle yang MASIH BERJALAN (belum closed), bukan
        cuma sekali saat candle itu beneran tutup. Kalau setiap update
        itu diperlakukan sebagai "bar baru", strategi akan dievaluasi
        ulang berkali-kali per detik dalam satu candle yang sama --
        ditemukan di lapangan: ini menyebabkan order dibatalkan &
        dikirim ulang secepat itu sampai bursa belum sempat memproses
        pembatalan sebelum order baru datang, bikin order baru ditolak
        karena tabrakan dengan order lama yang (menurut bursa) masih
        aktif.

        Solusinya: deteksi pergantian timestamp bar TERAKHIR. Selama
        timestamp-nya sama dengan sebelumnya, candle itu masih
        berjalan -- simpan versi terbarunya tapi JANGAN yield. Begitu
        timestamp bar terakhir berubah ke periode baru, itu tandanya
        bar SEBELUMNYA sudah benar-benar closed -- baru yield bar itu.
        """
        import ccxt.pro as ccxtpro  # diimpor di dalam method -- pola sama
        # seperti broker.py, supaya modul lain yang cuma perlu
        # RECONNECT_DELAYS_SECONDS dsb tidak wajib punya ccxt.pro terpasang.

        exchange_class = getattr(ccxtpro, self.exchange_id)
        exchange = exchange_class({"enableRateLimit": True})

        last_seen_timestamp = None
        pending_bar = None  # data bar TERAKHIR yang diamati, mungkin masih berjalan

        try:
            while not self._stopped:
                try:
                    ohlcv = await exchange.watch_ohlcv(self.symbol, self.timeframe)
                    self._reconnect_attempt = 0  # koneksi sehat lagi -- reset hitungan backoff

                    if not ohlcv:
                        continue

                    last = ohlcv[-1]
                    current_ts = last[0]

                    if last_seen_timestamp is None:
                        # Observasi pertama -- belum tahu ini bar yang
                        # sama dengan update berikutnya atau tidak.
                        # Tunggu satu putaran lagi sebelum bisa
                        # memutuskan apa pun.
                        last_seen_timestamp = current_ts
                        pending_bar = last
                        continue

                    if current_ts != last_seen_timestamp:
                        # Timestamp bar TERAKHIR sudah maju ke periode
                        # baru -- berarti pending_bar (bar sebelumnya)
                        # sudah benar-benar CLOSED. Baru sekarang yield.
                        closed = pending_bar
                        yield {
                            "timestamp": closed[0],
                            "open": closed[1],
                            "high": closed[2],
                            "low": closed[3],
                            "close": closed[4],
                            "volume": closed[5],
                        }
                        last_seen_timestamp = current_ts
                        pending_bar = last
                    else:
                        # Masih candle yang sama, cuma nilai OHLC-nya
                        # ter-update (candle masih berjalan) -- simpan
                        # versi terbaru, TAPI JANGAN yield. Strategi
                        # tidak dievaluasi ulang untuk update semacam ini.
                        pending_bar = last

                except Exception as e:
                    delay = self._reconnect_delay()
                    print(f"  [stream] Koneksi terputus ({e}) -- sambung ulang dalam {delay} detik...")
                    self._reconnect_attempt += 1
                    await asyncio.sleep(delay)
        finally:
            await exchange.close()


if __name__ == "__main__":
    # Uji asap TANPA jaringan -- cuma memastikan progresi backoff
    # reconnect masuk akal (naik bertahap, lalu mendatar di nilai
    # maksimum, tidak pernah melonjak tanpa batas).
    print("=== Uji progresi reconnect delay (tanpa jaringan) ===\n")
    stream = PriceStream()
    for attempt in range(8):
        stream._reconnect_attempt = attempt
        print(f"  percobaan ke-{attempt + 1}: tunggu {stream._reconnect_delay()} detik")

    print("\nCatatan: watch_bars() WAJIB dites dengan koneksi jaringan")
    print("sungguhan -- sandbox ini tidak punya akses ke bursa mana pun.")
    print("Contoh uji manual:")
    print("  python -c \"")
    print("import asyncio")
    print("from src.data.stream import PriceStream")
    print("async def main():")
    print("    s = PriceStream(symbol='BTC/USDT:USDT', timeframe='1m')")
    print("    async for bar in s.watch_bars():")
    print("        print(bar)")
    print("        break  # cukup satu bar untuk tes")
    print("asyncio.run(main())\"")