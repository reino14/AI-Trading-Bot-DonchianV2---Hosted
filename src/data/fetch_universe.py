"""
data/fetch_universe.py

Tarik data historis 100 (atau berapa pun) aset teratas dari Binance,
untuk spot dan/atau USDT-M perpetual futures, jadi panel LEBAR yang
sejajar -- format yang dibutuhkan portfolio_engine.py.

SUMBER DATA: BINANCE MAINNET PUBLIK, BUKAN DEMO
-------------------------------------------------
Skrip ini memanggil api.binance.com / fapi.binance.com (via ccxt
`binance()` dan `binanceusdm()` biasa, TANPA set_sandbox_mode). Ini
BEDA dari VPS live Anda yang sengaja memakai demo-api.binance.com untuk
paper trading -- endpoint demo itu untuk EKSEKUSI ORDER simulasi,
riwayat candle di sana tidak dijamin lengkap/akurat untuk 1-4 tahun ke
belakang. Data historis untuk backtest wajib dari harga pasar sungguhan.

Semua endpoint yang dipakai di sini (tickers, ohlcv, funding rate
history) PUBLIK -- tidak perlu API key. `enableRateLimit=True` membuat
ccxt otomatis menjaga jarak antar panggilan sesuai batas Binance.

KETERBATASAN YANG PERLU ANDA SADARI: UNIVERSE INI TIDAK SEPENUHNYA
POINT-IN-TIME
---------------------------------------------------------------------
Fungsi get_top_symbols() memilih top-N berdasarkan volume 24 jam SAAT
SKRIP DIJALANKAN, lalu menarik riwayatnya ke belakang. Ini menangani
SEBAGIAN masalah survivorship: aset yang baru listing di tengah
periode akan punya NaN sebelum tanggal listingnya (lihat
build_wide_panel), jadi portfolio_engine.py sudah otomatis
memperlakukannya dengan benar (tidak dipegang sebelum ada).

TAPI aset yang dulu masuk top-100 lalu ANJLOK atau DI-DELIST sebelum
hari ini tidak akan pernah masuk daftar sama sekali -- karena kita
memilih berdasarkan volume HARI INI, bukan volume di setiap tanggal
historis. Ini survivorship bias yang SESUNGGUHNYA, dan tidak
sepenuhnya bisa diperbaiki lewat API publik Binance (tidak ada
endpoint resmi untuk "daftar top-100 per tanggal X di masa lalu").
Efeknya condong membuat hasil backtest terlihat SEDIKIT LEBIH BAIK
dari kenyataan, karena aset yang gagal total tersaring keluar.
Mitigasi realistis: jangan percaya penuh angka Sharpe dari sini,
dan kalau memungkinkan silangkan dengan sumber independen (mis.
CoinMarketCap historical snapshot) nanti -- bukan pekerjaan hari ini.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd


def _make_exchange(market: str) -> ccxt.Exchange:
    """market: 'spot' atau 'futures'."""
    if market == "spot":
        return ccxt.binance({"enableRateLimit": True})
    if market == "futures":
        return ccxt.binanceusdm({"enableRateLimit": True})
    raise ValueError(f"market harus 'spot' atau 'futures', dapat: {market!r}")


def get_top_symbols(
    exchange: ccxt.Exchange,
    n: int,
    market_type: str,
    quote: str = "USDT",
) -> list[str]:
    """
    Ambil n simbol teratas berdasarkan volume quote 24 jam SAAT INI.
    Lihat catatan point-in-time di docstring modul -- ini bukan
    peringkat historis.

    market_type: 'spot' atau 'futures'. WAJIB DISEBUT EKSPLISIT DI SINI,
    tidak cukup mengandalkan exchange mana yang dipanggil.

    KENAPA INI BERUBAH DARI VERSI SEBELUMNYA (temuan dari run sungguhan)
    ----------------------------------------------------------------------
    Versi sebelumnya memakai satu kondisi longgar:
        m.get("type") == "spot" or m.get("swap")
    dan dipakai sama untuk kedua jenis exchange. Run sungguhan Nero
    menunjukkan ini SALAH: pemanggilan untuk 'spot' (exchange
    ccxt.binance()) tetap mengembalikan simbol berformat "BTC/USDT:USDT"
    -- notasi ccxt untuk kontrak derivatif bersettlement, TIDAK PERNAH
    dipakai untuk pasangan spot sungguhan -- dan persis 100 simbol yang
    sama, urutan sama persis, dengan hasil 'futures'. Apa pun alasan
    load_markets() exchange spot ikut menyertakan entri swap (bisa
    produk baru Binance, bisa perilaku versi ccxt), filter longgar di
    atas tidak pernah menyaringnya.

    Perbaikannya: JANGAN percaya field 'type' string yang bisa ambigu.
    Setiap market ccxt WAJIB punya flag boolean eksplisit 'spot' dan
    'swap' (bagian dari spesifikasi unified market structure ccxt) --
    market spot sungguhan HARUS punya spot=True DAN contract=False/absen.
    Market futures perpetual HARUS punya swap=True. Dua syarat ini
    saling eksklusif secara definisi, jadi kontaminasi silang seperti
    yang terjadi di atas tidak mungkin lolos lagi.
    """
    if market_type not in ("spot", "futures"):
        raise ValueError(f"market_type harus 'spot' atau 'futures', dapat: {market_type!r}")

    markets = exchange.load_markets()

    if market_type == "spot":
        candidates = [
            m["symbol"]
            for m in markets.values()
            if m.get("active")
            and m.get("quote") == quote
            and m.get("spot") is True
            and not m.get("contract", False)
        ]
    else:  # futures
        candidates = [
            m["symbol"]
            for m in markets.values()
            if m.get("active")
            and m.get("quote") == quote
            and m.get("swap") is True
            and m.get("linear", True)  # buang inverse contracts
        ]

    # SEBELUMNYA: exchange.fetch_tickers(candidates) -- mengirim SELURUH
    # daftar simbol (bisa 300-400+ untuk market spot) sebagai parameter
    # URL. Binance membatasi panjang URL permintaan, dan run sungguhan
    # Nero terbukti melampauinya: HTTP 414 "Request-URI Too Large".
    #
    # PERBAIKAN: minta SEMUA ticker sekaligus (satu panggilan, tanpa
    # daftar simbol di URL sama sekali -- endpoint /ticker/24hr Binance
    # mendukung ini), lalu saring di sisi kita ke `candidates`. Ini
    # BUKAN sekadar menghindari limit URL -- ini juga lebih hemat:
    # satu panggilan API untuk seluruh bursa, dibanding harus memecah
    # candidates jadi beberapa batch permintaan.
    all_tickers = exchange.fetch_tickers()
    candidate_set = set(candidates)
    tickers = {s: t for s, t in all_tickers.items() if s in candidate_set}

    ranked = sorted(
        tickers.items(),
        key=lambda kv: kv[1].get("quoteVolume") or 0.0,
        reverse=True,
    )
    return [symbol for symbol, _ in ranked[:n]]


def fetch_ohlcv_full(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Tarik OHLCV lengkap lewat pagination -- Binance membatasi ~1000 bar
    per panggilan, jadi satu panggilan tidak pernah cukup untuk 2-3
    tahun data harian (itu cuma perlu ~1x, tapi untuk timeframe lebih
    pendek ini WAJIB).

    Return DataFrame kosong (bukan error) kalau simbol tidak punya data
    di rentang ini -- caller yang memutuskan mau diapakan.
    """
    until_ms = until_ms or exchange.milliseconds()
    all_rows: list[list] = []
    cursor = since_ms

    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break  # jaga-jaga supaya tidak infinite loop kalau API aneh
        cursor = last_ts + 1
        # SENGAJA TIDAK berhenti hanya karena len(batch) < limit: server
        # boleh membatasi ukuran halaman sendiri di bawah limit yang
        # diminta (Binance kadang begini untuk simbol tertentu/timeframe
        # pendek). Berhenti hanya lewat batch kosong atau cursor mentok
        # di until_ms -- satu panggilan ekstra di akhir yang balik kosong
        # jauh lebih murah daripada diam-diam kehilangan data di tengah.

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        all_rows, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    df = df.drop_duplicates(subset="ts").set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df[df.index < pd.to_datetime(until_ms, unit="ms", utc=True)]


def fetch_funding_history(
    exchange: ccxt.Exchange,
    symbol: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = 1000,
) -> pd.Series:
    """
    Tarik riwayat funding rate (khusus futures) lewat pagination.
    Funding dibayar tiap 8 jam -- balikan Series bernilai bps per
    kejadian funding, index datetime UTC di saat funding terjadi.
    """
    until_ms = until_ms or exchange.milliseconds()
    all_rows: list[dict] = []
    cursor = since_ms

    while cursor < until_ms:
        batch = exchange.fetch_funding_rate_history(
            symbol, since=cursor, limit=limit
        )
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        # Sama seperti fetch_ohlcv_full: tidak berhenti karena
        # len(batch) < limit, lihat penjelasan di sana.

    if not all_rows:
        return pd.Series(dtype=float)

    idx = pd.to_datetime([r["timestamp"] for r in all_rows], unit="ms", utc=True)
    vals = [r["fundingRate"] * 10_000.0 for r in all_rows]  # fraksi -> bps
    return pd.Series(vals, index=idx).sort_index()


def build_wide_panel(
    per_symbol: dict[str, pd.DataFrame],
    field: str,
) -> pd.DataFrame:
    """
    Gabungkan {symbol: OHLCV df} jadi satu DataFrame lebar untuk satu
    kolom (mis. 'close'). index = union seluruh tanggal, kolom = simbol.

    Baris SEBELUM tanggal listing suatu simbol otomatis NaN (bukan 0)
    karena outer join -- ini yang membuat portfolio_engine.py bisa
    menahan bobotnya di nol dengan benar sebelum aset itu ada.
    """
    series = {sym: df[field] for sym, df in per_symbol.items() if not df.empty}
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series)
    return panel.sort_index()


def align_funding_to_bars(
    funding_by_symbol: dict[str, pd.Series],
    bar_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Funding terjadi tiap 8 jam, bar utama (mis. harian) lebih jarang.
    Jumlahkan seluruh funding yang jatuh DI DALAM satu bar -- kejadian
    funding di waktu t masuk ke bar_index[i] kalau
    bar_index[i] <= t < bar_index[i+1] (bar terakhir menampung semua
    sisanya).

    Dilakukan lewat searchsorted, BUKAN pd.infer_freq -- infer_freq
    butuh minimal 3 titik dan gagal untuk index pendek atau tidak rata,
    padahal pemetaan "kejadian ini masuk bar yang mana" tidak butuh
    tahu frekuensinya sama sekali, cukup posisi relatif terhadap
    batas-batas bar_index itu sendiri.

    Baris pertama tiap simbol yang belum punya funding jatuh di
    dalamnya akan NaN, bukan 0 -- fillna(0.0) SENGAJA dilakukan
    caller, bukan di sini, supaya perbedaan "belum ada data" vs
    "funding memang nol" tetap terlihat sampai titik itu.
    """
    bar_values = bar_index.values
    n = len(bar_index)
    cols: dict[str, pd.Series] = {}
    for sym, s in funding_by_symbol.items():
        if s.empty:
            continue
        # side='right' lalu -1: kejadian di waktu t masuk ke bar terakhir
        # yang batas awalnya <= t. Kejadian sebelum bar_index[0] dijepit
        # ke bar 0 (seharusnya tidak terjadi kalau since_ms konsisten,
        # tapi lebih aman daripada indeks negatif).
        positions = np.searchsorted(bar_values, s.index.values, side="right") - 1
        positions = np.clip(positions, 0, n - 1)
        grouped = pd.Series(s.values, index=bar_index[positions]).groupby(level=0).sum()
        cols[sym] = grouped.reindex(bar_index)
    if not cols:
        return pd.DataFrame(index=bar_index)
    df = pd.DataFrame(cols)
    return df.reindex(bar_index)


def run(
    market: str,
    n_symbols: int,
    timeframe: str,
    lookback_days: int,
    out_dir: Path,
    sleep_between_symbols: float = 0.2,
) -> None:
    markets_to_pull = ["spot", "futures"] if market == "both" else [market]
    until_ms = ccxt.Exchange().milliseconds()
    since_ms = until_ms - lookback_days * 86_400_000

    for m in markets_to_pull:
        print(f"\n=== {m.upper()} ===")
        exchange = _make_exchange(m)
        symbols = get_top_symbols(exchange, n_symbols, market_type=m)
        print(f"  {len(symbols)} simbol teratas (berdasarkan volume SAAT INI, "
              f"lihat catatan point-in-time di kepala file ini)")

        ohlcv: dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols, 1):
            df = fetch_ohlcv_full(exchange, sym, timeframe, since_ms, until_ms)
            ohlcv[sym] = df
            if df.empty:
                print(f"  [{i}/{len(symbols)}] {sym}: KOSONG -- dilewati")
            else:
                print(f"  [{i}/{len(symbols)}] {sym}: {len(df)} bar, "
                      f"mulai {df.index[0].date()}")
            time.sleep(sleep_between_symbols)

        close = build_wide_panel(ohlcv, "close")
        volume = build_wide_panel(ohlcv, "volume")

        out_dir.mkdir(parents=True, exist_ok=True)
        close.to_parquet(out_dir / f"{m}_close.parquet")
        volume.to_parquet(out_dir / f"{m}_volume.parquet")
        print(f"  Panel close : {close.shape} -> {out_dir / f'{m}_close.parquet'}")
        print(f"  Panel volume: {volume.shape} -> {out_dir / f'{m}_volume.parquet'}")

        if m == "futures":
            funding: dict[str, pd.Series] = {}
            for i, sym in enumerate(symbols, 1):
                funding[sym] = fetch_funding_history(exchange, sym, since_ms, until_ms)
                time.sleep(sleep_between_symbols)
            funding_panel = align_funding_to_bars(funding, close.index)
            funding_panel.to_parquet(out_dir / "futures_funding_bps.parquet")
            print(f"  Panel funding: {funding_panel.shape} -> "
                  f"{out_dir / 'futures_funding_bps.parquet'}")

    print("\nSelesai. Ingat: cross-check n_trials Anda kalau nanti panel ini "
          "dipakai untuk menguji banyak varian strategi sekaligus.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", choices=["spot", "futures", "both"], default="both")
    p.add_argument("--n", type=int, default=100, help="jumlah simbol teratas")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--days", type=int, default=900, help="lookback dalam hari")
    p.add_argument("--out-dir", default="data/raw")
    args = p.parse_args()

    run(
        market=args.market,
        n_symbols=args.n,
        timeframe=args.timeframe,
        lookback_days=args.days,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()