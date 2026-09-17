"""
scripts/smoke_fetch_universe.py

Uji asap src/data/fetch_universe.py TANPA menyentuh Binance sungguhan --
memakai bursa palsu yang meniru bentuk respons ccxt persis (list of
[ts, o, h, l, c, v] untuk OHLCV, list of dict untuk funding). Ini
menguji LOGIKA pagination, penyelarasan panel, dan penanganan tanggal
listing berbeda -- bukan konektivitas sungguhan ke Binance.

Konektivitas sungguhan WAJIB diuji terpisah dengan menjalankan
fetch_universe.py langsung di mesin yang punya akses ke Binance.
"""

import sys

import numpy as np
import pandas as pd

from src.data.fetch_universe import (
    align_funding_to_bars,
    build_wide_panel,
    fetch_funding_history,
    fetch_ohlcv_full,
    get_top_symbols,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class _FakeExchange:
    """
    Meniru bagian ccxt.Exchange yang dipakai fetch_universe.py.

    _rows adalah data OHLCV LENGKAP yang "sungguhan ada" di bursa palsu
    ini -- fetch_ohlcv memotongnya sesuai `since`/`limit` PERSIS seperti
    cara Binance membatasi tiap panggilan, supaya pagination sungguhan
    teruji, bukan cuma dipercaya.
    """

    def __init__(self, rows_by_symbol: dict[str, list[list]], page_limit: int = 5):
        self._rows = rows_by_symbol
        self._page_limit = page_limit
        self.call_log: list[tuple] = []

    def milliseconds(self) -> int:
        return 2_000_000_000_000  # tanggal jauh di masa depan, cukup untuk tes

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.call_log.append((symbol, since, limit))
        rows = [r for r in self._rows.get(symbol, []) if r[0] >= since]
        rows.sort(key=lambda r: r[0])
        return rows[: self._page_limit]  # bursa sungguhan juga membatasi per panggilan


class _FakeFundingExchange:
    def __init__(self, funding_by_symbol: dict[str, list[dict]], page_limit: int = 4):
        self._funding = funding_by_symbol
        self._page_limit = page_limit

    def fetch_funding_rate_history(self, symbol, since, limit):
        rows = [r for r in self._funding.get(symbol, []) if r["timestamp"] >= since]
        rows.sort(key=lambda r: r["timestamp"])
        return rows[: self._page_limit]


class _FakeMarketExchange:
    """Meniru exchange.load_markets() dan exchange.fetch_tickers()."""

    def __init__(self, markets: dict[str, dict], tickers: dict[str, dict]):
        self._markets = markets
        self._tickers = tickers

    def load_markets(self):
        return self._markets

    def fetch_tickers(self, symbols=None):
        if symbols is None:
            return dict(self._tickers)
        return {s: self._tickers[s] for s in symbols if s in self._tickers}


def _mk_market(symbol, quote, spot, swap, contract, linear=True, active=True):
    return {
        "symbol": symbol,
        "active": active,
        "quote": quote,
        "spot": spot,
        "swap": swap,
        "contract": contract,
        "linear": linear,
    }


DAY_MS = 86_400_000


def make_ohlcv_rows(start_day: int, n_days: int, base_price: float = 100.0) -> list[list]:
    rows = []
    price = base_price
    rng = np.random.default_rng(abs(hash((start_day, n_days))) % (2**31))
    for i in range(n_days):
        ts = (start_day + i) * DAY_MS
        price *= 1 + rng.normal(0, 0.01)
        rows.append([ts, price, price * 1.01, price * 0.99, price, 1000.0 + i])
    return rows


def main() -> int:
    print("== 0. get_top_symbols: kontaminasi spot/futures (BUG SUNGGUHAN yang ditemukan) ==")
    # Meniru PERSIS anomali dari run Nero: satu daftar market berisi
    # simbol spot genuine ("BTC/USDT") BERCAMPUR dengan simbol swap
    # ("BTC/USDT:USDT", "XAU/USDT:USDT") -- filter lama akan meloloskan
    # keduanya untuk seleksi 'spot', filter baru wajib memisahkannya.
    contaminated_markets = {
        "BTC/USDT": _mk_market("BTC/USDT", "USDT", spot=True, swap=False, contract=False),
        "ETH/USDT": _mk_market("ETH/USDT", "USDT", spot=True, swap=False, contract=False),
        "BTC/USDT:USDT": _mk_market("BTC/USDT:USDT", "USDT", spot=False, swap=True, contract=True),
        "XAU/USDT:USDT": _mk_market("XAU/USDT:USDT", "USDT", spot=False, swap=True, contract=True),
        "NVDA/USDT:USDT": _mk_market("NVDA/USDT:USDT", "USDT", spot=False, swap=True, contract=True),
    }
    contaminated_tickers = {
        sym: {"quoteVolume": 1_000_000.0 - i}
        for i, sym in enumerate(contaminated_markets)
    }
    fake_ex = _FakeMarketExchange(contaminated_markets, contaminated_tickers)

    spot_result = get_top_symbols(fake_ex, n=10, market_type="spot")
    futures_result = get_top_symbols(fake_ex, n=10, market_type="futures")

    check(
        "seleksi 'spot' TIDAK mengandung simbol berformat swap (':USDT')",
        all(":" not in s for s in spot_result),
        f"hasil={spot_result}",
    )
    check(
        "seleksi 'spot' berisi persis BTC/USDT dan ETH/USDT",
        set(spot_result) == {"BTC/USDT", "ETH/USDT"},
        f"hasil={spot_result}",
    )
    check(
        "seleksi 'futures' TIDAK mengandung simbol spot murni",
        set(futures_result).isdisjoint({"BTC/USDT", "ETH/USDT"}),
        f"hasil={futures_result}",
    )
    check(
        "seleksi 'futures' berisi ketiga simbol swap",
        set(futures_result) == {"BTC/USDT:USDT", "XAU/USDT:USDT", "NVDA/USDT:USDT"},
        f"hasil={futures_result}",
    )
    print("    (kalau salah satu GAGAL di sini, bug spot/futures yang sama akan")
    print("     lolos lagi -- ini persis pola yang ditemukan di run sungguhan Nero)")

    print("\n== 1. Pagination fetch_ohlcv_full menyatukan seluruh halaman ==")
    rows = make_ohlcv_rows(start_day=0, n_days=23)  # 23 bar, page_limit=5 -> 5 halaman
    ex = _FakeExchange({"FAKE/USDT": rows}, page_limit=5)
    df = fetch_ohlcv_full(ex, "FAKE/USDT", "1d", since_ms=0, until_ms=23 * DAY_MS)
    check("jumlah bar lengkap tertarik semua", len(df) == 23, f"dapat {len(df)}")
    check("tidak ada duplikat timestamp", df.index.is_unique)
    check("terurut naik", df.index.is_monotonic_increasing)
    check(
        "lebih dari satu panggilan API dilakukan (pagination sungguhan terjadi)",
        len(ex.call_log) >= 5,
        f"{len(ex.call_log)} panggilan",
    )

    print("\n== 2. Simbol tanpa data mengembalikan DataFrame kosong, bukan error ==")
    df_empty = fetch_ohlcv_full(ex, "TIDAKADA/USDT", "1d", since_ms=0, until_ms=23 * DAY_MS)
    check("kosong dan bertipe DataFrame", df_empty.empty and isinstance(df_empty, pd.DataFrame))

    print("\n== 3. build_wide_panel: tanggal listing berbeda -> NaN yang benar ==")
    early = make_ohlcv_rows(start_day=0, n_days=30)  # ada sejak hari 0
    late = make_ohlcv_rows(start_day=20, n_days=10)  # baru listing hari 20
    ex2 = _FakeExchange({"EARLY/USDT": early, "LATE/USDT": late}, page_limit=100)
    df_early = fetch_ohlcv_full(ex2, "EARLY/USDT", "1d", 0, 30 * DAY_MS)
    df_late = fetch_ohlcv_full(ex2, "LATE/USDT", "1d", 0, 30 * DAY_MS)
    panel = build_wide_panel({"EARLY/USDT": df_early, "LATE/USDT": df_late}, "close")
    check("kolom LATE sebelum listing adalah NaN", panel["LATE/USDT"].iloc[:20].isna().all())
    check("kolom LATE setelah listing terisi", panel["LATE/USDT"].iloc[20:].notna().all())
    check("kolom EARLY terisi penuh", panel["EARLY/USDT"].notna().all())
    check(
        "NaN bukan nol -- tidak akan disalahartikan sebagai harga nol",
        not (panel["LATE/USDT"].iloc[:20] == 0).any(),
    )

    print("\n== 4. Pagination funding rate ==")
    funding_rows = [
        {"timestamp": i * 8 * 3_600_000, "fundingRate": 0.0001 * ((-1) ** i)}
        for i in range(15)  # 15 kejadian funding, page_limit=4 -> 4 halaman
    ]
    fex = _FakeFundingExchange({"FAKE/USDT": funding_rows}, page_limit=4)
    fs = fetch_funding_history(fex, "FAKE/USDT", since_ms=0, until_ms=15 * 8 * 3_600_000)
    check("seluruh 15 kejadian funding tertarik", len(fs) == 15, f"dapat {len(fs)}")
    check(
        "konversi fraksi ke bps benar (0.0001 -> 1.0 bps)",
        abs(abs(fs.iloc[0]) - 1.0) < 1e-9,
        f"nilai={fs.iloc[0]}",
    )

    print("\n== 5. align_funding_to_bars: penjumlahan 3x funding harian per bar ==")
    bar_index = pd.date_range("1970-01-01", periods=2, freq="1D", tz="UTC")
    aligned = align_funding_to_bars({"FAKE/USDT": fs}, bar_index)
    check("index sejajar dengan bar_index", list(aligned.index) == list(bar_index))
    check(
        "hari pertama = jumlah 3 funding pertama",
        abs(aligned["FAKE/USDT"].iloc[0] - fs.iloc[:3].sum()) < 1e-9,
        f"dapat={aligned['FAKE/USDT'].iloc[0]:.4f}, harus={fs.iloc[:3].sum():.4f}",
    )

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS (logika teruji, konektivitas Binance BELUM diuji).")
    print("Langkah berikutnya: jalankan fetch_universe.py sungguhan di mesin")
    print("Anda dan kirim log + bentuk file parquet yang dihasilkan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())