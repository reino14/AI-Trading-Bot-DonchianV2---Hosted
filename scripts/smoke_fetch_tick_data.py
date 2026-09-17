"""
scripts/smoke_fetch_tick_data.py

Uji asap src/data/fetch_tick_data.py TANPA menyentuh data.binance.vision
sungguhan -- fetch_data() ditiru (monkeypatch) supaya menguji LOGIKA
pemecahan bulan, penulisan parquet incremental, dan penanganan periode
kosong/gagal. Konektivitas sungguhan WAJIB diuji terpisah oleh Nero.
"""

import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

import src.data.fetch_tick_data as mod
from src.data.fetch_tick_data import _month_range, fetch_symbol_tick_data

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


@dataclass
class _FakeResult:
    data: pd.DataFrame
    missing: list
    failed: list


def main() -> int:
    print("== 1. _month_range: pemecahan rentang lintas tahun ==")
    periods = _month_range(date(2024, 11, 15), date(2025, 2, 10))
    labels = [(s.isoformat(), e.isoformat()) for s, e in periods]
    check("4 bulan terpecah benar", len(periods) == 4, f"dapat {len(periods)}: {labels}")
    check("bulan pertama mulai dari tanggal start, bukan awal bulan",
          periods[0][0] == date(2024, 11, 15))
    check("bulan terakhir berhenti di tanggal end, bukan akhir bulan",
          periods[-1][1] == date(2025, 2, 10))
    check("bulan tengah penuh sebulan",
          periods[1] == (date(2024, 12, 1), date(2024, 12, 31)))

    print("\n== 2. _month_range: rentang dalam satu bulan saja ==")
    periods_single = _month_range(date(2025, 6, 5), date(2025, 6, 20))
    check("cuma 1 periode", len(periods_single) == 1, f"dapat {periods_single}")

    print("\n== 3. fetch_symbol_tick_data: tulis parquet per bulan ==")
    call_log = []

    def fake_fetch_data(ticker, start_date, end_date, market, data_type):
        call_log.append((ticker, start_date, end_date, market, data_type))
        df = pd.DataFrame({
            "price": [100.0, 101.0, 99.5],
            "quantity": [1.0, 2.0, 0.5],
            "is_buyer_maker": [True, False, True],
        })
        return _FakeResult(data=df, missing=[], failed=[])

    mod.fetch_data = fake_fetch_data  # monkeypatch modul yang diimpor fetch_tick_data.py

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        fetch_symbol_tick_data("BTCUSDT", date(2025, 1, 1), date(2025, 2, 15), out_dir)

        check("2 panggilan fetch_data (Jan, Feb)", len(call_log) == 2, f"{call_log}")
        jan_path = out_dir / "aggTrades" / "BTCUSDT" / "2025-01.parquet"
        feb_path = out_dir / "aggTrades" / "BTCUSDT" / "2025-02.parquet"
        check("file Januari tertulis", jan_path.exists())
        check("file Februari tertulis", feb_path.exists())

        saved = pd.read_parquet(jan_path)
        check("isi parquet sesuai data yang di-fetch", len(saved) == 3, f"{len(saved)} baris")

        print("\n== 4. Bulan yang sudah ada file dilewati, tidak fetch ulang ==")
        call_log.clear()
        fetch_symbol_tick_data("BTCUSDT", date(2025, 1, 1), date(2025, 2, 15), out_dir)
        check("tidak ada panggilan fetch_data baru (semua sudah ada)",
              len(call_log) == 0, f"{call_log}")

    print("\n== 5. Periode kosong (missing) tidak membuat file, tidak error ==")
    def fake_fetch_empty(ticker, start_date, end_date, market, data_type):
        return _FakeResult(data=pd.DataFrame(), missing=[start_date], failed=[])

    mod.fetch_data = fake_fetch_empty
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        try:
            fetch_symbol_tick_data("ETHUSDT", date(2025, 3, 1), date(2025, 3, 31), out_dir)
            check("periode kosong ditangani tanpa exception", True)
        except Exception as e:
            check("periode kosong ditangani tanpa exception", False, str(e))
        empty_path = out_dir / "aggTrades" / "ETHUSDT" / "2025-03.parquet"
        check("tidak ada file dibuat untuk periode kosong", not empty_path.exists())

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS (logika teruji, konektivitas data.binance.vision BELUM diuji).")
    return 0


if __name__ == "__main__":
    sys.exit(main())