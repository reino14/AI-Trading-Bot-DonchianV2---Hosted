"""
scripts/smoke_volume_profile.py

Uji asap microstructure/volume_profile.py. Beda dari uji asap lain di
proyek ini: di sini kita BISA hitung jawaban yang benar dengan tangan
(atau kalkulator), karena datanya sengaja dibuat sederhana -- jadi
tes ini memverifikasi KEBENARAN ALGORITMA, bukan cuma "tidak crash".
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.microstructure.volume_profile import (
    build_profile_history,
    compute_session_profile,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1. POC sederhana: satu harga jelas dominan ==")
    # Rentang harga 100-110, 10 bin -> tiap bin lebar 1.0.
    # Taruh volume BESAR di harga 105 (bin ke-5), sisanya kecil merata.
    prices = [100, 101, 102, 103, 104, 105, 105, 105, 105, 106, 107, 108, 109]
    qty =    [1,   1,   1,   1,   1,   5,   5,   5,   5,   1,   1,   1,   1]
    is_sell = [False]*13
    trades = pd.DataFrame({"price": prices, "quantity": qty, "is_buyer_maker": is_sell})
    profile = compute_session_profile(trades, "test1", n_bins=10, value_area_pct=0.70)
    check("POC ada di sekitar harga 105 (bin volume terbesar)",
          104.5 <= profile.poc_price <= 105.5, f"POC={profile.poc_price}")
    check("total volume benar", profile.total_volume == sum(qty),
          f"dapat={profile.total_volume}, harus={sum(qty)}")

    print("\n== 2. Value Area mencakup >= 70% volume, dan MINIMAL (tidak kelebihan bin) ==")
    # 7 bin, volume per bin: [5, 15, 30, 100, 25, 12, 3] -- total 190,
    # target 70% = 133. Sengaja TANPA dasi supaya ekspansi dua sisi
    # bisa diverifikasi tangan tanpa ambiguitas tie-breaking.
    #
    # Dua trade "sentinel" kuantitas nyaris nol dipasang PERSIS di
    # harga 100.0 dan 107.0 supaya low/high (dan karenanya bin_width)
    # yang dihitung fungsi PERSIS 1.0 -- tanpa ini, low/high diambil
    # dari titik tengah bin ekstrem, bukan batas bin sungguhan, dan
    # bin_width jadi tidak sesuai perhitungan tangan (ini yang bikin
    # versi tes sebelumnya salah).
    bin_volumes = [5, 15, 30, 100, 25, 12, 3]
    n_bins = len(bin_volumes)
    prices = [100.0, 107.0] + [100.5 + i for i in range(n_bins)]
    qtys = [1e-9, 1e-9] + [float(v) for v in bin_volumes]
    trades2 = pd.DataFrame({
        "price": prices, "quantity": qtys, "is_buyer_maker": [False] * len(prices),
    })
    profile2 = compute_session_profile(trades2, "test2", n_bins=n_bins, value_area_pct=0.70)
    # Tangan: POC=bin3(103.5). Langkah1: bin2(30)>=bin4(25) -> lo=bin2(102.5), va=130.
    # Langkah2: bin1(15)<bin4(25) -> hi=bin4(104.5), va=155>=133 -> berhenti.
    check("POC = bin index 3 (harga 103.5)", abs(profile2.poc_price - 103.5) < 1e-6,
          f"POC={profile2.poc_price}")
    check("VAL = bin index 2 (harga 102.5)", abs(profile2.val_price - 102.5) < 1e-6,
          f"VAL={profile2.val_price}")
    check("VAH = bin index 4 (harga 104.5)", abs(profile2.vah_price - 104.5) < 1e-6,
          f"VAH={profile2.vah_price}")
    check("bin index 1 dan 5 (di luar hasil ekspansi) TIDAK ikut masuk",
          profile2.val_price > 101.5 and profile2.vah_price < 105.5)

    print("\n== 3. Delta buy/sell dari is_buyer_maker dihitung benar ==")
    # is_buyer_maker=True -> agresor PENJUAL -> sell_volume.
    # is_buyer_maker=False -> agresor PEMBELI -> buy_volume.
    trades3 = pd.DataFrame({
        "price": [100.0, 100.0, 100.0],
        "quantity": [3.0, 7.0, 2.0],
        "is_buyer_maker": [True, False, True],  # sell, buy, sell
    })
    profile3 = compute_session_profile(trades3, "test3", n_bins=1, value_area_pct=0.70)
    check("sell_volume = 3+2 = 5", profile3.total_sell_volume == 5.0,
          f"dapat={profile3.total_sell_volume}")
    check("buy_volume = 7", profile3.total_buy_volume == 7.0,
          f"dapat={profile3.total_buy_volume}")

    print("\n== 4. Sesi kosong -> None, bukan error ==")
    empty = pd.DataFrame({"price": [], "quantity": [], "is_buyer_maker": []})
    result = compute_session_profile(empty, "empty", n_bins=10)
    check("mengembalikan None untuk sesi kosong", result is None)

    print("\n== 5. Harga tunggal (rentang nol) tidak crash ==")
    single = pd.DataFrame({"price": [100.0, 100.0], "quantity": [1.0, 2.0],
                            "is_buyer_maker": [False, True]})
    result_single = compute_session_profile(single, "single", n_bins=10)
    check("tidak crash, POC = harga itu sendiri",
          result_single is not None and result_single.poc_price == 100.0)

    print("\n== 6. build_profile_history: pemisahan sesi UTC harian dari file bulanan ==")
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        sym_dir = data_dir / "aggTrades" / "BTCUSDT"
        sym_dir.mkdir(parents=True)

        rng = np.random.default_rng(7)
        n = 5000
        # 3 hari kalender UTC dalam satu file bulanan sintetis.
        times = pd.date_range("2026-07-01", periods=n, freq="15min", tz="UTC")[: n]
        # Batasi ke 3 hari saja dengan mengulang start
        times = pd.date_range("2026-07-01", periods=3, freq="1D", tz="UTC")
        full_times = pd.to_datetime(
            np.random.default_rng(3).choice(
                pd.date_range("2026-07-01", "2026-07-03 23:59", freq="1min", tz="UTC"), n
            )
        )
        df = pd.DataFrame({
            "price": 100 + rng.normal(0, 1, n).cumsum() * 0.01,
            "quantity": rng.uniform(0.1, 2.0, n),
            "is_buyer_maker": rng.choice([True, False], n),
            "transact_time": full_times,
        })
        df.to_parquet(sym_dir / "2026-07.parquet")

        history = build_profile_history(data_dir, "BTCUSDT", n_bins=20)
        check("3 sesi harian terdeteksi", len(history) == 3, f"dapat {len(history)} sesi: {history['session_id'].tolist() if not history.empty else []}")
        check("kolom penting ada", all(
            c in history.columns for c in ["poc_price", "vah_price", "val_price", "total_volume"]
        ))

    print("\n== 7. File tidak ada -> error jelas, bukan silent empty ==")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            build_profile_history(Path(tmp), "NOSUCHSYMBOL", n_bins=10)
            check("melempar error kalau file tidak ada", False, "tidak melempar error")
        except FileNotFoundError:
            check("melempar error kalau file tidak ada", True)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())