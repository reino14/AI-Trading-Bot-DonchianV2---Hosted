"""
scripts/session_pnl_report.py

Laporan untung/rugi REALISASI berdasarkan data/fills.csv -- catatan
transaksi yang DICATAT SENDIRI oleh runner/paper.py tiap kali order
ke-fill (lihat execution/slippage_tracker.py).

PENTING: ini BEDA dari saldo total dompet di dashboard Binance --
laporan ini CUMA menghitung transaksi yang dilakukan BOT ini sendiri,
tidak kecampur transaksi manual atau sisa dari sesi-sesi lama yang
tidak berhubungan. Kalau ada transaksi manual (mis. jual manual sisa
ETH nyangkut), itu TIDAK akan muncul di sini -- justru itu tujuannya,
supaya laporan ini murni performa strategi, bukan tercampur aksi
manual.

Cara pakai:
    python -m scripts.session_pnl_report
    python -m scripts.session_pnl_report --symbol ETH/USDT
    python -m scripts.session_pnl_report --since "2026-09-13 13:00"
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

FILLS_PATH = Path("data/fills.csv")


def load_fills(symbol: str | None = None, since: datetime | None = None) -> list[dict]:
    if not FILLS_PATH.exists():
        raise FileNotFoundError(
            f"Belum ada {FILLS_PATH} -- belum ada fill yang tercatat. "
            f"Jalankan dulu runner/paper.py sampai minimal satu order ke-fill."
        )

    rows = []
    with open(FILLS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if symbol is not None and row["symbol"] != symbol:
                continue
            fill_time = datetime.fromisoformat(row["fill_time"])
            if fill_time.tzinfo is None:
                fill_time = fill_time.replace(tzinfo=timezone.utc)
            if since is not None and fill_time < since:
                continue
            row["_fill_time"] = fill_time
            row["fill_price"] = float(row["fill_price"])
            row["amount_filled"] = float(row["amount_filled"])
            row["fee"] = float(row["fee"])
            rows.append(row)

    return sorted(rows, key=lambda r: r["_fill_time"])


def pair_round_trips(fills: list[dict]) -> tuple[list[dict], dict | None]:
    """
    Pasangkan fill BELI dengan fill JUAL berikutnya jadi satu "round trip".

    Asumsi: strategi LONG-ONLY (spot) -- urutan fill wajar bergantian
    beli-jual-beli-jual. Kalau ada beli yang belum ketemu jual
    pasangannya, itu POSISI YANG MASIH TERBUKA (dikembalikan terpisah,
    bukan dianggap round trip).
    """
    round_trips = []
    open_buy = None

    for fill in fills:
        if fill["side"] == "buy":
            if open_buy is not None:
                # Dua beli beruntun tanpa jual di antaranya -- seharusnya
                # tidak terjadi untuk strategi long-only, tapi jangan
                # sampai bikin skrip ini crash kalau ternyata ada.
                round_trips.append(
                    {"buy": open_buy, "sell": None, "note": "beli ganda tanpa jual di antaranya"}
                )
            open_buy = fill
        elif fill["side"] == "sell":
            if open_buy is not None:
                round_trips.append({"buy": open_buy, "sell": fill, "note": None})
                open_buy = None
            else:
                round_trips.append({"buy": None, "sell": fill, "note": "jual tanpa beli sebelumnya tercatat"})

    return round_trips, open_buy


def compute_pnl(round_trips: list[dict]) -> list[dict]:
    results = []
    for rt in round_trips:
        if rt["buy"] is None or rt["sell"] is None:
            results.append({**rt, "pnl": None})
            continue

        buy, sell = rt["buy"], rt["sell"]
        gross = (sell["fill_price"] - buy["fill_price"]) * buy["amount_filled"]
        net = gross - buy["fee"] - sell["fee"]
        results.append({**rt, "pnl": net, "gross": gross})

    return results


def print_report(results: list[dict], open_buy: dict | None, symbol_label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  LAPORAN UNTUNG/RUGI REALISASI -- {symbol_label}")
    print(f"  (sumber: data/fills.csv, HANYA transaksi yang dilakukan bot ini)")
    print("=" * 70)

    completed = [r for r in results if r["pnl"] is not None]

    if not completed and open_buy is None:
        print("\n  Belum ada transaksi tercatat.")
        return

    total_pnl = sum(r["pnl"] for r in completed)
    wins = [r for r in completed if r["pnl"] > 0]

    print(f"\n  Round trip selesai   : {len(completed)}")
    if completed:
        print(f"  Win rate             : {len(wins)}/{len(completed)} ({len(wins)/len(completed)*100:.1f}%)")
        print(f"  Total P&L realisasi  : {total_pnl:+.4f} (satuan quote, mis. USDT)")
        print(f"  Rata-rata per trip   : {total_pnl/len(completed):+.4f}")
        print()
        for i, r in enumerate(completed, 1):
            b, s = r["buy"], r["sell"]
            print(f"    #{i}  beli {b['fill_price']:.2f} -> jual {s['fill_price']:.2f}"
                  f"  |  P&L: {r['pnl']:+.4f}")

    if open_buy is not None:
        print(f"\n  POSISI MASIH TERBUKA:")
        print(f"    Dibeli {open_buy['fill_price']:.2f} pada {open_buy['_fill_time']:%Y-%m-%d %H:%M UTC}"
              f", amount {open_buy['amount_filled']}")
        print(f"    (belum ada penjualan pasangannya tercatat -- P&L belum bisa dihitung final,")
        print(f"     tergantung harga jual nanti)")

    incomplete = [r for r in results if r["pnl"] is None]
    if incomplete:
        print(f"\n  PERHATIAN: {len(incomplete)} entri tidak bisa dipasangkan sempurna:")
        for r in incomplete:
            print(f"    - {r['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Laporan P&L realisasi dari data/fills.csv")
    parser.add_argument("--symbol", default=None, help="filter simbol, mis. ETH/USDT (default: semua)")
    parser.add_argument("--since", default=None, help="cuma fill sejak waktu ini, mis. '2026-09-13 13:00'")
    args = parser.parse_args()

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    try:
        fills = load_fills(symbol=args.symbol, since=since)
    except FileNotFoundError as e:
        print(f"\n{e}\n")
        return

    round_trips, open_buy = pair_round_trips(fills)
    results = compute_pnl(round_trips)
    print_report(results, open_buy, args.symbol or "SEMUA SIMBOL")


if __name__ == "__main__":
    main()