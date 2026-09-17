"""
scripts/smoke_broker_fetch_position.py

Uji asap MockBroker.fetch_position() -- REPRODUKSI PERSIS skenario bug
Anda: short dibuka, order penutup (reduceOnly) tidak langsung fill,
posisi SEHARUSNYA masih short sampai order itu benar-benar terisi,
BARU jadi flat setelah fill sungguhan terjadi -- bukan ditebak flat
lebih awal.
"""

import sys

from src.execution.broker import MockBroker, OrderStatus

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    symbol = "BTC/USDT:USDT"

    print("== 1. Flat di awal -> fetch_position() None ==")
    broker = MockBroker(fill_immediately=False)
    check("posisi awal None (flat)", broker.fetch_position(symbol) is None)

    print("\n== 2. Buka SHORT (sell), fill_immediately=False -> order OPEN, posisi TETAP flat sampai fill ==")
    o1 = broker.place_limit_order(symbol, "sell", 0.001, 76000.0, client_order_id="open-short")
    check("order status OPEN (belum fill)", o1.status == OrderStatus.OPEN)
    check("posisi MASIH flat (order belum fill, jangan ditebak short duluan)",
          broker.fetch_position(symbol) is None)

    print("\n== 3. Order open-short AKHIRNYA fill (simulasi delayed fill) -> posisi jadi SHORT ==")
    broker.simulate_delayed_fill("open-short")
    pos = broker.fetch_position(symbol)
    check("posisi sekarang short, 0.001 BTC", pos is not None and pos["side"] == "short" and abs(pos["contracts"] - 0.001) < 1e-9,
          f"dapat {pos}")

    print("\n== 4. Kirim order PENUTUP (buy reduceOnly), TIDAK langsung fill -- PERSIS skenario bug Anda ==")
    o2 = broker.place_limit_order(symbol, "buy", 0.001, 75990.0, client_order_id="close-short", reduce_only=True)
    check("order penutup status OPEN (belum fill -- ini yang menipu logika lama)",
          o2.status == OrderStatus.OPEN)
    check("posisi MASIH short (order penutup belum benar-benar fill) -- INI KUNCI PERBAIKANNYA",
          broker.fetch_position(symbol) is not None and broker.fetch_position(symbol)["side"] == "short")
    print("     -> Kalau program mengecek fetch_position() di sini SEBELUM kirim order baru,")
    print("        dia akan lihat 'masih short, order penutup belum fill' -- BUKAN langsung")
    print("        anggap flat dan coba kirim reduceOnly lagi (itu penyebab error -2022 Anda).")

    print("\n== 5. Order penutup AKHIRNYA fill -> posisi baru benar-benar flat ==")
    broker.simulate_delayed_fill("close-short")
    check("posisi sekarang flat (None)", broker.fetch_position(symbol) is None)

    print("\n== 6. Reversal: short -> flat -> long, posisi dilacak benar di tiap tahap ==")
    broker2 = MockBroker(fill_immediately=True)  # fill langsung, uji arah/besaran saja
    broker2.place_limit_order(symbol, "sell", 0.002, 76000.0, client_order_id="s1")
    pos_short = broker2.fetch_position(symbol)
    check("setelah sell 0.002 -> short 0.002", pos_short == {"side": "short", "contracts": 0.002}, f"dapat {pos_short}")

    broker2.place_limit_order(symbol, "buy", 0.002, 76000.0, client_order_id="b1", reduce_only=True)
    check("setelah buy 0.002 (menutup) -> flat", broker2.fetch_position(symbol) is None)

    broker2.place_limit_order(symbol, "buy", 0.001, 76000.0, client_order_id="b2")
    pos_long = broker2.fetch_position(symbol)
    check("setelah buy 0.001 lagi (buka baru) -> long 0.001",
          pos_long == {"side": "long", "contracts": 0.001}, f"dapat {pos_long}")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())