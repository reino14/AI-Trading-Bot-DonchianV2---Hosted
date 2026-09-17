"""
scripts/smoke_smc_signal_generator.py

Uji asap microstructure/smc_signal_generator.py -- skenario end-to-end
yang menggabungkan BOS/CHoCH + zona + Order Block jadi satu setup
valid, plus satu tes per alasan penolakan.
"""

import sys

import pandas as pd

from src.microstructure.smc_liquidity import detect_liquidity_sweeps  # noqa: F401 (dicek importable)
from src.microstructure.smc_patterns import detect_order_blocks
from src.microstructure.smc_signal_generator import generate_smc_setups
from src.microstructure.smc_structure import detect_bos_choch
from src.microstructure.smc_zones import compute_premium_discount

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_df(rows: list[tuple], start="2026-01-01", freq="1h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def build_scenario():
    """
    Skenario TERVERIFIKASI langkah demi langkah di sandbox sebelum
    ditulis di sini: downtrend ke swing low (78) -> reli tembus swing
    high lama (98, CHoCH bullish, candle merah index4 SEBENARNYA bukan
    OB untuk event ini -- OB-nya candle merah index12 tepat sebelum
    breakout index13) -> reli BERLANJUT JAUH ke swing high baru (180)
    supaya equilibrium ikut naik dan Order Block (90-93) jatuh di
    PARUH BAWAH range baru (zona discount) -> pullback tajam balik ke
    area Order Block -> SETUP LONG di situ.

    Percobaan pertama saya (reli pendek, cuma sampai ~98) GAGAL karena
    Order Block malah jatuh di ATAS titik tengah range saat itu --
    "di dalam OB" dan "zona discount" jadi saling meniadakan secara
    geometris. Baru setelah range diperlebar (reli lanjut ke 180),
    keduanya bisa align.
    """
    rows = [
        (100, 101, 95, 96),
        (96, 97, 85, 87),
        (87, 90, 84, 88),
        (88, 89, 83, 84),
        (84, 86, 80, 82),
        (82, 84, 79, 81),
        (81, 83, 78, 80),    # swing low = 78
        (80, 95, 79, 93),
        (93, 96, 91, 94),
        (94, 98, 92, 96),    # swing high kandidat = 98
        (96, 97, 93, 94),
        (94, 95, 91, 92),
        (92, 93, 90, 91),    # konfirmasi swing high@9; candle MERAH -- Order Block
        (91, 105, 90, 103),  # BOS/CHoCH bullish (tembus 98)
        (103, 140, 100, 138),
        (138, 160, 135, 158),
        (158, 180, 150, 178),  # swing high baru kandidat = 180
        (178, 179, 170, 172),
        (172, 173, 168, 169),
        (169, 170, 165, 166),  # konfirmasi swing high@16
        (166, 167, 88, 91),    # pullback tajam balik ke area Order Block (90-93)
    ]
    return make_df(rows)


def main() -> int:
    df = build_scenario()
    bos_events = detect_bos_choch(df, lookback=3)
    order_blocks = detect_order_blocks(df, bos_events, lookback_candles=5)
    zone_series = compute_premium_discount(df, lookback=3)

    print("== 1. Komponen individual terbentuk sebagaimana mestinya ==")
    check("ADA peristiwa BOS/CHoCH terdeteksi", not bos_events.empty,
          f"jenis: {bos_events['event_type'].tolist() if not bos_events.empty else []}")
    check("ADA Order Block bullish terdeteksi",
          not order_blocks.empty and (order_blocks["ob_type"] == "bullish").any(),
          f"jumlah OB: {len(order_blocks)}")

    print("\n== 2. generate_smc_setups: end-to-end tidak crash, hasil berbentuk benar ==")
    setups = generate_smc_setups(df, bos_events, order_blocks, zone_series, swing_lookback=3)
    check("Return berupa DataFrame (bukan error)", isinstance(setups, pd.DataFrame))
    check("ADA setup dihasilkan (skenario dirancang supaya semua syarat align)",
          not setups.empty, f"jumlah setup: {len(setups)}")
    if not setups.empty:
        check("Kolom yang dibutuhkan lengkap",
              all(c in setups.columns for c in
                  ["direction", "entry_price", "stop_price", "target_price", "zone", "ob_low", "ob_high"]))
        check("Semua target LONG > entry, semua target SHORT < entry (arah masuk akal)",
              ((setups["direction"] != "long") | (setups["target_price"] > setups["entry_price"])).all()
              and ((setups["direction"] != "short") | (setups["target_price"] < setups["entry_price"])).all())
        check("entry berada DI DALAM rentang OB (bukan cuma wick bersinggungan)",
              ((setups["entry_price"] >= setups["ob_low"]) & (setups["entry_price"] <= setups["ob_high"])).all())

    print("\n== 3. GAGAL: bias tidak ada (belum ada BOS/CHoCH sama sekali) -> tidak ada setup ==")
    empty_events = pd.DataFrame(
        columns=["bar_pos", "event_type"], index=pd.DatetimeIndex([], tz="UTC"),
    )
    setups3 = generate_smc_setups(df, empty_events, order_blocks, zone_series, swing_lookback=3)
    check("tidak ada setup tanpa bias", setups3.empty)

    print("\n== 4. GAGAL: Order Block kosong -> tidak ada setup walau bias+zona terpenuhi ==")
    empty_obs = pd.DataFrame(
        columns=["ob_type", "ob_low", "ob_high"], index=pd.DatetimeIndex([], tz="UTC"),
    )
    setups4 = generate_smc_setups(df, bos_events, empty_obs, zone_series, swing_lookback=3)
    check("tidak ada setup tanpa Order Block", setups4.empty)

    print("\n== 5. GAGAL: paksa semua zona jadi 'premium' -> bias 'up' tidak pernah cocok lokasi ==")
    forced_premium = pd.Series(["premium"] * len(df), index=df.index)
    setups5 = generate_smc_setups(df, bos_events, order_blocks, forced_premium, swing_lookback=3)
    check("tidak ada setup LONG (bias up butuh zona discount, bukan premium)",
          setups5.empty or (setups5["direction"] != "long").all())

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())