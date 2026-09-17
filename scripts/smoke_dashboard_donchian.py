"""
scripts/smoke_dashboard_donchian.py

Uji asap inti perhitungan dashboard -- terutama yang Nero minta:
"BERSIH = total trade DIKURANGI fee di Binance". Semua angka di sini
bisa dihitung tangan, dan bentuk datanya meniru respons ccxt/Binance
yang sungguhan (fee unified + info.realizedPnl/commission mentah).
"""

import sys

sys.path.insert(0, ".")
from scripts.dashboard_donchian import aggregate_trades  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def trade(pnl, fee, side="buy", price=100.0, amount=0.01, use_unified_fee=True):
    """Bentuk trade meniru ccxt: fee unified + info mentah Binance."""
    t = {
        "side": side, "price": price, "amount": amount,
        "info": {"realizedPnl": str(pnl), "commission": str(fee)},
    }
    if use_unified_fee:
        t["fee"] = {"cost": fee, "currency": "USDT"}
    return t


def main() -> int:
    print("== 1. BERSIH = total realisasi DIKURANGI fee (inti permintaan Nero) ==")
    # 3 trade: +10, -4, +6 -> realisasi total = +12
    # fee: 0.5 + 0.5 + 0.5 = 1.5
    # BERSIH = 12 - 1.5 = 10.5
    trades = [trade(10.0, 0.5), trade(-4.0, 0.5), trade(6.0, 0.5)]
    a = aggregate_trades(trades)
    check("total_realized = 12.0", abs(a["total_realized"] - 12.0) < 1e-9, f"dapat {a['total_realized']}")
    check("total_fee = 1.5", abs(a["total_fee"] - 1.5) < 1e-9, f"dapat {a['total_fee']}")
    check("net = 10.5 (12 - 1.5)", abs(a["net"] - 10.5) < 1e-9, f"dapat {a['net']}")

    print("\n== 2. Fee BISA lebih besar dari untung -> BERSIH jadi NEGATIF walau realisasi positif ==")
    # Ini persis kekhawatiran Nero: 'kotor' kelihatan untung, tapi habis kena fee.
    trades2 = [trade(0.6, 1.2), trade(0.4, 1.2)]  # realisasi +1.0, fee 2.4 -> bersih -1.4
    a2 = aggregate_trades(trades2)
    check("total_realized POSITIF (+1.0)", abs(a2["total_realized"] - 1.0) < 1e-9)
    check("net NEGATIF (-1.4) -- fee mengalahkan untung",
          abs(a2["net"] - (-1.4)) < 1e-9, f"dapat {a2['net']}")

    print("\n== 3. Trade PEMBUKA (realizedPnl=0) TIDAK dihitung menang maupun kalah ==")
    # Pola persis yang Nero tanyakan sebelumnya: baris "0.000" itu order
    # pembuka, bukan transaksi seri.
    trades3 = [trade(0.0, 0.3), trade(5.0, 0.3), trade(0.0, 0.3), trade(-2.0, 0.3)]
    a3 = aggregate_trades(trades3)
    check("n_fills = 4 (semua fill dihitung)", a3["n_fills"] == 4)
    check("n_closing_trades = 2 (cuma yang merealisasikan sesuatu)", a3["n_closing_trades"] == 2,
          f"dapat {a3['n_closing_trades']}")
    check("menang=1, kalah=1", a3["n_wins"] == 1 and a3["n_losses"] == 1)
    check("fee TETAP dihitung dari SEMUA 4 fill (0.3x4=1.2)",
          abs(a3["total_fee"] - 1.2) < 1e-9, f"dapat {a3['total_fee']}")

    print("\n== 4. Fallback ke info.commission kalau fee unified ccxt kosong ==")
    trades4 = [trade(3.0, 0.7, use_unified_fee=False)]  # cuma ada di info.commission
    a4 = aggregate_trades(trades4)
    check("fee terbaca dari info.commission (0.7)", abs(a4["total_fee"] - 0.7) < 1e-9,
          f"dapat {a4['total_fee']}")
    check("n_fee_unknown = 0 (fee ketemu, cuma di tempat lain)", a4["n_fee_unknown"] == 0)

    print("\n== 5. Fee BENAR-BENAR tidak ada -> dihitung 0 TAPI DITANDAI, bukan diam-diam ==")
    trades5 = [{"side": "buy", "price": 100, "amount": 0.01, "info": {"realizedPnl": "5.0"}}]
    a5 = aggregate_trades(trades5)
    check("net = 5.0 (fee dianggap 0)", abs(a5["net"] - 5.0) < 1e-9)
    check("n_fee_unknown = 1 -- DITANDAI supaya dashboard bisa peringatkan user",
          a5["n_fee_unknown"] == 1, f"dapat {a5['n_fee_unknown']}")

    print("\n== 6. Statistik turunan dihitung tangan ==")
    # menang: +10, +6 (rata +8, terbaik +10) | kalah: -4 (rata -4, terburuk -4)
    # profit factor = 16 / 4 = 4.0 | win rate = 2/3 = 66.7%
    a6 = aggregate_trades([trade(10.0, 0.0), trade(-4.0, 0.0), trade(6.0, 0.0)])
    check("avg_win = 8.0", abs(a6["avg_win"] - 8.0) < 1e-9, f"dapat {a6['avg_win']}")
    check("avg_loss = -4.0", abs(a6["avg_loss"] - (-4.0)) < 1e-9, f"dapat {a6['avg_loss']}")
    check("best_win = 10.0", abs(a6["best_win"] - 10.0) < 1e-9)
    check("worst_loss = -4.0", abs(a6["worst_loss"] - (-4.0)) < 1e-9)
    check("profit_factor = 4.0", abs(a6["profit_factor"] - 4.0) < 1e-9, f"dapat {a6['profit_factor']}")
    check("win_rate = 2/3", abs(a6["win_rate"] - 2/3) < 1e-9, f"dapat {a6['win_rate']}")

    print("\n== 7. Tidak ada kalah sama sekali -> profit_factor None, bukan pembagian nol ==")
    a7 = aggregate_trades([trade(5.0, 0.1), trade(3.0, 0.1)])
    check("profit_factor = None (tidak crash)", a7["profit_factor"] is None)

    print("\n== 8. Daftar kosong -> semua nol, tidak crash ==")
    a8 = aggregate_trades([])
    check("n_fills=0, net=0, win_rate=0",
          a8["n_fills"] == 0 and a8["net"] == 0.0 and a8["win_rate"] == 0.0)

    print("\n== 9. Data bentuk BINANCE SUNGGUHAN (split fill, persis pola yang Nero lihat) ==")
    # Order pembuka terpecah 2 fill (realizedPnl 0 keduanya), lalu 1 fill penutup.
    real_shape = [
        {"side": "sell", "price": 75631.20, "amount": 0.0007, "fee": {"cost": 0.01058836, "currency": "USDT"},
         "info": {"realizedPnl": "0", "commission": "0.01058836", "maker": True}},
        {"side": "sell", "price": 75631.20, "amount": 0.0013, "fee": {"cost": 0.01966411, "currency": "USDT"},
         "info": {"realizedPnl": "0", "commission": "0.01966411", "maker": True}},
        {"side": "buy", "price": 75570.60, "amount": 0.0100, "fee": {"cost": 0.30228240, "currency": "USDT"},
         "info": {"realizedPnl": "0.74599999", "commission": "0.30228240", "maker": False}},
    ]
    a9 = aggregate_trades(real_shape)
    expected_fee = 0.01058836 + 0.01966411 + 0.30228240
    expected_net = 0.74599999 - expected_fee
    check("3 fill terbaca, tapi cuma 1 transaksi penutup",
          a9["n_fills"] == 3 and a9["n_closing_trades"] == 1)
    check(f"total_fee = {expected_fee:.8f}", abs(a9["total_fee"] - expected_fee) < 1e-9,
          f"dapat {a9['total_fee']:.8f}")
    check(f"net = {expected_net:.8f} (realisasi 0.746 - fee {expected_fee:.4f})",
          abs(a9["net"] - expected_net) < 1e-9, f"dapat {a9['net']:.8f}")
    print(f"     -> Realisasi +0.746 terlihat UNTUNG, tapi setelah fee: {a9['net']:+.4f} "
          f"({'RUGI' if a9['net'] < 0 else 'untung'}) -- persis poin Nero.")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())