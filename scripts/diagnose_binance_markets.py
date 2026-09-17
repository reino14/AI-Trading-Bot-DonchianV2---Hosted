"""
scripts/diagnose_binance_markets.py

Cetak struktur MENTAH market dari ccxt.binance() (exchange 'spot') untuk
beberapa simbol yang mengandung "BTC". Tujuannya satu: melihat nilai
SUNGGUHAN dari field spot/swap/contract/type/settle di instalasi ccxt
Anda, bukan menebak lagi berdasarkan spesifikasi umum ccxt.

Jalankan: python scripts/diagnose_binance_markets.py
"""

import ccxt

ex = ccxt.binance({"enableRateLimit": True})
markets = ex.load_markets()

print(f"ccxt version : {ccxt.__version__}")
print(f"Total market di exchange 'binance' (spot): {len(markets)}")
print()

btc_symbols = [s for s in markets if "BTC" in s and "/USDT" in s]
print(f"Simbol mengandung 'BTC' dan '/USDT': {len(btc_symbols)}")
print()

fields = ["symbol", "type", "spot", "swap", "future", "contract", "linear", "settle", "quote", "active"]

for sym in btc_symbols[:15]:
    m = markets[sym]
    print(f"--- {sym} ---")
    for f in fields:
        print(f"    {f:10s}: {m.get(f, '<tidak ada field ini>')}")
    print()

# Hitung ringkas: dari SELURUH market di exchange 'binance' (bukan cuma
# yang mengandung BTC), berapa yang punya spot=True vs swap=True --
# ini akan langsung menunjukkan apakah exchange 'binance' di instalasi
# Anda benar-benar cuma berisi market spot, atau sudah tercampur.
n_spot_true = sum(1 for m in markets.values() if m.get("spot") is True)
n_swap_true = sum(1 for m in markets.values() if m.get("swap") is True)
n_contract_true = sum(1 for m in markets.values() if m.get("contract") is True)
print("=== RINGKASAN SELURUH MARKET DI exchange 'binance' (spot) ===")
print(f"  spot=True     : {n_spot_true}")
print(f"  swap=True     : {n_swap_true}")
print(f"  contract=True : {n_contract_true}")