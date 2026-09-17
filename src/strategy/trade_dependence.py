"""
strategy/trade_dependence.py

Tiga hal dari video kedua: ekstraksi trade individual dari sinyal
always-in-market, runs test (Wald-Wolfowitz) untuk uji trade
dependence, dan filter "cuma masuk kalau trade sebelumnya kalah"
(aturan Turtle Traders).

CATATAN: fungsi ekstraksi trade dan runs test di sini SAYA TULIS
SENDIRI dari definisi statistik standar (Wald-Wolfowitz runs test
sudah ada sejak 1940-an, bukan milik video manapun) -- bukan
menyalin pseudocode dari video, supaya jelas ini implementasi
independen yang saya validasi sendiri lewat kasus hitungan tangan.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def extract_trades(
    signal: pd.Series,
    close: pd.Series,
    cost_bps_per_unit: float = 0.0,
) -> pd.DataFrame:
    """
    signal: hasil compute_donchian_signal() -- boleh ada NaN di awal
        (sebelum breakout pertama), diabaikan otomatis.
    close: WAJIB index sama dengan signal.
    cost_bps_per_unit: SAMA PERSIS konvensi compute_strategy_returns()
        -- ongkos per satu unit perubahan sinyal. Trade WAJIB dikenai
        ongkos MASUK-nya sendiri (biasanya 2 unit untuk reversal
        long<->short, 1 unit untuk trade pertama dari flat) -- kalau
        ini tidak diisi (default 0), return_pct yang dilaporkan itu
        KOTOR, BUKAN hasil sungguhan. Jangan pakai default 0 untuk
        kesimpulan apa pun soal kelayakan strategi.

    Return DataFrame urut waktu: entry_time, exit_time, direction,
        entry_price, exit_price, gross_return_pct, cost_pct,
        return_pct (= net, gross - cost), hold_bars, units_changed_at_entry.

    Posisi yang MASIH TERBUKA di akhir data (belum ada breakout
    berlawanan lagi) SENGAJA TIDAK dihitung sebagai trade selesai --
    sama prinsipnya dengan engine.py lama: lebih jujur dibuang
    daripada ditebak seolah-olah sudah closed di harga terakhir.
    """
    valid = signal.dropna()
    empty_cols = [
        "entry_time", "exit_time", "direction", "entry_price", "exit_price",
        "gross_return_pct", "cost_pct", "return_pct", "hold_bars", "units_changed_at_entry",
    ]
    if valid.empty:
        return pd.DataFrame(columns=empty_cols)

    trades = []
    current_dir = None
    entry_time = None
    entry_price = None
    entry_pos = None
    entry_units_changed = None
    prev_dir = None

    for pos, (t, val) in enumerate(valid.items()):
        if current_dir is None:
            current_dir = val
            entry_time = t
            entry_price = float(close.loc[t])
            entry_pos = pos
            entry_units_changed = 1.0  # dari flat -- masuk pertama kali
            prev_dir = val
            continue

        if val != current_dir:
            exit_time = t
            exit_price = float(close.loc[t])
            gross_return_pct = (exit_price / entry_price - 1.0) * current_dir
            cost_pct = entry_units_changed * (cost_bps_per_unit / 10_000.0)
            trades.append({
                "entry_time": entry_time, "exit_time": exit_time,
                "direction": int(current_dir), "entry_price": entry_price,
                "exit_price": exit_price, "gross_return_pct": gross_return_pct,
                "cost_pct": cost_pct, "return_pct": gross_return_pct - cost_pct,
                "hold_bars": pos - entry_pos, "units_changed_at_entry": entry_units_changed,
            })
            entry_units_changed = abs(val - current_dir)  # 2.0 untuk reversal penuh
            current_dir = val
            entry_time = t
            entry_price = exit_price
            entry_pos = pos

    if not trades:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(trades)


def runs_test(signs: np.ndarray) -> float:
    """
    Wald-Wolfowitz runs test. signs: array berisi +1/-1 SAJA (bukan 0
    -- trade dengan return TEPAT nol adalah kasus tepi yang tidak
    didefinisikan uji ini, caller WAJIB membuang atau menangani
    sendiri sebelum memanggil fungsi ini).

    Return z-score. Positif = LEBIH BANYAK runs dari yang diharapkan
    acak murni -- pemenang cenderung diikuti pecundang dan sebaliknya
    (persis yang dicari video: dasar untuk aturan "cuma masuk setelah
    kalah"). NaN kalau semua sama tanda (tidak ada variasi untuk diuji).
    """
    n1 = int(np.sum(signs > 0))
    n2 = int(np.sum(signs < 0))
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 2:
        return float("nan")

    expected_runs = (2.0 * n1 * n2) / n + 1.0
    var_runs = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n ** 2 * (n - 1))
    if var_runs <= 0:
        return float("nan")
    std_runs = math.sqrt(var_runs)

    observed_runs = 1 + int(np.sum(signs[1:] != signs[:-1]))

    return (observed_runs - expected_runs) / std_runs


def filter_by_previous_trade(trades: pd.DataFrame, only_after: str) -> pd.Series:
    """
    only_after: "loser" (aturan Turtle -- cuma masuk kalau trade
        SEBELUMNYA rugi) atau "winner" (kebalikannya, dicek video
        sebagai pembanding -- terbukti hasilnya buruk).

    Return Series boolean sepanjang trades.index -- True = trade ini
    DIPERTAHANKAN. Trade PERTAMA selalu False (tidak ada "trade
    sebelumnya" untuk dibandingkan) -- BUKAN diam-diam diloloskan.
    """
    if only_after not in ("loser", "winner"):
        raise ValueError(f"only_after harus 'loser' atau 'winner', dapat: {only_after!r}")

    prev_return = trades["return_pct"].shift(1)
    if only_after == "loser":
        mask = prev_return < 0
    else:
        mask = prev_return > 0
    return mask.fillna(False)


def summarize_trades(trades: pd.DataFrame) -> dict:
    """Ringkasan dasar: jumlah, win rate, profit factor, avg win/loss."""
    if trades.empty:
        return {"n_trades": 0}

    wins = trades.loc[trades["return_pct"] > 0, "return_pct"]
    losses = trades.loc[trades["return_pct"] < 0, "return_pct"]

    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if not losses.empty and losses.sum() != 0
        else float("inf") if not wins.empty else float("nan")
    )

    return {
        "n_trades": len(trades),
        "win_rate": float((trades["return_pct"] > 0).mean()),
        "profit_factor": profit_factor,
        "avg_win_pct": float(wins.mean()) if not wins.empty else float("nan"),
        "avg_loss_pct": float(losses.mean()) if not losses.empty else float("nan"),
        "total_return_pct_sum": float(trades["return_pct"].sum()),
    }