#!/usr/bin/env python3
"""
Membuat seluruh struktur folder dan file kosong untuk proyek bot trading.

Jalankan sekali di awal:
    python scaffold.py

Aman dijalankan berulang: file yang sudah ada TIDAK akan ditimpa.
Yang sudah diisi tetap aman, yang belum ada akan dibuatkan kerangkanya.

Tiap file stub berisi docstring yang menjelaskan fungsinya dan sprint
keberapa file itu dikerjakan, supaya tidak perlu bolak-balik ke dokumen.
"""

from pathlib import Path

ROOT = Path(__file__).parent

TODO_TAG = "TODO"  # dipisah supaya scaffold.py sendiri tidak ikut ter-grep

# ---------------------------------------------------------------------------
# Daftar file: path -> (sprint, deskripsi singkat)
# Sprint 1 = minggu 1. Sprint 2-6 dibuat sebagai stub kosong supaya
# strukturnya terlihat sejak awal, tapi belum diisi.
# ---------------------------------------------------------------------------

FILES: dict[str, tuple[int, str]] = {
    # --- core: fondasi, tidak bergantung pada apa pun -----------------------
    "src/core/cost_model.py": (
        1,
        "Satu-satunya sumber kebenaran soal ongkos. Dipanggil backtest DAN live.",
    ),
    "src/core/instruments.py": (
        1,
        "Spesifikasi instrumen dari bursa: tick size, step size, notional minimum.",
    ),
    "src/core/types.py": (
        1,
        "Bentuk data bersama: Bar, Signal, Order, Fill, Position.",
    ),
    # --- data: menyediakan bar ----------------------------------------------
    "src/data/fetch_history.py": (1, "Unduh bar 1 menit dari bursa ke Parquet."),
    "src/data/store.py": (1, "Baca data historis, bagi data latih dan data uji."),
    "src/data/stream.py": (
        1,
        "Aliran bar langsung lewat WebSocket. Keluarannya identik dengan store.py.",
    ),
    # --- strategy: hanya berpikir, tidak menyentuh order ---------------------
    "src/strategy/base.py": (1, "Kontrak yang wajib dipenuhi tiap strategi."),
    "src/strategy/vwap_reversion.py": (
        1,
        "Strategi pertama. Maksimal tiga parameter, tanpa pengecualian.",
    ),
    # --- backtest: menguji ---------------------------------------------------
    "src/backtest/engine.py": (1, "Simulasi bar per bar, memanggil cost_model."),
    "src/backtest/metrics.py": (1, "Expectancy, t-stat, drawdown, Sharpe, win rate."),
    "src/backtest/validate.py": (
        1,
        "Pembagian out-of-sample dan uji sensitivitas parameter.",
    ),
    # --- execution: bertindak di bursa --------------------------------------
    "src/execution/broker.py": (
        1,
        "SATU-SATUNYA file yang menyentuh API bursa. Bungkus ccxt.",
    ),
    "src/execution/order_manager.py": (
        1,
        "Siklus hidup order + client order ID idempoten.",
    ),
    "src/execution/reconciler.py": (
        1,
        "Cocokkan posisi bot dengan posisi bursa tiap 30 detik.",
    ),
    "src/execution/slippage_tracker.py": (
        1,
        "Ukur selisih harga sinyal vs harga fill. Bahan koreksi cost_model.",
    ),
    "src/execution/repricer.py": (2, "Geser harga limit order yang belum terisi."),
    # --- risk: menegakkan batas ---------------------------------------------
    "src/risk/limits.py": (1, "Batas rugi harian, batas posisi, jam operasi."),
    "src/risk/sizing.py": (1, "Ukuran posisi berbasis volatilitas."),
    # --- ops: pemantauan dan kendali ----------------------------------------
    "src/ops/logger.py": (1, "Log JSON terstruktur."),
    "src/ops/notifier.py": (1, "Notifikasi Telegram."),
    "src/ops/killswitch.py": (1, "Tutup semua posisi dan hentikan bot."),
    "src/ops/watchdog.py": (3, "Pantau detak jantung bot, restart kalau mati."),
    "src/ops/anomaly.py": (3, "Deteksi perilaku tidak wajar sebelum jadi kerugian."),
    # --- runner: menjahit semuanya ------------------------------------------
    "src/runner/paper.py": (1, "Jalankan sistem lengkap di testnet."),
    "src/runner/live.py": (1, "Versi uang sungguhan. Beda tipis dari paper.py."),
    # --- analysis: peninjauan hasil -----------------------------------------
    "src/analysis/slippage_report.py": (2, "Bandingkan slippage nyata vs asumsi model."),
    "src/analysis/live_review.py": (4, "Evaluasi seluruh transaksi nyata."),
    "src/analysis/dashboard.py": (6, "Dashboard HTML: equity curve, slippage, drawdown."),
    # --- ml: Sprint 5, tidak lebih awal -------------------------------------
    "src/ml/features.py": (5, "Order book imbalance, volume profile, regime volatilitas."),
    "src/ml/purged_cv.py": (5, "Cross-validation yang mencegah kebocoran lintas waktu."),
    "src/ml/train.py": (5, "Latih LightGBM sebagai PENYARING, bukan penghasil sinyal."),
    "src/ml/filter.py": (5, "Terapkan penyaring ke sinyal baseline saat live."),
    # --- scripts: dijalankan manual -----------------------------------------
    "scripts/analyze_costs.py": (1, "Rasio pergerakan harga terhadap ongkos."),
    "scripts/run_backtest.py": (1, "Jalankan satu simulasi + cek syarat lulus."),
    "scripts/run_sweep.py": (1, "Sapu banyak kombinasi parameter."),
    "scripts/daily_report.py": (1, "Laporan harian ke Telegram."),
    # --- tests: tiga yang wajib sebelum uang sungguhan masuk ----------------
    "tests/test_cost_model.py": (1, "WAJIB. Salah di sini merusak semua penilaian."),
    "tests/test_limits.py": (1, "WAJIB. Pengaman yang tidak diuji sama dengan tidak ada."),
    "tests/test_order_manager.py": (
        1,
        "WAJIB. Simulasi koneksi putus, pastikan tidak ada order ganda.",
    ),
}

DATA_DIRS = ["data/raw", "data/live", "docs", "notebooks"]

PACKAGE_DIRS = [
    "src",
    "src/core",
    "src/data",
    "src/strategy",
    "src/backtest",
    "src/execution",
    "src/risk",
    "src/ops",
    "src/runner",
    "src/analysis",
    "src/ml",
    "scripts",
    "tests",
]

GITIGNORE = """\
.venv/
__pycache__/
*.pyc

# Data mentah bisa diunduh ulang, tidak perlu masuk git
data/raw/*.parquet
data/live/*.jsonl

# JANGAN PERNAH commit kredensial
config/secrets.env
.env
*.key
"""

LESSONS_HEADER = """\
# Catatan Pelajaran

Satu entri per insiden atau keputusan. Ditulis saat kejadian, bukan diingat
belakangan. Keputusan yang tidak dicatat akan diperdebatkan ulang dari nol
dua minggu lagi.

Format:

    ## YYYY-MM-DD - judul singkat
    Apa yang terjadi:
    Kenapa terjadi:
    Apa yang diubah:

---
"""

SECRETS_TEMPLATE = """\
# Salin file ini menjadi config/secrets.env lalu isi.
# JANGAN commit secrets.env ke git.
#
# API key WAJIB dibatasi: izin trading saja, TANPA izin penarikan,
# dengan IP whitelist ke alamat VPS.

EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
"""


def stub_content(path: str, sprint: int, desc: str) -> str:
    """Isi awal file stub: docstring + penanda pekerjaan."""
    if path.startswith("tests/"):
        module = path.replace("tests/test_", "").replace(".py", "")
        return (
            f'"""\n{desc}\n\nSprint {sprint}.\n"""\n\n'
            f"import pytest\n\n\n"
            f"def test_{module}_placeholder():\n"
            f"    # {TODO_TAG} Sprint {sprint}: tulis pengujian yang sesungguhnya\n"
            f"    pytest.skip('belum ditulis')\n"
        )

    if sprint == 1:
        note = ""
    else:
        note = f"Belum dikerjakan sampai Sprint {sprint}.\n"

    return f'"""\n{desc}\n\nSprint {sprint}.\n{note}"""\n\n# {TODO_TAG} Sprint {sprint}\n'


def main() -> None:
    created: list[str] = []
    skipped: list[str] = []

    for d in DATA_DIRS:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        keep = p / ".gitkeep"
        if not keep.exists():
            keep.touch()

    for d in PACKAGE_DIRS:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        init = p / "__init__.py"
        if not init.exists():
            init.touch()
            created.append(str(init.relative_to(ROOT)))

    (ROOT / "config").mkdir(exist_ok=True)

    for rel, (sprint, desc) in sorted(FILES.items()):
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            skipped.append(rel)
            continue
        path.write_text(stub_content(rel, sprint, desc), encoding="utf-8")
        created.append(rel)

    extras = {
        ".gitignore": GITIGNORE,
        "lessons.md": LESSONS_HEADER,
        "config/secrets.env.example": SECRETS_TEMPLATE,
    }
    for rel, content in extras.items():
        path = ROOT / rel
        if path.exists():
            skipped.append(rel)
            continue
        path.write_text(content, encoding="utf-8")
        created.append(rel)

    print(f"Dibuat  : {len(created)} file")
    print(f"Dilewati: {len(skipped)} file (sudah ada, tidak ditimpa)\n")

    if created:
        print("File baru:")
        for f in sorted(created):
            print(f"  + {f}")

    by_sprint: dict[int, int] = {}
    for _, (sprint, _) in FILES.items():
        by_sprint[sprint] = by_sprint.get(sprint, 0) + 1

    print("\nJumlah file kode per sprint:")
    for sprint in sorted(by_sprint):
        print(f"  Sprint {sprint}: {by_sprint[sprint]} file")

    print("\nLangkah berikutnya:")
    print("  1. python -m venv .venv && source .venv/bin/activate")
    print("  2. pip install -r requirements.txt")
    print("  3. cp config/secrets.env.example config/secrets.env   (isi nanti, hari 6)")
    print("  4. python -m src.core.cost_model")


if __name__ == "__main__":
    main()
