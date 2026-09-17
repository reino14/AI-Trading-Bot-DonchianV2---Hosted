"""
backtest/cost_adapter.py

Jembatan RESMI antara src/core/cost_model.py (akuntansi per TRANSAKSI
bolak-balik, dipakai engine.py lama & paper.py live) dan
backtest/portfolio_engine.py (akuntansi per TURNOVER, dipakai backtest
cross-sectional).

Ini menggantikan cost_spec_from_config() yang kemarin saya tulis sebagai
tebakan sementara di portfolio_engine.py -- JANGAN pakai fungsi lama itu
lagi, pakai build_cost_spec() di sini. Fungsi lama saya biarkan ada
supaya tidak merusak import, tapi anggap sudah tidak berlaku.

KENAPA PERLU ADAPTER, BUKAN SEKADAR DIBAGI DUA
-----------------------------------------------
round_trip_cost() menjumlahkan ongkos entry + exit jadi SATU angka
bolak-balik. portfolio_engine.py butuh ongkos SATU SISI, karena ia
mengenakan ongkos proporsional terhadap turnover (|Δw|), dan tiap unit
turnover mewakili satu sisi transaksi (beli SAJA, atau jual SAJA), bukan
pasangan lengkap.

Membagi dua HANYA SAH kalau ongkos entry dan exit memang sama besar.
Untuk skenario taker/taker atau maker/maker (entry_is_maker ==
exit_is_maker), itu benar -- lihat cost_model.py: maker_bps dan
taker_bps dipakai simetris untuk kedua sisi, begitu juga spread dan
slippage. Tapi untuk skenario CAMPURAN (mis. entry maker, exit taker),
ongkos dua sisi TIDAK SAMA, dan membagi dua akan salah secara diam-diam.
Fungsi di bawah MEMVERIFIKASI kesamaan itu secara eksplisit dan menolak
kalau skenarionya asimetris, alih-alih menebak.

FUNDING: perbedaan satuan waktu
--------------------------------
cost_model.py menyatakan funding per 8 jam (funding_bps_per_8h).
portfolio_engine.py butuh per HARI (financing_bps_per_day), karena
engine mengalikannya dengan eksposur bruto lalu membaginya rata ke
seluruh bar dalam satu hari (lihat bars_per_day di
run_portfolio_backtest). Konversinya: ada 3 periode 8-jam dalam sehari,
jadi per_day = per_8h x 3.

Untuk CRYPTO_SPOT, funding_bps_per_8h = 0.0 -- dikonfirmasi langsung
dari cost_model.py Anda (spot tidak punya mekanisme funding sama
sekali), jadi financing_bps_per_day otomatis 0 juga. Ini BUKAN
asumsi saya, ini nilai yang sudah eksplisit dinolkan di preset Anda.
"""

from src.backtest.portfolio_engine import CostSpec
from src.core.cost_model import CostConfig, round_trip_cost


def build_cost_spec(
    cfg: CostConfig,
    entry_is_maker: bool = False,
    exit_is_maker: bool = False,
) -> CostSpec:
    """
    Bangun CostSpec (per-sisi, untuk portfolio_engine.py) dari CostConfig
    (bolak-balik, dari cost_model.py Anda).

    entry_is_maker / exit_is_maker WAJIB SAMA -- lihat penjelasan di atas
    soal kenapa skenario campuran tidak bisa dibagi dua begitu saja.
    Kalau Anda benar-benar butuh skenario campuran (mis. masuk pakai
    limit order sabar, keluar pakai market order darurat), fungsi ini
    akan menolak dengan ValueError yang menjelaskan kenapa -- itu sengaja,
    supaya Anda menanganinya secara eksplisit, bukan lewat pembagian yang
    diam-diam salah.
    """
    if entry_is_maker != exit_is_maker:
        raise ValueError(
            "entry_is_maker != exit_is_maker: ongkos dua sisi tidak simetris, "
            "jadi round_trip_cost() tidak bisa dibagi dua begitu saja untuk "
            "mendapat ongkos satu sisi. Ini bukan batasan portfolio_engine.py, "
            "ini batasan matematis dari cara round_trip_cost() menjumlahkan "
            "kedua sisi jadi satu angka. Kalau skenario campuran memang "
            "yang Anda mau, panggil round_trip_cost() dua kali secara "
            "terpisah (sekali dengan entry_is_maker=True/exit_is_maker=False "
            "dan pahami sendiri sisi mana menyumbang berapa) -- jangan lewat "
            "fungsi ini."
        )

    # hold_minutes=0 supaya komponen funding di round_trip_cost() nol --
    # kita HANYA mau bagian fee + spread + slippage di sini. Funding
    # ditangani terpisah lewat financing_bps_per_day di bawah, karena
    # engine portofolio menghitungnya per HARI dari eksposur yang
    # sedang dipegang, bukan per transaksi seperti di round_trip_cost().
    rt = round_trip_cost(
        cfg,
        entry_is_maker=entry_is_maker,
        exit_is_maker=exit_is_maker,
        hold_minutes=0.0,
    )

    # Verifikasi simetri secara eksplisit, bukan cuma dipercaya dari
    # entry_is_maker == exit_is_maker. Kalau suatu saat cost_model.py
    # diubah supaya entry_fee_bps dan exit_fee_bps bisa beda walau
    # skenarionya sama (mis. fee tier berbeda per sisi), ini akan
    # ketahuan di sini alih-alih lolos diam-diam.
    if abs(rt.entry_fee_bps - rt.exit_fee_bps) > 1e-9:
        raise ValueError(
            f"entry_fee_bps ({rt.entry_fee_bps}) != exit_fee_bps "
            f"({rt.exit_fee_bps}) walau skenarionya sama -- cost_model.py "
            f"kemungkinan sudah diubah jadi asimetris. Adapter ini perlu "
            f"ditulis ulang, jangan dipaksa dipakai."
        )

    per_side_bps = rt.total_bps / 2.0

    financing_bps_per_day = cfg.funding_bps_per_8h * 3.0

    return CostSpec(
        per_side_bps=per_side_bps,
        financing_bps_per_day=financing_bps_per_day,
    )


if __name__ == "__main__":
    # Uji asap: bandingkan preset CRYPTO_SPOT dan CRYPTO_PERP taker/taker,
    # cetak angka yang benar-benar akan dipakai portfolio_engine.py.
    from src.core.cost_model import CRYPTO_PERP, CRYPTO_SPOT

    for label, cfg in [("crypto_spot", CRYPTO_SPOT), ("crypto_perp", CRYPTO_PERP)]:
        spec = build_cost_spec(cfg, entry_is_maker=False, exit_is_maker=False)
        print(f"{label:12s} taker/taker -> per_side={spec.per_side_bps:.2f}bps, "
              f"financing={spec.financing_bps_per_day:.3f}bps/hari")

    print()
    print("Uji penolakan skenario campuran (harus melempar ValueError):")
    try:
        build_cost_spec(CRYPTO_SPOT, entry_is_maker=True, exit_is_maker=False)
        print("  GAGAL -- seharusnya menolak, tapi tidak")
    except ValueError as e:
        print(f"  LULUS -- ditolak dengan benar: {e}")