"""
microstructure/signal_generator.py

LANGKAH 7: rangkai tiga lapis jadi setup entry/exit yang eksplisit.
INI VERSI MEKANIS SAYA dari empat langkah Chris -- interpretasi,
bukan aturan pasti yang diverifikasi langsung dari mulutnya.

ATURAN (persis seperti yang disepakati dengan Nero)
------------------------------------------------------
1. ENVIRONMENT (bias arah): struktur 1H DAN 4H WAJIB SEPAKAT.
   Keduanya "up" -> bias LONG. Keduanya "down" -> bias SHORT.
   Selain itu (termasuk salah satu "sideways") -> TIDAK ADA bias,
   tidak ada setup dihasilkan sama sekali di bar itu.

2. LOCATION: dibandingkan terhadap Value Area HARI KALENDER UTC
   SEBELUMNYA (bukan hari yang sedang berjalan -- value area hari
   ini belum final selama harinya belum berakhir, memakainya akan
   jadi look-ahead). Bias LONG perlu close bar <= VAL hari sebelumnya
   (harga di "discount"). Bias SHORT perlu close bar >= VAH hari
   sebelumnya ("premium").

3. CONFIRMATION: bar itu WAJIB tertandai absorption (langkah 6) DAN
   warna candle-nya searah pembalikan yang diharapkan -- hijau
   (close > open) untuk LONG, merah (close < open) untuk SHORT.

Kalau ketiganya terpenuhi -> satu setup dihasilkan:
    entry = close bar konfirmasi
    stop  = low bar itu (LONG) / high bar itu (SHORT)
    target = target_mode="prev_poc" (default): POC hari kalender
        sebelumnya. target_mode="swing": swing high/low TERKONFIRMASI
        terakhir sebelum bar konfirmasi (LONG -> swing high terakhir,
        SHORT -> swing low terakhir) -- lihat generate_setups() untuk
        detail "terkonfirmasi" (tidak boleh mengintip swing yang baru
        pasti setelah bar konfirmasi ini).

CATATAN: dua target_mode ini adalah DUA HIPOTESIS TERPISAH yang harus
dihitung sebagai trial berbeda kalau Anda melacak n_trials untuk gate
statistik -- bukan "menyetel ulang sampai satu di antaranya lolos".

POLA "GAGAL DUA KALI" (require_two_attempts=True) -- TRIAL TERPISAH LAGI
--------------------------------------------------------------------------
Chris berulang kali menekankan ini di transkrip: dia TIDAK masuk begitu
absorption+flip pertama terlihat -- dia tunggu percobaan itu GAGAL
(harga balik lagi menguji level yang sama atau lebih baik), lalu masuk
di KEGAGALAN KEDUA. Interpretasi mekanis saya:

1. Bar pertama yang penuhi bias+lokasi+absorption+flip -> dicatat
   sebagai "percobaan pertama" (belum jadi setup).
2. Kalau ADA percobaan pertama yang masih "berlaku" (dalam
   `two_attempt_max_gap_bars`) dan bar baru penuhi syarat yang sama
   DENGAN LOW LEBIH TINGGI (LONG) / HIGH LEBIH RENDAH (SHORT) dari
   percobaan pertama -- itu "percobaan kedua", INI YANG JADI SETUP.
3. Kalau bar baru justru menembus LEBIH JAUH dari percobaan pertama
   (low lebih rendah untuk LONG, high lebih tinggi untuk SHORT) --
   penjual/pembeli MENANG, bukan gagal lagi -- bar ini jadi percobaan
   pertama YANG BARU (reset), bukan percobaan kedua.
4. Percobaan pertama yang tidak diikuti percobaan kedua dalam
   `two_attempt_max_gap_bars` -- KEDALUARSA, tidak pernah jadi setup.

INI TEBAKAN SAYA soal "berapa lama percobaan pertama tetap berlaku" dan
"apa persisnya definisi gagal versus menang" -- Chris tidak memberi
angka. Kalau require_two_attempts=True gagal gate, itu bisa berarti
angka `two_attempt_max_gap_bars` saya salah, bukan berarti pola dua-
percobaan Chris tidak nyata.

FILTER PARTISIPASI (min_avg_volume) -- TRIAL TERPISAH LAGI
--------------------------------------------------------------
Chris eksplisit: dia TIDAK mencari setup sama sekali kalau volume
sedang "mati" (dia sebut ambang 20.000 kontrak/5menit di MNQ, biasanya
tanda mendekati jam sepi/makan siang). INI BUKAN sama dengan syarat
volume tinggi di definisi absorption -- absorption menilai SATU bar
konfirmasi, filter ini menilai REZIM di sekitarnya (rata-rata volume
`avg_volume_window` bar terakhir, TERMASUK bar konfirmasi itu sendiri
-- tidak melihat ke depan). Kalau rezimnya sedang sepi, setup yang
technically absorption pun tetap dibuang.

Ambang (`min_avg_volume`) HARUS dikalibrasi dari data sendiri (lihat
compute_participation_threshold()) -- persis disiplin yang sama
seperti calibrate_absorption_thresholds() di absorption.py: dihitung
terpisah dari deteksi, supaya kalau nanti dipakai train/test split,
kalibrasi cuma boleh dari periode TRAIN.

ZONA FIBONACCI (require_fib_zone=True) -- TRIAL TERPISAH LAGI
------------------------------------------------------------------
Chris eksplisit: "discount" BUKAN cuma "di bawah Value Area" -- dia
menggambar retracement Fibonacci dari swing LOW ke swing HIGH
(untuk bias naik), dan level 70.5%-88.6% ("golden pocket" diperluas)
itulah zona discount sesungguhnya. Dia bahkan bilang kalau level Fib
jatuh DI DALAM value area, dia tidak mau memakainya -- artinya dua
syarat ini (value area dan Fib) memang dimaksud SALING MELENGKAPI,
bukan salah satu saja.

Implementasi: cari swing LOW terkonfirmasi terakhir, lalu swing HIGH
terkonfirmasi terakhir yang terjadi SETELAH low itu (pasangan yang
sedang "di-retrace"). Zona discount = [H - 0.886*(H-L), H - 0.705*(H-L)].
Untuk bias turun, dibalik: swing HIGH lalu swing LOW setelahnya, zona
premium = [L + 0.705*(H-L), L + 0.886*(H-L)].

Kalau require_fib_zone=True, syarat lokasi jadi GABUNGAN: harus di
bawah/atas value area hari sebelumnya DAN di dalam zona Fib ini.
Kalau pasangan swing yang valid belum ada -- lewati, jangan menebak.

MEAN-REVERSION TANPA BIAS (bias_mode="none") -- UNTUK REGIME SIDEWAYS
--------------------------------------------------------------------------
Aturan trend-following Chris (bias_mode="trend", default) BUTUH struktur
1H/4H yang punya arah -- di regime "sideways" (lihat regime_detector.py),
struktur itu memang tidak ada, jadi sistem trend-following secara
struktural TIDAK PERNAH menghasilkan sinyal di situ, berapa pun syarat
lain dilonggarkan.

bias_mode="none" adalah strategi BERBEDA, bukan versi longgar dari yang
sama: SEMUA syarat bias dilepas, dan LOKASI-nya dibalik maknanya --
bukan lagi "beli di discount karena trennya naik", tapi "beli di VAL
karena harga memantul dalam RENTANG, target kembali ke POC". Dasarnya
mean-reversion murni, bukan trend-following -- masuk akal justru KETIKA
tidak ada tren untuk diikuti. long DAN short dievaluasi independen
(bukan saling meniadakan lewat bias), karena tidak ada bias yang
menentukan arah mana yang "benar".

FUNGSI INI CUMA MENGHASILKAN KANDIDAT SETUP -- tidak mensimulasikan
apakah stop atau target yang kena duluan, tidak menangani "sudah ada
posisi terbuka, abaikan sinyal baru". Itu tanggung jawab simulator
trade (langkah 8), SENGAJA dipisah supaya generate_setups() bisa
diuji sendiri tanpa logika manajemen posisi ikut campur.

TIDAK ADA LOOK-AHEAD: bias dari structure_1h/4h diambil dari bar
TERAKHIR YANG SUDAH SELESAI pada atau sebelum waktu bar konfirmasi
(bukan bar yang sedang berjalan/sebagian). Referensi value area dari
hari KALENDER SEBELUMNYA, bukan hari berjalan.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.microstructure.market_structure import find_swing_points


_FIB_SHALLOW = 0.705
_FIB_DEEP = 0.886


def _fib_zone_uptrend(swings: list, as_of_pos: int, lookback: int) -> tuple[float, float] | None:
    """
    Zona discount untuk bias naik: cari swing LOW terkonfirmasi
    terakhir, lalu swing HIGH terkonfirmasi terakhir yang terjadi
    SETELAH low itu (pasangan yang sedang di-retrace). None kalau
    pasangan valid belum ada.
    """
    confirmed = [s for s in swings if s.index + lookback <= as_of_pos]
    lows = [s for s in confirmed if s.kind == "low"]
    if not lows:
        return None
    last_low = lows[-1]

    highs_after = [s for s in confirmed if s.kind == "high" and s.index > last_low.index]
    if not highs_after:
        return None
    last_high = highs_after[-1]

    L, H = last_low.price, last_high.price
    if H <= L:
        return None
    zone_low = H - _FIB_DEEP * (H - L)
    zone_high = H - _FIB_SHALLOW * (H - L)
    return zone_low, zone_high


def _fib_zone_downtrend(swings: list, as_of_pos: int, lookback: int) -> tuple[float, float] | None:
    """Cermin dari _fib_zone_uptrend: swing HIGH lalu swing LOW setelahnya."""
    confirmed = [s for s in swings if s.index + lookback <= as_of_pos]
    highs = [s for s in confirmed if s.kind == "high"]
    if not highs:
        return None
    last_high = highs[-1]

    lows_after = [s for s in confirmed if s.kind == "low" and s.index > last_high.index]
    if not lows_after:
        return None
    last_low = lows_after[-1]

    H, L = last_high.price, last_low.price
    if H <= L:
        return None
    zone_low = L + _FIB_SHALLOW * (H - L)
    zone_high = L + _FIB_DEEP * (H - L)
    return zone_low, zone_high


def compute_participation_threshold(
    footprint_df: pd.DataFrame,
    window_bars: int = 12,
    percentile: float = 0.25,
) -> float:
    """
    Kalibrasi ambang rata-rata volume dari DataFrame yang diberikan.
    PANGGIL HANYA DENGAN DATA TRAIN kalau dipakai bertahap -- lihat
    catatan di kepala file.

    PENTING -- rata-rata dihitung dari `window_bars` bar SEBELUM bar
    yang sedang dinilai, TIDAK TERMASUK bar itu sendiri (lihat
    _rolling_avg_volume). Kalau bar konfirmasi ikut dihitung, satu bar
    bervolume tinggi (yang MEMANG disyaratkan definisi absorption)
    akan selalu menarik rata-ratanya naik sendiri -- membuat filter ini
    jadi cuma mengukur ulang syarat absorption, bukan menilai rezim di
    SEKITARNYA secara independen. Ini bug yang sungguhan ditemukan di
    data BTC Nero: filter dengan versi lama menghasilkan HASIL IDENTIK
    dengan tanpa filter sama sekali -- 0% dari setup pernah tersaring.

    window_bars=12 default: 12 bar x 5 menit = 1 jam SEBELUM bar
    konfirmasi. percentile=0.25: ambang = kuartil bawah.
    """
    rolling_avg = _rolling_avg_volume(footprint_df, window_bars)
    return float(rolling_avg.dropna().quantile(percentile))


def _rolling_avg_volume(footprint_df: pd.DataFrame, window_bars: int) -> pd.Series:
    """
    Rata-rata volume `window_bars` bar SEBELUM bar saat ini -- TIDAK
    TERMASUK bar itu sendiri (shift(1) dulu baru rolling). Ini SENGAJA
    berbeda dari versi awal yang menyertakan bar konfirmasi -- lihat
    penjelasan panjang di compute_participation_threshold().
    """
    return footprint_df["volume"].shift(1).rolling(window_bars, min_periods=window_bars).mean()


def _nearest_confirmed_swing(
    swings: list,
    as_of_pos: int,
    lookback: int,
    kind: str,
) -> float | None:
    """
    Cari swing point jenis `kind` ("high"/"low") yang PALING BARU tapi
    SUDAH TERKONFIRMASI pada posisi bar `as_of_pos`.

    Swing di index i baru terkonfirmasi setelah `lookback` bar
    berikutnya terlihat (persis definisi find_swing_points: butuh
    jendela i-lookback..i+lookback). Jadi syaratnya BUKAN index < as_of_pos
    saja -- itu akan mengintip swing yang secara teknis belum bisa
    diketahui pada bar ini. Syarat yang benar: index + lookback <= as_of_pos.
    """
    candidates = [s for s in swings
                  if s.kind == kind and s.index + lookback <= as_of_pos]
    if not candidates:
        return None
    return candidates[-1].price  # swing TERBARU yang terkonfirmasi (index terbesar)


@dataclass
class TradeSetup:
    timestamp: pd.Timestamp
    direction: str  # "long" atau "short"
    entry_price: float
    stop_price: float
    target_price: float
    bias_1h: str
    bias_4h: str
    prev_session_id: str
    val_price: float  # value area hari sebelumnya -- dibawa sekalian
    vah_price: float  # supaya trade_simulator bisa deteksi "reclaim value area"
    # tanpa perlu daily_profile terpisah lagi.


def _structure_as_of(structure: pd.Series, t: pd.Timestamp) -> str | None:
    """Nilai struktur dari bar TERAKHIR YANG <= t. None kalau belum ada."""
    eligible = structure[structure.index <= t]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


@dataclass
class _Candidate:
    bar_pos: int
    timestamp: pd.Timestamp
    direction: str
    entry: float
    stop: float  # low (long) / high (short) dari bar KANDIDAT itu sendiri
    bias_1h: str
    bias_4h: str
    prev_session_id: str


def _collect_candidates(
    footprint_df: pd.DataFrame,
    structure_1h: pd.Series,
    structure_4h: pd.Series,
    daily_profile: pd.DataFrame,
    min_avg_volume: float | None = None,
    avg_volume_window: int = 12,
    fib_swings: list | None = None,
    fib_lookback: int = 3,
    require_both_timeframes_agree: bool = True,
    bias_mode: str = "trend",
) -> list[_Candidate]:
    """
    Kumpulkan SEMUA bar yang penuhi bias+lokasi+absorption+flip warna
    (+ rezim partisipasi kalau min_avg_volume diisi) (+ zona Fibonacci
    kalau fib_swings diisi) -- TANPA menghitung target dan TANPA logika
    satu/dua percobaan. Dipakai ulang oleh direct-entry
    (require_two_attempts=False) maupun state machine dua-percobaan,
    supaya syarat dasar ini cuma ditulis SEKALI.

    require_both_timeframes_agree: True (default, keputusan asli
        Nero) = 1H DAN 4H wajib sepakat. False = PELONGGARAN --
        cuma bias 1H yang dipakai, 4H diabaikan sepenuhnya. Ini
        SECARA STRUKTURAL mengubah frekuensi sinyal berkali lipat,
        bukan penyesuaian kecil -- lihat catatan n_trials.
    """
    candidates: list[_Candidate] = []

    avg_volume = (
        _rolling_avg_volume(footprint_df, avg_volume_window)
        if min_avg_volume is not None else None
    )

    for bar_pos, (t, bar) in enumerate(footprint_df.iterrows()):
        if not bool(bar["is_absorption"]):
            continue

        if min_avg_volume is not None:
            v = avg_volume.iloc[bar_pos]
            if pd.isna(v) or v < min_avg_volume:
                continue  # rezim sedang sepi -- buang walau bar ini absorption

        if bias_mode == "none":
            # Mean-reversion: TIDAK BUTUH bias arah sama sekali -- dipakai
            # untuk regime "sideways" di regime_detector.py, di mana
            # struktur trending memang tidak ada untuk dijadikan acuan.
            bias_1h = bias_4h = "none"
        else:
            bias_1h = _structure_as_of(structure_1h, t)
            if require_both_timeframes_agree:
                bias_4h = _structure_as_of(structure_4h, t)
                if bias_1h is None or bias_4h is None or bias_1h != bias_4h:
                    continue
            else:
                bias_4h = bias_1h  # dicatat sama dengan 1h -- 4h tidak lagi jadi syarat
                if bias_1h is None:
                    continue
            if bias_1h not in ("up", "down"):
                continue

        prev_session_id = (t.normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if prev_session_id not in daily_profile.index:
            continue

        prev = daily_profile.loc[prev_session_id]
        is_green = bar["close"] > bar["open"]
        is_red = bar["close"] < bar["open"]
        close = float(bar["close"])

        want_long = is_green and close <= prev["val_price"] and (
            bias_mode == "none" or bias_1h == "up"
        )
        want_short = is_red and close >= prev["vah_price"] and (
            bias_mode == "none" or bias_1h == "down"
        )

        if want_long:
            if fib_swings is not None:
                zone = _fib_zone_uptrend(fib_swings, bar_pos, fib_lookback)
                if zone is None or not (zone[0] <= close <= zone[1]):
                    continue  # belum ada pasangan swing valid, atau di luar zona Fib
            candidates.append(_Candidate(
                bar_pos=bar_pos, timestamp=t, direction="long",
                entry=close, stop=float(bar["low"]),
                bias_1h=bias_1h, bias_4h=bias_4h, prev_session_id=prev_session_id,
            ))
        elif want_short:
            if fib_swings is not None:
                zone = _fib_zone_downtrend(fib_swings, bar_pos, fib_lookback)
                if zone is None or not (zone[0] <= close <= zone[1]):
                    continue
            candidates.append(_Candidate(
                bar_pos=bar_pos, timestamp=t, direction="short",
                entry=close, stop=float(bar["high"]),
                bias_1h=bias_1h, bias_4h=bias_4h, prev_session_id=prev_session_id,
            ))

    return candidates


def _apply_two_attempt_filter(
    candidates: list[_Candidate],
    max_gap_bars: int,
) -> list[_Candidate]:
    """
    State machine per arah: cuma loloskan kandidat yang jadi "percobaan
    KEDUA" -- lihat penjelasan lengkap di docstring modul.
    """
    pending: dict[str, _Candidate] = {}  # direction -> percobaan pertama yang masih berlaku
    confirmed: list[_Candidate] = []

    for c in candidates:
        first = pending.get(c.direction)

        if first is None or (c.bar_pos - first.bar_pos) > max_gap_bars:
            pending[c.direction] = c  # tidak ada percobaan pertama valid -> ini jadi yang baru
            continue

        if c.direction == "long":
            is_higher_low = c.stop > first.stop
        else:  # short
            is_higher_low = c.stop < first.stop  # "lebih rendah" untuk short

        if is_higher_low:
            confirmed.append(c)  # percobaan KEDUA -- ini yang jadi setup
            del pending[c.direction]  # mulai lagi dari awal untuk sikuen berikutnya
        else:
            pending[c.direction] = c  # penjual/pembeli menang -> jadi percobaan pertama BARU

    return confirmed


def _finalize_setup(
    c: _Candidate,
    target_mode: str,
    daily_profile: pd.DataFrame,
    swings: list | None,
    swing_lookback: int,
) -> TradeSetup | None:
    """Hitung target untuk satu kandidat, terapkan pengecekan arah yang masuk akal."""
    if target_mode == "swing":
        kind = "high" if c.direction == "long" else "low"
        target = _nearest_confirmed_swing(swings, c.bar_pos, swing_lookback, kind)
        if target is None:
            return None
    else:
        target = float(daily_profile.loc[c.prev_session_id, "poc_price"])

    if c.direction == "long":
        if target <= c.entry or c.stop >= c.entry:
            return None
    else:
        if target >= c.entry or c.stop <= c.entry:
            return None

    return TradeSetup(
        timestamp=c.timestamp, direction=c.direction, entry_price=c.entry,
        stop_price=c.stop, target_price=target,
        bias_1h=c.bias_1h, bias_4h=c.bias_4h, prev_session_id=c.prev_session_id,
        val_price=float(daily_profile.loc[c.prev_session_id, "val_price"]),
        vah_price=float(daily_profile.loc[c.prev_session_id, "vah_price"]),
    )


def generate_setups(
    footprint_df: pd.DataFrame,
    structure_1h: pd.Series,
    structure_4h: pd.Series,
    daily_profile: pd.DataFrame,
    target_mode: str = "prev_poc",
    swing_lookback: int = 3,
    require_two_attempts: bool = False,
    two_attempt_max_gap_bars: int = 48,
    min_avg_volume: float | None = None,
    avg_volume_window: int = 12,
    require_fib_zone: bool = False,
    require_both_timeframes_agree: bool = True,
    bias_mode: str = "trend",
) -> pd.DataFrame:
    """
    target_mode: "prev_poc" (default, lihat docstring modul) atau
        "swing" -- target = swing high/low TERKONFIRMASI terakhir di
        footprint_df itu sendiri (bar 5 menit, sesuai chart yang Chris
        sebut dia pakai). Dua mode ini HIPOTESIS TERPISAH -- lihat
        catatan di kepala file soal n_trials.

    require_two_attempts: False (default) = masuk di percobaan PERTAMA
        yang penuhi syarat (perilaku lama, tidak berubah). True = WAJIB
        ada percobaan pertama yang gagal dulu, baru masuk di percobaan
        kedua -- lihat docstring modul untuk detail lengkap. TRIAL
        TERPISAH dari target_mode.

    two_attempt_max_gap_bars: berapa lama (dalam bar) percobaan pertama
        tetap "berlaku" menunggu percobaan kedua. Diabaikan kalau
        require_two_attempts=False.

    min_avg_volume: None (default, tidak ada filter, perilaku lama)
        atau angka dari compute_participation_threshold() -- buang
        setup di rezim volume sepi. TRIAL TERPISAH LAGI dari dua yang
        di atas -- lihat docstring modul.
    avg_volume_window: jumlah bar untuk menghitung rata-rata rezim.
        Diabaikan kalau min_avg_volume=None.

    require_fib_zone: False (default, perilaku lama) atau True --
        syarat lokasi jadi GABUNGAN value area DAN zona Fibonacci
        (70.5%-88.6% retracement dari pasangan swing low->high yang
        sedang di-retrace) -- lihat docstring modul. TRIAL TERPISAH
        LAGI dari tiga yang di atas.

    footprint_df: WAJIB sudah punya kolom 'is_absorption' (hasil
        detect_absorption() -- lihat absorption.py), plus open/high/
        low/close/volume dari footprint.py.
    daily_profile: hasil build_profile_history() dari volume_profile.py.
        Boleh punya 'session_id' sebagai kolom ATAU index -- ditangani
        otomatis. Diabaikan sepenuhnya kalau target_mode="swing" (tapi
        LOCATION tetap dari value area hari sebelumnya di kedua mode --
        cuma TARGET yang beda, bukan syarat lokasi).

    Return DataFrame kandidat setup, urut waktu. Kosong kalau tidak
    ada satu pun bar yang memenuhi syarat.
    """
    if target_mode not in ("prev_poc", "swing"):
        raise ValueError(f"target_mode harus 'prev_poc' atau 'swing', dapat: {target_mode!r}")

    if "session_id" in daily_profile.columns:
        daily_profile = daily_profile.set_index("session_id")

    if "is_absorption" not in footprint_df.columns:
        raise ValueError(
            "footprint_df butuh kolom 'is_absorption' -- jalankan "
            "detect_absorption() dulu dan gabungkan hasilnya sebagai kolom."
        )

    need_swings = target_mode == "swing" or require_fib_zone
    swings = find_swing_points(footprint_df, lookback=swing_lookback) if need_swings else None
    fib_swings = swings if require_fib_zone else None

    candidates = _collect_candidates(
        footprint_df, structure_1h, structure_4h, daily_profile,
        min_avg_volume=min_avg_volume, avg_volume_window=avg_volume_window,
        fib_swings=fib_swings, fib_lookback=swing_lookback,
        require_both_timeframes_agree=require_both_timeframes_agree,
        bias_mode=bias_mode,
    )
    if require_two_attempts:
        candidates = _apply_two_attempt_filter(candidates, two_attempt_max_gap_bars)

    setups = [
        s for c in candidates
        if (s := _finalize_setup(c, target_mode, daily_profile, swings, swing_lookback)) is not None
    ]

    if not setups:
        return pd.DataFrame(columns=[
            "timestamp", "direction", "entry_price", "stop_price", "target_price",
            "bias_1h", "bias_4h", "prev_session_id",
        ])

    return pd.DataFrame([vars(s) for s in setups]).set_index("timestamp").sort_index()