"""
3_KarlWeierstrass_NEXT7_2b3e — POGODNI deo iz 1_KarlWeierstrass_v2.py
Aparat 2b: Hurst / R-S analiza  +  Test 3e: NIST baterija (NIST-style)

Self-contained:
  - KORAK 1: ucitavanje 4624 izvlacenja i izgradnja f(t) = lex-indeks
  - KORAK 2b: rolling/local Hurst (priprema)
  - KORAK 2b3e: NIST-style testovi nad binarizovanim rolling H (bit=1 ako H>0.5)
                + f(t) NIST kontrola

Output:
  3_KarlWeierstrass_NEXT7_2b3e.png
  3_KarlWeierstrass_NEXT7_2b3e.txt
"""

import csv
import math
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


T0 = time.time()

CSV_DRAWS = "/data/loto7_4624_k43.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT7_2b3e.png")
TXT_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT7_2b3e.txt")

N_MAX = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_MAX, K_PICK)


# ─── helperi (samo oni potrebni za 2b + 2b3e) ────────────────────────
def read_loto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < K_PICK:
                continue
            try:
                nums = tuple(sorted(int(x) for x in row[:K_PICK]))
            except ValueError:
                continue
            if len(nums) == K_PICK and len(set(nums)) == K_PICK:
                rows.append(nums)
    return rows


def lex_rank_1based(combo, n=N_MAX, k=K_PICK):
    """1-based lex indeks (poklapa se sa rednim brojem u kombinacije_39C7.csv)."""
    combo = tuple(sorted(combo))
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo):
        remaining = k - i - 1
        for candidate in range(prev + 1, value):
            rank0 += math.comb(n - candidate, remaining)
        prev = value
    return rank0 + 1


def hurst_rs(series, min_window=8, max_window=None):
    """R/S Hurst procena: slope log(R/S) prema log(window)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if max_window is None:
        max_window = max(min_window * 2, n // 4)

    windows = []
    w = min_window
    while w <= max_window:
        windows.append(w)
        w = int(w * 1.45) + 1

    used_windows = []
    rs_values = []
    for w in windows:
        chunks = n // w
        if chunks < 2:
            continue
        vals = []
        for i in range(chunks):
            seg = x[i * w:(i + 1) * w]
            y = seg - seg.mean()
            z = np.cumsum(y)
            r = z.max() - z.min()
            s = seg.std(ddof=1)
            if s > 0:
                vals.append(r / s)
        if vals:
            used_windows.append(w)
            rs_values.append(float(np.mean(vals)))

    used_windows = np.asarray(used_windows, dtype=float)
    rs_values = np.asarray(rs_values, dtype=float)
    slope, intercept = np.polyfit(np.log(used_windows), np.log(rs_values), 1)
    fit = intercept + slope * np.log(used_windows)
    ss_res = float(np.sum((np.log(rs_values) - fit) ** 2))
    ss_tot = float(np.sum((np.log(rs_values) - np.log(rs_values).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2), used_windows, rs_values


def rolling_hurst_rs(series, window=768, step=128):
    """Rolling R/S Hurst procena kroz vreme."""
    x = np.asarray(series, dtype=float)
    centers = []
    hvals = []
    r2vals = []
    for start in range(0, len(x) - window + 1, step):
        seg = x[start:start + window]
        h, _, r2, _, _ = hurst_rs(seg, min_window=8, max_window=max(32, window // 4))
        centers.append(start + window // 2 + 1)
        hvals.append(h)
        r2vals.append(r2)
    return (
        np.asarray(centers, dtype=float),
        np.asarray(hvals, dtype=float),
        np.asarray(r2vals, dtype=float),
    )


def nist_bits_from_series(series):
    """Binarizacija za NIST-style test: 1 ako je vrednost iznad medijane."""
    x = np.asarray(series, dtype=float)
    med = float(np.median(x))
    bits = (x > med).astype(int)
    if bits.sum() == 0 or bits.sum() == len(bits):
        bits = (x > x.mean()).astype(int)
    return bits


def nist_monobit(bits):
    n = len(bits)
    s_obs = abs(int(np.sum(2 * bits - 1))) / np.sqrt(n)
    p = math.erfc(s_obs / np.sqrt(2))
    return float(p), float(s_obs)


def nist_runs(bits):
    n = len(bits)
    pi = float(bits.mean())
    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return 0.0, 0, pi
    runs = int(1 + np.sum(bits[1:] != bits[:-1]))
    denom = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    p = math.erfc(abs(runs - 2 * n * pi * (1 - pi)) / denom)
    return float(p), runs, pi


def nist_block_frequency(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return float("nan"), float("nan"), 0
    trimmed = bits[:n_blocks * block_size].reshape(n_blocks, block_size)
    props = trimmed.mean(axis=1)
    chi2 = 4 * block_size * float(np.sum((props - 0.5) ** 2))
    p = float(stats.chi2.sf(chi2, n_blocks))
    return p, chi2, n_blocks


def nist_cumulative_sums(bits):
    x = 2 * bits - 1
    walk = np.cumsum(x)
    z = int(np.max(np.abs(walk)))
    if z == 0:
        return 1.0, z, walk
    n = len(bits)
    p = math.erfc(z / np.sqrt(2 * n))
    return float(p), z, walk


def _pattern_counts_circular(bits, m):
    n = len(bits)
    ext = np.concatenate([bits, bits[:m - 1]])
    counts = np.zeros(2 ** m, dtype=float)
    for i in range(n):
        value = 0
        for b in ext[i:i + m]:
            value = (value << 1) | int(b)
        counts[value] += 1
    return counts


def nist_approximate_entropy(bits, m=2):
    n = len(bits)

    def phi(mm):
        counts = _pattern_counts_circular(bits, mm)
        probs = counts[counts > 0] / n
        return float(np.sum(probs * np.log(probs)))

    ap_en = phi(m) - phi(m + 1)
    chi2 = 2 * n * (np.log(2) - ap_en)
    df = 2 ** (m - 1)
    p = float(stats.chi2.sf(chi2, df))
    return p, float(ap_en), float(chi2), df


# ─── KORAK 1: f(t) = lex-indeks ──────────────────────────────────────
draws = read_loto_csv(CSV_DRAWS)
N = len(draws)
lex_idx = np.array([lex_rank_1based(c) for c in draws], dtype=np.float64)

print()
print("3_KarlWeierstrass_NEXT7_2b3e — KORAK 1: formiranje krive f(t)")
print(f"  CSV:                  {CSV_DRAWS}")
print(f"  Ucitano izvlacenja:    {N}")
print(f"  C(39,7):              {TOTAL_COMBOS:,}")
print()

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("3_KarlWeierstrass_NEXT7_2b3e — Hurst/R-S + NIST baterija (POGODNO)\n")
    f.write("=" * 60 + "\n\n")
    f.write("KORAK 1: Weierstrass-ova funkcija nad svih izvucenih kombinacija\n\n")
    f.write(f"  CSV izvucenih:        {CSV_DRAWS}\n")
    f.write(f"  Ucitano izvlacenja:    {N}\n")
    f.write(f"  C(39,7):              {TOTAL_COMBOS:,}\n")
    f.write("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7\n\n")


# ─── KORAK 2b: rolling/local Hurst (priprema) ────────────────────────
rolling_window = 768
rolling_step = 128
roll_centers, roll_h, roll_r2 = rolling_hurst_rs(
    lex_idx, window=rolling_window, step=rolling_step
)


# ─── KORAK 2b3e: NIST baterija nad binarizovanim rolling H ───────────
T0_2B3E = time.time()

roll_h_bits = (roll_h > 0.5).astype(int)
roll_h_n = len(roll_h_bits)
roll_h_ones = int(roll_h_bits.sum())
roll_h_zeros = int(roll_h_n - roll_h_ones)

roll_monobit_p, roll_monobit_s = nist_monobit(roll_h_bits)
roll_runs_p, roll_runs_count, roll_runs_pi = nist_runs(roll_h_bits)
roll_block_p, roll_block_chi2, roll_block_count = nist_block_frequency(
    roll_h_bits, block_size=8
)
roll_cusum_p, roll_cusum_z, roll_cusum_walk = nist_cumulative_sums(roll_h_bits)
roll_apen_p, roll_apen_value, roll_apen_chi2, roll_apen_df = nist_approximate_entropy(
    roll_h_bits, m=2
)

roll_nist_rows = [
    ("Monobit frequency", roll_monobit_p, roll_monobit_s),
    ("Runs", roll_runs_p, roll_runs_count),
    ("Block frequency", roll_block_p, roll_block_chi2),
    ("Cumulative sums", roll_cusum_p, roll_cusum_z),
    ("Approx entropy", roll_apen_p, roll_apen_value),
]
roll_nist_pass_count = int(sum(p > 0.05 for _, p, _ in roll_nist_rows if np.isfinite(p)))
roll_nist_total = int(sum(np.isfinite(p) for _, p, _ in roll_nist_rows))

f_control_bits = nist_bits_from_series(lex_idx)
f_monobit_p, f_monobit_s = nist_monobit(f_control_bits)
f_runs_p, f_runs_count, f_runs_pi = nist_runs(f_control_bits)
f_block_p, f_block_chi2, f_block_count = nist_block_frequency(f_control_bits, block_size=128)
f_cusum_p, f_cusum_z, _ = nist_cumulative_sums(f_control_bits)
f_apen_p, f_apen_value, f_apen_chi2, f_apen_df = nist_approximate_entropy(f_control_bits, m=2)
f_nist_rows = [
    ("Monobit frequency", f_monobit_p, f_monobit_s),
    ("Runs", f_runs_p, f_runs_count),
    ("Block frequency", f_block_p, f_block_chi2),
    ("Cumulative sums", f_cusum_p, f_cusum_z),
    ("Approx entropy", f_apen_p, f_apen_value),
]
f_nist_pass_count = int(sum(p > 0.05 for _, p, _ in f_nist_rows if np.isfinite(p)))
f_nist_total = int(sum(np.isfinite(p) for _, p, _ in f_nist_rows))

if roll_nist_pass_count == roll_nist_total:
    roll_nist_note = "rolling H bitovi prolaze sve NIST-style testove"
elif roll_nist_pass_count >= max(1, roll_nist_total - 1):
    roll_nist_note = "rolling H bitovi uglavnom prolaze, slab signal za proveru"
else:
    roll_nist_note = "rolling H bitovi padaju vise NIST-style testova"

print()
print("KORAK 2b3e: Aparat 2b Hurst/R-S + Test 3e NIST baterija")
print(f"  rolling H bits: n={roll_h_n}  zeros={roll_h_zeros}  ones={roll_h_ones}")
for name, p, stat_val in roll_nist_rows:
    print(f"  {name:<20} p={p:.6f}  stat={stat_val}")
print(f"  prolaz rolling H: {roll_nist_pass_count}/{roll_nist_total}  ⇒ {roll_nist_note}")
print(f"  kontrola f(t) prolaz: {f_nist_pass_count}/{f_nist_total}")
print()

fig2b3e, ax2b3e = plt.subplots(1, 3, figsize=(16, 5))
fig2b3e.suptitle("KORAK 2b3e: Hurst/R-S aparat + NIST-style testovi  (POGODNO)",
                 fontsize=13, fontweight="bold")

ax2b3e[0].bar(["H<=0.5", "H>0.5"], [roll_h_zeros, roll_h_ones],
              color=["steelblue", "darkslateblue"])
ax2b3e[0].set_title("Binarizovani rolling H rezimi")
ax2b3e[0].set_xlabel("bit")
ax2b3e[0].set_ylabel("broj")
ax2b3e[0].grid(True, alpha=0.2, axis="y")

names = [row[0] for row in roll_nist_rows]
roll_pvals = np.array([row[1] for row in roll_nist_rows], dtype=float)
colors = ["seagreen" if p > 0.05 else "crimson" for p in roll_pvals]
ax2b3e[1].barh(names, roll_pvals, color=colors)
ax2b3e[1].axvline(0.05, color="black", linestyle="--", linewidth=1.2)
ax2b3e[1].set_xlim(0, 1)
ax2b3e[1].set_title("Rolling H NIST-style p-vrednosti")
ax2b3e[1].set_xlabel("p-value")
ax2b3e[1].grid(True, alpha=0.2, axis="x")

ax2b3e[2].plot(np.arange(1, roll_h_n + 1), roll_cusum_walk,
               linewidth=1.2, marker="o", markersize=3, color="purple")
ax2b3e[2].axhline(0, color="black", linewidth=0.6)
ax2b3e[2].set_title(f"Rolling H cumulative sums (z={roll_cusum_z})")
ax2b3e[2].set_xlabel("rolling window index")
ax2b3e[2].set_ylabel("cum sum")
ax2b3e[2].grid(True, alpha=0.25)

for a in ax2b3e:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig2b3e.tight_layout()
fig2b3e.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
plt.show()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("KORAK 2b3e: Aparat 2b Hurst/R-S + Test 3e NIST baterija\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  PNG:                  {PNG_PATH}\n\n")
    f.write("Binarizacija rolling/local Hurst niza:\n")
    f.write("  bit = 1 ako je lokalni H > 0.5, inace 0\n")
    f.write(f"  n bits                = {roll_h_n}\n")
    f.write(f"  zeros                 = {roll_h_zeros}\n")
    f.write(f"  ones                  = {roll_h_ones}\n\n")
    f.write("NIST-style testovi nad rolling H bitovima (prolaz ako p > 0.05):\n")
    f.write(f"  {'test':<22}{'p-value':>14}{'stat':>18}{'pass':>10}\n")
    for name, p, stat_val in roll_nist_rows:
        f.write(f"  {name:<22}{p:>14,.8f}{float(stat_val):>18,.8f}{str(p > 0.05):>10}\n")
    f.write("\n")
    f.write("Detalji rolling H:\n")
    f.write(f"  Runs pi               = {roll_runs_pi:.8f}\n")
    f.write(f"  Block size            = 8\n")
    f.write(f"  Block count           = {roll_block_count}\n")
    f.write(f"  Approx entropy m      = 2\n")
    f.write(f"  Approx entropy chi2   = {roll_apen_chi2:.8f}\n")
    f.write(f"  Approx entropy df     = {roll_apen_df}\n")
    f.write(f"  pass count            = {roll_nist_pass_count}/{roll_nist_total}\n")
    f.write(f"  interpret.            = {roll_nist_note}\n\n")
    f.write("Kontrola: NIST-style testovi nad f(t) binarizacijom:\n")
    f.write(f"  {'test':<22}{'p-value':>14}{'stat':>18}{'pass':>10}\n")
    for name, p, stat_val in f_nist_rows:
        f.write(f"  {name:<22}{p:>14,.8f}{float(stat_val):>18,.8f}{str(p > 0.05):>10}\n")
    f.write(f"  pass count            = {f_nist_pass_count}/{f_nist_total}\n")
    f.write(f"  f(t) runs pi          = {f_runs_pi:.8f}\n")
    f.write(f"  f(t) block count      = {f_block_count}\n")
    f.write(f"  f(t) approx chi2      = {f_apen_chi2:.8f}\n")
    f.write(f"  f(t) approx df        = {f_apen_df}\n\n")

    elapsed_2b3e = time.time() - T0_2B3E
    f.write(f"Vreme KORAKA 2b3e: {timedelta(seconds=int(elapsed_2b3e))} ({elapsed_2b3e:.1f} s)\n")
    f.write(f"Ukupno vreme:       {timedelta(seconds=int(time.time()-T0))} ({time.time()-T0:.1f} s)\n")

print(f"PNG saved → {PNG_PATH}")
print(f"TXT saved → {TXT_PATH}")
print(f"Vreme KORAKA 2b3e: {timedelta(seconds=int(time.time()-T0_2B3E))} "
      f"({time.time()-T0_2B3E:.1f} s)")
print(f"Ukupno vreme:      {timedelta(seconds=int(time.time()-T0))} "
      f"({time.time()-T0:.1f} s)")
print()
print("KRAJ 3_KarlWeierstrass_NEXT7_2b3e.")
print()
"""
3_KarlWeierstrass_NEXT7_2b3e — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2b3e: Aparat 2b Hurst/R-S + Test 3e NIST baterija
  rolling H bits: n=31  zeros=0  ones=31
  Monobit frequency    p=0.000000  stat=5.567764362830022
  Runs                 p=0.000000  stat=0
  Block frequency      p=0.000025  stat=24.0
  Cumulative sums      p=0.000000  stat=31
  Approx entropy       p=0.000000  stat=0.0
  prolaz rolling H: 0/5  ⇒ rolling H bitovi padaju vise NIST-style testova
  kontrola f(t) prolaz: 4/5

PNG saved → /3_KarlWeierstrass_NEXT7_2b3e.png
TXT saved → /3_KarlWeierstrass_NEXT7_2b3e.txt
Vreme KORAKA 2b3e: 0:00:27 (27.6 s)
Ukupno vreme:      0:00:27 (27.8 s)

KRAJ 3_KarlWeierstrass_NEXT7_2b3e.
"""



###############   PREDIKCIJA 7  ###############################

"""
NEXT7 (2b3e, rolling H bits) — svi prozori H>0.5, drži se trenda + skok = lokalna varijansa.
"""


def lex_unrank_1based(rank, n=N_MAX, k=K_PICK):
    """Vracanje 1-based lex indeksa u Loto 7/39 kombinaciju."""
    rank0 = int(rank) - 1
    combo = []
    prev = 0
    for i in range(k):
        remaining = k - i - 1
        for candidate in range(prev + 1, n + 1):
            count = math.comb(n - candidate, remaining)
            if rank0 >= count:
                rank0 -= count
            else:
                combo.append(candidate)
                prev = candidate
                break
    return tuple(combo)


T0_PRED7 = time.time()

# Rolling H bitovi su svi 1: signal nije fino-granularan, ali jasno kaze
# da je rezim perzistentan. Zato koristimo lokalni trend i lokalnu varijansu.
last_lex = float(lex_idx[-1])
last_h = float(roll_h[-1])
mean_h = float(roll_h.mean())
persistent_ratio = float(roll_h_ones / roll_h_n) if roll_h_n else 0.0

local_window = rolling_window
local_y = np.asarray(lex_idx[-local_window:], dtype=float)
local_x = np.arange(len(local_y), dtype=float)
local_slope, local_intercept = np.polyfit(local_x, local_y, 1)
local_fit = local_intercept + local_slope * local_x
local_resid = local_y - local_fit
local_resid_std = float(local_resid.std(ddof=1))

recent_incr = np.diff(local_y)
recent_mean_incr = float(recent_incr.mean())
recent_std_incr = float(recent_incr.std(ddof=1))
last_incr = float(np.diff(lex_idx)[-1])

trend_strength = float(np.clip((mean_h - 0.5) / 0.15, 0.0, 1.0)) * persistent_ratio
pred_incr = (1.0 - trend_strength) * recent_mean_incr + trend_strength * last_incr
pred_lex_float = last_lex + pred_incr
pred_lex = int(np.clip(round(pred_lex_float), 1, TOTAL_COMBOS))
pred_combo = lex_unrank_1based(pred_lex)

# Ovde koristimo lokalnu varijansu inkremenata, jer NIST-bit aparat ne daje amplitudu.
z_grid = [-1.28, -0.84, -0.43, 0.0, 0.43, 0.84, 1.28]
candidate_rows = []
seen_lex = set()
for z in z_grid:
    cand_lex = int(np.clip(round(pred_lex_float + z * recent_std_incr), 1, TOTAL_COMBOS))
    if cand_lex in seen_lex:
        continue
    seen_lex.add(cand_lex)
    candidate_rows.append((z, cand_lex, lex_unrank_1based(cand_lex)))

print()
print("PREDIKCIJA 7 — NEXT7 / 2b3e / rolling H bit rezim")
print(f"  rolling H bits         = zeros:{roll_h_zeros} ones:{roll_h_ones}")
print(f"  persistent ratio       = {persistent_ratio:.8f}")
print(f"  mean rolling H         = {mean_h:.8f}")
print(f"  zadnji rolling H       = {last_h:.8f}")
print(f"  trend strength         = {trend_strength:.6f}")
print(f"  lokalni slope          = {local_slope:,.2f}")
print(f"  recent mean dX         = {recent_mean_incr:,.2f}")
print(f"  recent std dX          = {recent_std_incr:,.2f}")
print(f"  zadnji dX              = {last_incr:,.2f}")
print(f"  pred. inkrement        = {pred_incr:,.2f}")
print(f"  pred. lex              = {pred_lex:,}")
print(f"  pred. kombinacija      = {pred_combo}")
print("  kandidati trend + lokalna varijansa:")
for z, cand_lex, combo in candidate_rows:
    print(f"    z={z:>5.2f}  lex={cand_lex:>10,}  combo={combo}")
print()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("PREDIKCIJA 7: NEXT7 / 2b3e / rolling H bit rezim\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model:\n")
    f.write("  Rolling H bitovi padaju NIST testove jer su svi H>0.5.\n")
    f.write("  To je grubi rezimski filter: tretira se kao stabilno perzistentan rezim.\n")
    f.write("  Prognoza kombinuje lokalni trend, zadnji inkrement i lokalnu varijansu.\n\n")
    f.write("Parametri:\n")
    f.write(f"  rolling H zeros        = {roll_h_zeros}\n")
    f.write(f"  rolling H ones         = {roll_h_ones}\n")
    f.write(f"  persistent ratio       = {persistent_ratio:.8f}\n")
    f.write(f"  mean rolling H         = {mean_h:.8f}\n")
    f.write(f"  zadnji rolling H       = {last_h:.8f}\n")
    f.write(f"  trend strength         = {trend_strength:.8f}\n")
    f.write(f"  local window           = {local_window}\n")
    f.write(f"  lokalni slope          = {local_slope:,.8f}\n")
    f.write(f"  lokalni resid std      = {local_resid_std:,.8f}\n")
    f.write(f"  recent mean dX         = {recent_mean_incr:,.8f}\n")
    f.write(f"  recent std dX          = {recent_std_incr:,.8f}\n")
    f.write(f"  zadnji dX              = {last_incr:,.8f}\n")
    f.write(f"  zadnji lex             = {int(last_lex):,}\n")
    f.write(f"  pred. inkrement        = {pred_incr:,.8f}\n\n")
    f.write("Glavna prognoza:\n")
    f.write(f"  pred. lex float        = {pred_lex_float:,.8f}\n")
    f.write(f"  pred. lex              = {pred_lex:,}\n")
    f.write(f"  pred. kombinacija      = {pred_combo}\n\n")
    f.write("Kandidati trend + lokalna varijansa:\n")
    f.write(f"  {'z':>8}{'lex':>14}  kombinacija\n")
    for z, cand_lex, combo in candidate_rows:
        f.write(f"  {z:>8.2f}{cand_lex:>14,}  {combo}\n")
    f.write("\n")
    elapsed_pred7 = time.time() - T0_PRED7
    f.write(f"Vreme PREDIKCIJE 7: {timedelta(seconds=int(elapsed_pred7))} ({elapsed_pred7:.1f} s)\n")

print(f"TXT updated → {TXT_PATH}")
print(f"Vreme PREDIKCIJE 7: {timedelta(seconds=int(time.time()-T0_PRED7))} "
      f"({time.time()-T0_PRED7:.1f} s)")
print()


"""
Pošto je ovde signal NIST nad rolling H bitovima, predikcija će koristiti H>0.5 režim kao filter perzistentnog trenda.

Pošto su svi rolling H bitovi 1, tretiram režim kao stabilno perzistentan i pravim trend + lokalna varijansa kandidate.

koristi činjenicu da su rolling H bitovi svi 1 (H > 0.5)
tretira to kao stabilan perzistentni režim
kombinuje lokalni trend, zadnji inkrement i lokalnu varijansu
generiše glavnu Loto kombinaciju + kandidate
upisuje u 3_KarlWeierstrass_NEXT7_2b3e.txt
"""


"""
3_KarlWeierstrass_NEXT7_2b3e — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2b3e: Aparat 2b Hurst/R-S + Test 3e NIST baterija
  rolling H bits: n=31  zeros=0  ones=31
  Monobit frequency    p=0.000000  stat=5.567764362830022
  Runs                 p=0.000000  stat=0
  Block frequency      p=0.000025  stat=24.0
  Cumulative sums      p=0.000000  stat=31
  Approx entropy       p=0.000000  stat=0.0
  prolaz rolling H: 0/5  ⇒ rolling H bitovi padaju vise NIST-style testova
  kontrola f(t) prolaz: 4/5

PNG saved → /3_KarlWeierstrass_NEXT7_2b3e.png
TXT saved → /3_KarlWeierstrass_NEXT7_2b3e.txt
Vreme KORAKA 2b3e: 0:00:03 (3.3 s)
Ukupno vreme:      0:00:03 (3.5 s)

KRAJ 3_KarlWeierstrass_NEXT7_2b3e.


PREDIKCIJA 7 — NEXT7 / 2b3e / rolling H bit rezim
  rolling H bits         = zeros:0 ones:31
  persistent ratio       = 1.00000000
  mean rolling H         = 0.59744976
  zadnji rolling H       = 0.62382045
  trend strength         = 0.649665
  lokalni slope          = -739.26
  recent mean dX         = -6,859.23
  recent std dX          = 6,269,244.17
  zadnji dX              = -2,143,496.00
  pred. inkrement        = -1,394,957.53
  pred. lex              = 1
  pred. kombinacija      = (1, 2, 3, 4, 5, 6, 7)
  kandidati trend + lokalna varijansa:
    z=-1.28  lex=         1  combo=(1, 2, 3, 4, 5, 6, 7)
    z= 0.43  lex= 1,813,931  combo=(1, x, 16, y, 21, z, 36)
    z= 0.84  lex= 4,384,322  combo=(2, x, 11, y, 27, z,36)
    z= 1.28  lex= 7,142,789  combo=(4, x, 9, y, 14, z, 25)

TXT updated → /3_KarlWeierstrass_NEXT7_2b3e.txt
Vreme PREDIKCIJE 7: 0:00:00 (0.0 s)
"""
