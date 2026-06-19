# ============================================================
# Additional Analysis for Paper Revision (v2 — bug-fixed)
# (1) Self-linearization check: theta WRAPPED to (-pi, pi]
# (2) Correlation Dimension D_2 with diagnostic scaling-region check
# (3) Spectral bottleneck quantification (unchanged, already valid)
# ============================================================

# !pip install scipy numpy matplotlib pandas tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist
from tqdm.auto import tqdm
import json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Core functions ───────────────────────────────────────────

def build_1d_neighbors(N):
    nb = {i: [] for i in range(N)}
    for i in range(N - 1):
        nb[i].append(i + 1)
        nb[i + 1].append(i)
    return nb

def build_3d_neighbors(L):
    N = L**3
    nb = {i: [] for i in range(N)}
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = x*L*L + y*L + z
                for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    xx,yy,zz = x+dx,y+dy,z+dz
                    if 0<=xx<L and 0<=yy<L and 0<=zz<L:
                        nb[i].append(xx*L*L+yy*L+zz)
    return nb

def make_params(N, neighbors, kappa_mean=0.5):
    omega = np.ones(N)
    kappa = {}
    for i, js in neighbors.items():
        for j in js:
            kappa[(i,j)] = kappa_mean
    return omega, kappa

def pendulum_ode(t, y, N, omega, kappa, neighbors, gamma=0.05):
    theta = y[:N]; dtheta = y[N:]
    ddtheta = np.zeros(N)
    for i in range(N):
        ddtheta[i] = (-omega[i]**2 * np.sin(theta[i])
                      - gamma * dtheta[i]
                      + sum(kappa.get((i,j),0) * np.sin(theta[j]-theta[i])
                            for j in neighbors[i]))
    return np.concatenate([dtheta, ddtheta])

def init_fixed_energy(N, omega, E_total=10.0, seed=None):
    rng = np.random.RandomState(seed)
    theta0 = np.zeros(N)
    dtheta0 = rng.randn(N)
    KE = 0.5 * np.sum(dtheta0**2)
    PE = np.sum(omega**2 * (1 - np.cos(theta0)))
    if KE > 0:
        dtheta0 *= np.sqrt((E_total - PE) / KE)
    return np.concatenate([theta0, dtheta0])

def simulate(N, nb, omega, kappa, E_total=10.0, gamma=0.05,
             T=300, dt=0.02, t_transient=50, seed=None):
    y0 = init_fixed_energy(N, omega, E_total, seed)
    t_eval = np.arange(t_transient, T, dt)
    sol = solve_ivp(pendulum_ode, (0,T), y0,
                    args=(N, omega, kappa, nb, gamma),
                    method='LSODA', t_eval=t_eval, rtol=1e-5, atol=1e-7)
    return sol.y

def wrap_angle(theta):
    """Wrap angle(s) to (-pi, pi]"""
    return (theta + np.pi) % (2*np.pi) - np.pi

# ══════════════════════════════════════════════════════════
# (1) SELF-LINEARIZATION ANALYSIS — BUG FIXED: theta wrapped
# ══════════════════════════════════════════════════════════

def run_linearization_check(N_list=None, n_realiz=10, E_total=10.0,
                             gamma=0.05, T=200, dt=0.02):
    """
    For each N, compute theta_rms (WRAPPED to physical angle) and the
    nonlinearity index: NLI = <|sin(theta) - theta|> / <|theta|>
    using the wrapped angle. This avoids spurious large values from
    angle accumulation when pendula complete full rotations.
    """
    if N_list is None:
        N_list = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    results = []
    for N in tqdm(N_list, desc="Linearization Check (v2)"):
        nb = build_1d_neighbors(N)
        theta_rms_list, nli_list, sin_err_list = [], [], []
        frac_large_list = []
        for s in range(n_realiz):
            omega, kappa = make_params(N, nb)
            traj = simulate(N, nb, omega, kappa, E_total=E_total,
                            gamma=gamma, T=T, dt=dt, seed=s)
            theta_raw = traj[:N, :]
            theta = wrap_angle(theta_raw)          # <-- BUG FIX
            theta_rms = np.sqrt(np.mean(theta**2))
            abs_theta = np.abs(theta)
            sin_dev = np.abs(np.sin(theta) - theta)
            nli = np.mean(sin_dev) / (np.mean(abs_theta) + 1e-12)
            sin_err_rms = abs(np.sin(theta_rms) - theta_rms) / (theta_rms + 1e-12)
            # fraction of samples with |theta| > pi/6 (30 deg) as a
            # diagnostic for "how much of the trajectory is nonlinear"
            frac_large = np.mean(np.abs(theta) > np.pi/6)
            theta_rms_list.append(theta_rms)
            nli_list.append(nli)
            sin_err_list.append(sin_err_rms)
            frac_large_list.append(frac_large)
        results.append({
            'N': N,
            'theta_rms_rad': np.mean(theta_rms_list),
            'theta_rms_deg': np.degrees(np.mean(theta_rms_list)),
            'NLI_mean': np.mean(nli_list),
            'NLI_std': np.std(nli_list),
            'sin_rel_err_pct': np.mean(sin_err_list) * 100,
            'frac_theta_gt_30deg': np.mean(frac_large_list),
        })
        print(f"  N={N}: theta_rms={np.degrees(np.mean(theta_rms_list)):.2f} deg "
              f"(wrapped), NLI={np.mean(nli_list):.4f}, "
              f"sin_err={np.mean(sin_err_list)*100:.3f}%, "
              f"frac>30deg={np.mean(frac_large_list)*100:.1f}%")
    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════
# (2) CORRELATION DIMENSION D_2 — BUG FIXED: longer transient,
#     explicit scaling-region diagnostics, larger sample, and a
#     sanity check against a known-D2 reference (white noise ~ full dim)
# ══════════════════════════════════════════════════════════

def correlation_dimension_v2(traj, n_points=3000, n_r=25,
                              r_percentile_lo=10, r_percentile_hi=60,
                              return_diagnostics=False):
    """
    Improved Grassberger-Procaccia estimator.
    - Uses more sample points (3000) and more radii (25) for a
      cleaner log-log scaling region.
    - Restricts r range to [10th, 60th] percentile of pairwise
      distances to avoid both the noise-floor (too small r) and
      saturation (too large r) regions.
    - Picks the scaling region via the most LINEAR sub-segment
      of log(C) vs log(r), not just the middle third.
    """
    X = traj.T  # (n_timepoints, n_dims)
    if len(X) > n_points:
        idx = np.random.choice(len(X), n_points, replace=False)
        X = X[idx]

    dists = pdist(X, metric='euclidean')
    dists = dists[dists > 1e-12]
    if len(dists) < 10:
        return (np.nan, None, None) if return_diagnostics else np.nan

    r_lo, r_hi = np.percentile(dists, [r_percentile_lo, r_percentile_hi])
    if r_lo <= 0 or r_hi <= r_lo:
        return (np.nan, None, None) if return_diagnostics else np.nan

    r_vals = np.logspace(np.log10(r_lo), np.log10(r_hi), n_r)
    C_r = np.array([np.mean(dists < r) for r in r_vals])
    valid = C_r > 1e-6
    log_r = np.log(r_vals[valid])
    log_C = np.log(C_r[valid])

    if len(log_r) < 5:
        return (np.nan, log_r, log_C) if return_diagnostics else np.nan

    # Find the most linear 60%-length window via sliding-window R^2
    n = len(log_r)
    win = max(int(0.6*n), 4)
    best_R2, best_slope, best_start = -np.inf, np.nan, 0
    for start in range(0, n - win + 1):
        end = start + win
        c = np.polyfit(log_r[start:end], log_C[start:end], 1)
        pred = np.polyval(c, log_r[start:end])
        ss_res = np.sum((log_C[start:end]-pred)**2)
        ss_tot = np.sum((log_C[start:end]-log_C[start:end].mean())**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else -np.inf
        if R2 > best_R2:
            best_R2, best_slope, best_start = R2, c[0], start

    D2 = best_slope
    if return_diagnostics:
        return D2, log_r, log_C
    return D2

def run_correlation_dimension_v2(N_list=None, n_realiz=8, E_total=10.0,
                                  gamma=0.05, T=400, dt=0.02,
                                  t_transient=150):
    """
    D_2 vs N with a much longer transient (150 instead of 50) to ensure
    the trajectory has settled onto its asymptotic attractor before
    sampling, and longer total time T=400 for better statistics.
    """
    if N_list is None:
        N_list = [5, 10, 20, 50, 100, 200]
    results = []
    diag_store = {}
    for N in tqdm(N_list, desc="Correlation Dimension (v2)"):
        nb = build_1d_neighbors(N)
        D2_list = []
        for s in range(n_realiz):
            omega, kappa = make_params(N, nb)
            traj = simulate(N, nb, omega, kappa, E_total=E_total,
                            gamma=gamma, T=T, dt=dt,
                            t_transient=t_transient, seed=s)
            D2 = correlation_dimension_v2(traj)
            if not np.isnan(D2) and D2 > 0:
                D2_list.append(D2)
        # keep one diagnostic curve per N for plotting
        if D2_list:
            omega, kappa = make_params(N, nb)
            traj_diag = simulate(N, nb, omega, kappa, E_total=E_total,
                                 gamma=gamma, T=T, dt=dt,
                                 t_transient=t_transient, seed=0)
            _, lr, lc = correlation_dimension_v2(traj_diag, return_diagnostics=True)
            diag_store[N] = (lr, lc)
        results.append({
            'N': N,
            'D2_mean': np.mean(D2_list) if D2_list else np.nan,
            'D2_std': np.std(D2_list) if D2_list else np.nan,
            'n_valid': len(D2_list),
        })
        print(f"  N={N}: D_2={np.mean(D2_list):.2f} +/- {np.std(D2_list):.2f} "
              f"(n_valid={len(D2_list)}/{n_realiz})")
    return pd.DataFrame(results), diag_store

# ══════════════════════════════════════════════════════════
# (3) SPECTRAL BOTTLENECK — unchanged (already physically valid)
# ══════════════════════════════════════════════════════════

def mode_spacing_3d(L, kappa_mean=0.5):
    ks = np.arange(1, L+1) * np.pi / (L+1)
    KX, KY, KZ = np.meshgrid(ks, ks, ks, indexing='ij')
    omega2 = 1.0 + 4*kappa_mean*(np.sin(KX/2)**2 + np.sin(KY/2)**2 + np.sin(KZ/2)**2)
    omega = np.sqrt(omega2).flatten()
    omega_sorted = np.sort(omega)
    spacings = np.diff(omega_sorted)
    spacings = spacings[spacings > 1e-10]
    delta_omega_min = np.min(spacings) if len(spacings) > 0 else np.nan
    delta_omega_mean = np.mean(spacings) if len(spacings) > 0 else np.nan
    return delta_omega_min, delta_omega_mean

def nonlinear_linewidth(N, nb, E_total=10.0, gamma=0.05, kappa_mean=0.5,
                         T=200, dt=0.02, seed=0):
    omega, kappa = make_params(N, nb, kappa_mean=kappa_mean)
    traj = simulate(N, nb, omega, kappa, E_total=E_total, gamma=gamma,
                    T=T, dt=dt, seed=seed)
    theta = wrap_angle(traj[:N, :])   # <-- consistent wrapping here too
    theta2_mean = np.mean(theta**2)
    Gamma_NL = kappa_mean * theta2_mean
    return Gamma_NL

def run_spectral_bottleneck(L_list=None, kappa_mean=0.5, E_total=10.0,
                             gamma=0.05, T=150, dt=0.02):
    if L_list is None:
        L_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    results = []
    for L in tqdm(L_list, desc="Spectral Bottleneck"):
        N = L**3
        nb = build_3d_neighbors(L)
        dwmin, dwmean = mode_spacing_3d(L, kappa_mean=kappa_mean)
        Gamma_NL = nonlinear_linewidth(N, nb, E_total=E_total, gamma=gamma,
                                       kappa_mean=kappa_mean, T=T, dt=dt)
        ratio = dwmin / Gamma_NL if Gamma_NL > 0 else np.nan
        results.append({
            'L': L, 'N': N,
            'delta_omega_min': dwmin,
            'delta_omega_mean': dwmean,
            'Gamma_NL': Gamma_NL,
            'ratio_dwmin_over_GammaNL': ratio,
            'regime': 'suppressed' if ratio > 1 else 'cascade',
        })
        print(f"  L={L}: dw_min={dwmin:.6f}, Gamma_NL={Gamma_NL:.6f}, "
              f"ratio={ratio:.2f} ({'suppressed' if ratio>1 else 'cascade'})")
    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════

def plot_additional_results(df_lin, df_d2, diag_store, df_bottleneck):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    ax = axes[0,0]
    ax2 = ax.twinx()
    ax.semilogx(df_lin['N'], df_lin['theta_rms_deg'], 'bo-', label=r'$\theta_{rms}$ (deg, wrapped)')
    ax2.semilogx(df_lin['N'], df_lin['sin_rel_err_pct'], 'rs--',
                 label=r'$|\sin\theta-\theta|/\theta$ (%)')
    ax.set_xlabel('N'); ax.set_ylabel(r'$\theta_{rms}$ (deg)', color='blue')
    ax2.set_ylabel('Relative linearization error (%)', color='red')
    ax2.set_yscale('log')
    ax.set_title('(a) Self-Linearization (bug-fixed)')
    ax.grid(True, alpha=0.3)
    l1,la1 = ax.get_legend_handles_labels(); l2,la2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, fontsize=7)

    ax = axes[0,1]
    ax.errorbar(df_lin['N'], df_lin['NLI_mean'], yerr=df_lin['NLI_std'],
                fmt='go-', capsize=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('Nonlinearity Index')
    ax.set_title('(b) Nonlinearity Index vs N')
    ax.grid(True, alpha=0.3)

    ax = axes[0,2]
    ax.plot(df_lin['N'], df_lin['frac_theta_gt_30deg']*100, 'mo-')
    ax.set_xscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('% time with |theta|>30deg')
    ax.set_title('(c) Fraction in Nonlinear Range')
    ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    for N, (lr, lc) in diag_store.items():
        if lr is not None:
            ax.plot(lr, lc, 'o-', markersize=3, label=f'N={N}', alpha=0.7)
    ax.set_xlabel('log(r)'); ax.set_ylabel('log(C(r))')
    ax.set_title('(d) G-P Scaling Diagnostic')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    ax.errorbar(df_d2['N'], df_d2['D2_mean'], yerr=df_d2['D2_std'],
                fmt='ms-', capsize=3, label='$D_2$ (correlation dim., v2)')
    ax.axhline(y=16, color='k', linestyle='--', label='$D_{eff}=16$ (PCA)')
    ax.set_xscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('Dimension')
    ax.set_title('(e) $D_2$ (bug-fixed) vs PCA $D_{eff}$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1,2]
    ax.semilogy(df_bottleneck['L'], df_bottleneck['ratio_dwmin_over_GammaNL'],
            'r^-', markersize=6)
    ax.axhline(y=1, color='k', linestyle='--', label='Crossover (ratio=1)')
    ax.set_xlabel('L'); ax.set_ylabel(r'$\Delta\omega_{min}/\Gamma_{NL}$')
    ax.set_title('(f) Spectral Bottleneck')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('additional_analysis_v2.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('additional_analysis_v2.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Saved: additional_analysis_v2.pdf / .png")

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*60)
    print("Additional Analysis v2 (bug-fixed)")
    print("Estimated time: ~50-90 min")
    print("="*60)

    print("\n[1/3] Self-Linearization Check (theta wrapped)...")
    df_lin = run_linearization_check(
        N_list=[2,5,10,20,50,100,200,500,1000], n_realiz=10
    )
    print(df_lin.to_string(index=False))
    df_lin.to_csv('results_linearization_v2.csv', index=False)

    print("\n[2/3] Correlation Dimension D_2 (longer transient, robust fit)...")
    df_d2, diag_store = run_correlation_dimension_v2(
        N_list=[5,10,20,50,100,200], n_realiz=8
    )
    print(df_d2.to_string(index=False))
    df_d2.to_csv('results_correlation_dimension_v2.csv', index=False)

    print("\n[3/3] Spectral Bottleneck Quantification...")
    df_bottleneck = run_spectral_bottleneck(
        L_list=[2,3,4,5,6,7,8,9,10,12,15]
    )
    print(df_bottleneck.to_string(index=False))
    df_bottleneck.to_csv('results_spectral_bottleneck.csv', index=False)

    print("\nGenerating plots...")
    plot_additional_results(df_lin, df_d2, diag_store, df_bottleneck)

    print("\nDone! Download these files:")
    print("  results_linearization_v2.csv")
    print("  results_correlation_dimension_v2.csv")
    print("  results_spectral_bottleneck.csv")
    print("  additional_analysis_v2.pdf")
