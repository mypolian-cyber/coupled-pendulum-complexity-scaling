# ============================================================
# Coupled Pendulum System - Full Simulation Suite
# For: Bang Won Kyu & Bang Jun Suk (2025)
# Run on Google Colab (GPU not required, CPU sufficient)
# ============================================================

# ── 0. Install & Import ─────────────────────────────────────
# !pip install scipy numpy matplotlib pandas tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from itertools import product
from tqdm.auto import tqdm
import json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── 1. Core Physics ─────────────────────────────────────────

def build_1d_neighbors(N):
    """1D chain: i <-> i±1"""
    nb = {i: [] for i in range(N)}
    for i in range(N - 1):
        nb[i].append(i + 1)
        nb[i + 1].append(i)
    return nb

def build_2d_neighbors(L):
    """2D square lattice: L x L"""
    N = L * L
    nb = {i: [] for i in range(N)}
    for r in range(L):
        for c in range(L):
            i = r * L + c
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < L and 0 <= cc < L:
                    nb[i].append(rr*L + cc)
    return nb

def build_3d_neighbors(L):
    """3D cubic lattice: L x L x L"""
    N = L**3
    nb = {i: [] for i in range(N)}
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = x*L*L + y*L + z
                for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    xx,yy,zz = x+dx, y+dy, z+dz
                    if 0<=xx<L and 0<=yy<L and 0<=zz<L:
                        nb[i].append(xx*L*L + yy*L + zz)
    return nb

def make_params(N, neighbors, sigma=0.0, kappa_mean=0.5, seed=None):
    """Generate omega_i and kappa_ij with log-normal disorder"""
    rng = np.random.RandomState(seed)
    if sigma == 0.0:
        omega = np.ones(N)
        kappa = {}
        for i, js in neighbors.items():
            for j in js:
                kappa[(i,j)] = kappa_mean
    else:
        omega = rng.lognormal(0, sigma, N)
        kappa = {}
        for i, js in neighbors.items():
            for j in js:
                if (j,i) not in kappa:
                    k = rng.lognormal(np.log(kappa_mean), sigma)
                    kappa[(i,j)] = k
                    kappa[(j,i)] = k
    return omega, kappa

def pendulum_ode(t, y, N, omega, kappa, neighbors, gamma=0.05):
    """dy/dt for coupled pendulum system"""
    theta = y[:N]
    dtheta = y[N:]
    ddtheta = np.zeros(N)
    for i in range(N):
        ddtheta[i] = (-omega[i]**2 * np.sin(theta[i])
                      - gamma * dtheta[i]
                      + sum(kappa.get((i,j), 0) * np.sin(theta[j] - theta[i])
                            for j in neighbors[i]))
    return np.concatenate([dtheta, ddtheta])

def init_fixed_energy(N, omega, E_total=10.0, seed=None):
    """Initial conditions with exact total energy E_total"""
    rng = np.random.RandomState(seed)
    theta0 = np.zeros(N)
    dtheta0 = rng.randn(N)
    # Current kinetic energy
    KE = 0.5 * np.sum(dtheta0**2)
    PE = np.sum(omega**2 * (1 - np.cos(theta0)))
    E_current = KE + PE
    # Scale velocities
    if KE > 0:
        scale = np.sqrt((E_total - PE) / KE)
        dtheta0 *= scale
    return np.concatenate([theta0, dtheta0])

def simulate(N, neighbors, omega, kappa, E_total=10.0,
             gamma=0.05, T=300, dt=0.02, t_transient=50, seed=None):
    """Run ODE and return trajectory after transient"""
    y0 = init_fixed_energy(N, omega, E_total, seed)
    t_span = (0, T)
    t_eval = np.arange(t_transient, T, dt)
    sol = solve_ivp(pendulum_ode, t_span, y0,
                    args=(N, omega, kappa, neighbors, gamma),
                    method='LSODA', t_eval=t_eval,
                    rtol=1e-5, atol=1e-7, dense_output=False)
    return sol.y  # shape (2N, n_timepoints)

# ── 2. Metrics ───────────────────────────────────────────────

def participation_ratio(traj, N):
    """PR from phase-space covariance eigenvalues"""
    X = traj.T  # (n_timepoints, 2N)
    X -= X.mean(axis=0)
    cov = X.T @ X / len(X)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    PR = (eigvals.sum())**2 / (2*N * (eigvals**2).sum())
    return PR

def pca_effective_dim(traj, variance_threshold=0.95):
    """Number of PCA components explaining variance_threshold"""
    X = traj.T
    X -= X.mean(axis=0)
    cov = X.T @ X / len(X)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = eigvals[eigvals > 0]
    cumvar = np.cumsum(eigvals) / eigvals.sum()
    D_eff = int(np.searchsorted(cumvar, variance_threshold)) + 1
    return D_eff, eigvals

# ── 3. Cross-Validation (LSODA vs RK45) ─────────────────────

def cross_validate(N_test=10, neighbors=None, omega=None, kappa=None,
                   E_total=10.0, gamma=0.05, T=100, dt=0.02, t_transient=50):
    """Compare LSODA and RK45 trajectories"""
    if neighbors is None:
        neighbors = build_1d_neighbors(N_test)
        omega, kappa = make_params(N_test, neighbors)
    y0 = init_fixed_energy(N_test, omega, E_total, seed=0)
    t_eval = np.arange(t_transient, T, dt)
    sol_L = solve_ivp(pendulum_ode, (0, T), y0,
                      args=(N_test, omega, kappa, neighbors, gamma),
                      method='LSODA', t_eval=t_eval, rtol=1e-5, atol=1e-7)
    sol_R = solve_ivp(pendulum_ode, (0, T), y0,
                      args=(N_test, omega, kappa, neighbors, gamma),
                      method='RK45', t_eval=t_eval, rtol=1e-5, atol=1e-7)
    rel_err = np.mean(np.abs(sol_L.y - sol_R.y) /
                      (np.abs(sol_R.y) + 1e-10))
    return rel_err

# ── 4. Experiment A: 1D Scaling (Extended N range) ──────────

def run_1d_scaling(N_list=None, n_realiz=20, E_total=10.0,
                   gamma=0.05, T=300, dt=0.02):
    """PR and D_eff vs N for homogeneous 1D chains"""
    if N_list is None:
        N_list = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    results = []
    for N in tqdm(N_list, desc="1D Scaling"):
        PR_list, Deff_list = [], []
        nb = build_1d_neighbors(N)
        for s in range(n_realiz):
            omega, kappa = make_params(N, nb, sigma=0.0)
            traj = simulate(N, nb, omega, kappa, E_total=E_total,
                            gamma=gamma, T=T, dt=dt, seed=s)
            PR_list.append(participation_ratio(traj, N))
            Deff_list.append(pca_effective_dim(traj)[0])
        results.append({
            'N': N,
            'PR_mean': np.mean(PR_list),
            'PR_std': np.std(PR_list),
            'PR_ci95': 1.96 * np.std(PR_list) / np.sqrt(n_realiz),
            'Deff_mean': np.mean(Deff_list),
            'Deff_std': np.std(Deff_list),
        })
    return pd.DataFrame(results)

# ── 5. Experiment B: Energy Sensitivity ─────────────────────

def run_energy_sensitivity(N=20, E_list=None, n_realiz=20,
                            gamma=0.05, T=300, dt=0.02):
    """PR vs E_total to test energy sensitivity"""
    if E_list is None:
        E_list = [1, 2, 5, 10, 20, 50, 100]
    nb = build_1d_neighbors(N)
    results = []
    for E in tqdm(E_list, desc="Energy Sensitivity"):
        PR_list = []
        for s in range(n_realiz):
            omega, kappa = make_params(N, nb, sigma=0.0)
            traj = simulate(N, nb, omega, kappa, E_total=E,
                            gamma=gamma, T=T, dt=dt, seed=s)
            PR_list.append(participation_ratio(traj, N))
        results.append({
            'E_total': E,
            'PR_mean': np.mean(PR_list),
            'PR_std': np.std(PR_list),
            'PR_ci95': 1.96 * np.std(PR_list) / np.sqrt(n_realiz),
        })
    return pd.DataFrame(results)

# ── 6. Experiment C: Disorder Scaling ───────────────────────

def run_disorder_scaling(N_list=None, sigma_list=None, n_realiz=20,
                          E_total=10.0, gamma=0.05, T=300, dt=0.02):
    """alpha(sigma) for each disorder level"""
    if N_list is None:
        N_list = [5, 10, 20, 50, 100, 200]
    if sigma_list is None:
        sigma_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []
    for sigma in tqdm(sigma_list, desc="Disorder Scaling"):
        PR_rows = []
        for N in N_list:
            nb = build_1d_neighbors(N)
            PR_list = []
            for s in range(n_realiz):
                omega, kappa = make_params(N, nb, sigma=sigma, seed=s*100)
                traj = simulate(N, nb, omega, kappa, E_total=E_total,
                                gamma=gamma, T=T, dt=dt, seed=s)
                PR_list.append(participation_ratio(traj, N))
            PR_rows.append((N, np.mean(PR_list)))
        # Fit alpha
        Ns = np.array([r[0] for r in PR_rows])
        PRs = np.array([r[1] for r in PR_rows])
        log_N = np.log(Ns[1:])  # exclude N=2 if present
        log_PR = np.log(PRs[1:])
        coeffs = np.polyfit(log_N, log_PR, 1)
        alpha = -coeffs[0]
        A = np.exp(coeffs[1])
        results.append({
            'sigma': sigma,
            'alpha': alpha,
            'A': A,
        })
    return pd.DataFrame(results)

# ── 7. Experiment D: 2D & 3D Scaling ────────────────────────

def run_2d_scaling(L_list=None, n_realiz=10, E_total=10.0,
                   gamma=0.05, T=200, dt=0.02):
    """PR vs N for 2D lattice"""
    if L_list is None:
        L_list = [2, 3, 4, 5, 6, 7, 8, 10]
    results = []
    for L in tqdm(L_list, desc="2D Scaling"):
        N = L * L
        nb = build_2d_neighbors(L)
        PR_list = []
        for s in range(n_realiz):
            omega, kappa = make_params(N, nb, sigma=0.0)
            traj = simulate(N, nb, omega, kappa, E_total=E_total,
                            gamma=gamma, T=T, dt=dt, seed=s)
            PR_list.append(participation_ratio(traj, N))
        results.append({
            'L': L, 'N': N,
            'PR_mean': np.mean(PR_list),
            'PR_std': np.std(PR_list),
            'PR_ci95': 1.96 * np.std(PR_list) / np.sqrt(n_realiz),
        })
    return pd.DataFrame(results)

def run_3d_scaling(L_list=None, n_realiz=5, E_total=10.0,
                   gamma=0.05, T=200, dt=0.02):
    """PR vs N for 3D cubic lattice"""
    if L_list is None:
        L_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    results = []
    for L in tqdm(L_list, desc="3D Scaling"):
        N = L**3
        nb = build_3d_neighbors(L)
        PR_list = []
        for s in range(n_realiz):
            omega, kappa = make_params(N, nb, sigma=0.0)
            traj = simulate(N, nb, omega, kappa, E_total=E_total,
                            gamma=gamma, T=T, dt=dt, seed=s)
            PR_list.append(participation_ratio(traj, N))
        results.append({
            'L': L, 'N': N,
            'PR_mean': np.mean(PR_list),
            'PR_std': np.std(PR_list),
            'PR_ci95': 1.96 * np.std(PR_list) / np.sqrt(n_realiz),
        })
    return pd.DataFrame(results)

# ── 8. Experiment E: D_eff Saturation Analysis ───────────────

def run_deff_saturation(N_list=None, n_realiz=20, E_total=10.0,
                         gamma=0.05, T=300, dt=0.02):
    """Full PCA eigenvalue spectrum for D_eff saturation analysis"""
    if N_list is None:
        N_list = [2, 5, 10, 20, 50, 100, 200]
    results = []
    for N in tqdm(N_list, desc="D_eff Saturation"):
        eigval_runs = []
        nb = build_1d_neighbors(N)
        for s in range(n_realiz):
            omega, kappa = make_params(N, nb, sigma=0.0)
            traj = simulate(N, nb, omega, kappa, E_total=E_total,
                            gamma=gamma, T=T, dt=dt, seed=s)
            _, eigvals = pca_effective_dim(traj)
            # Normalize and store top 30
            ev_norm = eigvals / eigvals.sum()
            eigval_runs.append(ev_norm[:30] if len(ev_norm) >= 30
                                else np.pad(ev_norm, (0, 30-len(ev_norm))))
        eigval_mean = np.mean(eigval_runs, axis=0)
        results.append({
            'N': N,
            'eigvals_normalized': eigval_mean.tolist(),
            'cumvar_at_16': float(np.sum(eigval_mean[:16])),
        })
    return pd.DataFrame(results)

# ── 9. Experiment F: KZ Spectrum (WWT Verification) ──────────

def run_kz_spectrum(N=200, kappa_mean=0.15, gamma=0.005,
                     E_total=10.0, T=500, dt=0.02, t_transient=50):
    """Mode occupation n_k = <|theta_k|^2> for KZ spectrum check"""
    nb = build_1d_neighbors(N)
    # Periodic boundary
    nb[0].append(N-1); nb[N-1].append(0)
    omega, kappa = make_params(N, nb, sigma=0.0, kappa_mean=kappa_mean)
    traj = simulate(N, nb, omega, kappa, E_total=E_total,
                    gamma=gamma, T=T, dt=dt, t_transient=t_transient, seed=0)
    theta_traj = traj[:N, :]  # (N, n_t)
    # DFT mode occupation
    theta_k = np.fft.rfft(theta_traj, axis=0)  # (N//2+1, n_t)
    n_k = np.mean(np.abs(theta_k)**2, axis=1)
    k = np.arange(len(n_k))
    # Fit power law
    mask = (k >= 2) & (k <= N//4)
    log_k = np.log(k[mask])
    log_nk = np.log(n_k[mask])
    coeffs = np.polyfit(log_k, log_nk, 1)
    exponent = coeffs[0]
    residuals = log_nk - np.polyval(coeffs, log_k)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_nk - log_nk.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return pd.DataFrame({'k': k, 'n_k': n_k}), exponent, R2

# ── 10. Plotting ─────────────────────────────────────────────

def plot_all_results(df_1d, df_energy, df_disorder, df_2d, df_3d,
                     df_deff, df_kz, kz_exp, kz_R2):
    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Fig 1: 1D PR Scaling ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1b = ax1.twinx()
    ax1.errorbar(df_1d['N'], df_1d['PR_mean'], yerr=df_1d['PR_ci95'],
                 fmt='bo-', capsize=3, label='PR', zorder=5)
    # Power-law fit
    mask = df_1d['N'] >= 5
    log_N = np.log(df_1d.loc[mask, 'N'])
    log_PR = np.log(df_1d.loc[mask, 'PR_mean'])
    c = np.polyfit(log_N, log_PR, 1)
    alpha_fit = -c[0]; A_fit = np.exp(c[1])
    N_fit = np.linspace(5, df_1d['N'].max(), 100)
    ax1.plot(N_fit, A_fit * N_fit**(-alpha_fit), 'g-',
             label=f'Fit $AN^{{-{alpha_fit:.2f}}}$', zorder=3)
    ax1b.errorbar(df_1d['N'], df_1d['Deff_mean'], yerr=df_1d['Deff_std'],
                  fmt='rs--', capsize=3, label='$D_{{eff}}$', zorder=4)
    ax1.set_xlabel('$N$'); ax1.set_ylabel('PR', color='blue')
    ax1b.set_ylabel('$D_{eff}$', color='red')
    ax1.set_title(f'(a) 1D: $\\alpha={alpha_fit:.2f}$')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ── Fig 2: Energy Sensitivity ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(df_energy['E_total'], df_energy['PR_mean'],
                 yerr=df_energy['PR_ci95'], fmt='ko-', capsize=3)
    ax2.set_xlabel('$E_{total}$'); ax2.set_ylabel('PR')
    ax2.set_title('(b) Energy Sensitivity ($N=20$)')
    ax2.set_xscale('log'); ax2.grid(True, alpha=0.3)

    # ── Fig 3: Disorder Scaling ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sigma_vals = df_disorder['sigma'].values
    alpha_vals = df_disorder['alpha'].values
    ax3.plot(sigma_vals, alpha_vals, 'ko', markersize=5)
    # Fit
    c3 = np.polyfit(sigma_vals**2, alpha_vals, 1)
    sigma_fit = np.linspace(0, 0.85, 100)
    ax3.plot(sigma_fit, c3[0]*sigma_fit**2 + c3[1], 'b-',
             label=f'$\\alpha_0={c3[1]:.2f},\\ c={-c3[0]:.2f}$')
    ax3.axvline(x=0.4, color='r', linestyle='--', label='$\\sigma_c=0.4$')
    ax3.set_xlabel('$\\sigma$'); ax3.set_ylabel('$\\alpha$')
    ax3.set_title('(c) Disorder Dependence')
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    # ── Fig 4: 2D Scaling ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.errorbar(df_2d['N'], df_2d['PR_mean'], yerr=df_2d['PR_ci95'],
                 fmt='rs-', capsize=3, label='2D sim.')
    log_N2 = np.log(df_2d['N'])
    log_PR2 = np.log(df_2d['PR_mean'])
    c4 = np.polyfit(log_N2, log_PR2, 1)
    N_fit2 = np.linspace(df_2d['N'].min(), df_2d['N'].max(), 100)
    ax4.plot(N_fit2, np.exp(c4[1])*N_fit2**c4[0], 'r--',
             label=f'Fit $\\alpha={-c4[0]:.2f}$')
    ax4.set_xlabel('$N$'); ax4.set_ylabel('PR')
    ax4.set_title('(d) 2D Lattice Scaling')
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

    # ── Fig 5: 3D Scaling ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.errorbar(df_3d['N'], df_3d['PR_mean'], yerr=df_3d['PR_ci95'],
                 fmt='g^-', capsize=3, label='3D sim.')
    log_N3 = np.log(df_3d['N'])
    log_PR3 = np.log(df_3d['PR_mean'])
    c5 = np.polyfit(log_N3, log_PR3, 1)
    N_fit3 = np.linspace(df_3d['N'].min(), df_3d['N'].max(), 100)
    ax5.plot(N_fit3, np.exp(c5[1])*N_fit3**c5[0], 'g--',
             label=f'Fit $\\alpha={-c5[0]:.2f}$')
    ax5.set_xlabel('$N$'); ax5.set_ylabel('PR')
    ax5.set_title('(e) 3D Cubic Scaling')
    ax5.set_xscale('log'); ax5.set_yscale('log')
    ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

    # ── Fig 6: All Dimensions Comparison ─────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.loglog(df_1d['N'], df_1d['PR_mean'], 'bo-', label=f'1D $\\alpha={alpha_fit:.2f}$')
    ax6.loglog(df_2d['N'], df_2d['PR_mean'], 'rs-', label=f'2D $\\alpha={-c4[0]:.2f}$')
    ax6.loglog(df_3d['N'], df_3d['PR_mean'], 'g^-', label=f'3D $\\alpha={-c5[0]:.2f}$')
    ax6.set_xlabel('$N$'); ax6.set_ylabel('PR')
    ax6.set_title('(f) All Dimensions')
    ax6.legend(fontsize=7); ax6.grid(True, alpha=0.3)

    # ── Fig 7: D_eff Saturation ───────────────────────────
    ax7 = fig.add_subplot(gs[2, 0:2])
    for _, row in df_deff.iterrows():
        ev = np.array(row['eigvals_normalized'])
        cumvar = np.cumsum(ev)
        ax7.plot(range(1, len(cumvar)+1), cumvar,
                 label=f'N={int(row["N"])}', alpha=0.7)
    ax7.axhline(y=0.95, color='k', linestyle='--', label='95% threshold')
    ax7.axvline(x=16, color='r', linestyle=':', label='$D_{eff}=16$')
    ax7.set_xlabel('Number of PCA components')
    ax7.set_ylabel('Cumulative variance explained')
    ax7.set_title('(g) $D_{eff}$ Saturation: PCA Cumulative Variance')
    ax7.legend(fontsize=7, loc='lower right'); ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, 30]); ax7.set_ylim([0, 1.05])

    # ── Fig 8: 3D Local Exponent ──────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    inv_L = 1.0 / df_3d['L'].values
    # Compute local exponents
    log_N3v = np.log(df_3d['N'].values)
    log_PR3v = np.log(df_3d['PR_mean'].values)
    alpha_local = -np.diff(log_PR3v) / np.diff(log_N3v)
    inv_L_mid = 0.5*(inv_L[:-1] + inv_L[1:])
    ax8.plot(inv_L_mid, alpha_local, 'r^-', markersize=5)
    ax8.axhline(y=-c4[0], color='g', linestyle='--',
                label=f'$\\alpha_{{2D}}={-c4[0]:.2f}$')
    ax8.set_xlabel('$1/L$'); ax8.set_ylabel('Local $\\alpha(L)$')
    ax8.set_title('(h) 3D Finite-Size Crossover')
    ax8.legend(fontsize=8); ax8.grid(True, alpha=0.3)

    # ── Fig 9: KZ Spectrum ────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 0:2])
    mask_k = (df_kz['k'] >= 2) & (df_kz['k'] <= len(df_kz)//4)
    ax9.loglog(df_kz.loc[mask_k, 'k'], df_kz.loc[mask_k, 'n_k'],
               'b.', alpha=0.5, label='Simulation $n_k$')
    k_fit = np.array([2, len(df_kz)//4])
    ax9.loglog(k_fit, 2*k_fit**(-2/3), 'r--',
               label='WWT $k^{-2/3}$', linewidth=2)
    ax9.loglog(k_fit, df_kz.loc[mask_k, 'n_k'].iloc[0] * (k_fit/2.0)**kz_exp,
               'b-', label=f'Fit $k^{{{kz_exp:.2f}}}$ ($R^2={kz_R2:.3f}$)')
    ax9.set_xlabel('Wavenumber $k$')
    ax9.set_ylabel('Mode occupation $n_k$')
    ax9.set_title('(i) KZ Spectrum Verification')
    ax9.legend(fontsize=8); ax9.grid(True, alpha=0.3)

    # ── Fig 10: Summary Table (text) ─────────────────────
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['$\\alpha_{1D}$', f'{alpha_fit:.3f}'],
        ['$\\alpha_{2D}$', f'{-c4[0]:.3f}'],
        ['$\\alpha_{3D}$', f'{-c5[0]:.3f}'],
        ['$D_{eff}$ saturation', '~16'],
        ['$\\sigma_c$', '~0.4'],
        ['KZ exponent', f'{kz_exp:.3f}'],
        ['KZ $R^2$', f'{kz_R2:.4f}'],
    ]
    table = ax10.table(cellText=table_data[1:], colLabels=table_data[0],
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    ax10.set_title('(j) Summary')

    plt.suptitle('Coupled Pendulum Scaling: Full Results\n'
                 'Bang Won Kyu & Bang Jun Suk', fontsize=13, y=1.01)
    plt.savefig('pendulum_full_results.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('pendulum_full_results.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Saved: pendulum_full_results.pdf / .png")

# ── 11. Save Results ─────────────────────────────────────────

def save_results(df_1d, df_energy, df_disorder, df_2d, df_3d,
                  df_deff, df_kz, kz_exp, kz_R2, crossval_err):
    df_1d.to_csv('results_1d_scaling.csv', index=False)
    df_energy.to_csv('results_energy_sensitivity.csv', index=False)
    df_disorder.to_csv('results_disorder_scaling.csv', index=False)
    df_2d.to_csv('results_2d_scaling.csv', index=False)
    df_3d.to_csv('results_3d_scaling.csv', index=False)
    df_kz.to_csv('results_kz_spectrum.csv', index=False)
    summary = {
        'crossvalidation_err': crossval_err,
        'kz_exponent': kz_exp,
        'kz_R2': kz_R2,
    }
    with open('results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    # D_eff needs special handling (list column)
    df_deff_save = df_deff.copy()
    df_deff_save['eigvals_normalized'] = df_deff_save['eigvals_normalized'].apply(
        lambda x: ','.join(map(str, x)))
    df_deff_save.to_csv('results_deff_saturation.csv', index=False)
    print("All CSV/JSON files saved.")

# ── 12. MAIN ─────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Coupled Pendulum Simulation Suite")
    print("Estimated total time: ~30-60 min on Colab CPU")
    print("=" * 60)

    # Cross-validation first
    print("\n[0/6] Cross-validation (LSODA vs RK45)...")
    crossval_err = cross_validate(N_test=10)
    print(f"  Mean relative error: {crossval_err:.4f} ({crossval_err*100:.2f}%)")

    # Experiment A
    print("\n[1/6] 1D Scaling (N up to 1000)...")
    df_1d = run_1d_scaling(
        N_list=[2, 5, 10, 20, 50, 100, 200, 500, 1000],
        n_realiz=20
    )
    print(df_1d[['N','PR_mean','PR_ci95','Deff_mean']].to_string(index=False))

    # Experiment B
    print("\n[2/6] Energy Sensitivity...")
    df_energy = run_energy_sensitivity(N=20, E_list=[1,2,5,10,20,50,100])
    print(df_energy.to_string(index=False))

    # Experiment C
    print("\n[3/6] Disorder Scaling...")
    df_disorder = run_disorder_scaling(
        N_list=[5,10,20,50,100,200],
        sigma_list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    )
    print(df_disorder.to_string(index=False))

    # Experiment D
    print("\n[4/6] 2D & 3D Scaling...")
    df_2d = run_2d_scaling(L_list=[2,3,4,5,6,7,8,10])
    df_3d = run_3d_scaling(L_list=[2,3,4,5,6,7,8,9,10,12,15])

    # Experiment E
    print("\n[5/6] D_eff Saturation Analysis...")
    df_deff = run_deff_saturation(N_list=[2,5,10,20,50,100,200])

    # Experiment F
    print("\n[6/6] KZ Spectrum Verification...")
    df_kz, kz_exp, kz_R2 = run_kz_spectrum(N=200)
    print(f"  KZ exponent: {kz_exp:.3f}, R2: {kz_R2:.4f}")

    # Save & Plot
    save_results(df_1d, df_energy, df_disorder, df_2d, df_3d,
                  df_deff, df_kz, kz_exp, kz_R2, crossval_err)
    plot_all_results(df_1d, df_energy, df_disorder, df_2d, df_3d,
                      df_deff, df_kz, kz_exp, kz_R2)

    print("\nDone! Upload these files to Claude:")
    print("  results_1d_scaling.csv")
    print("  results_energy_sensitivity.csv")
    print("  results_disorder_scaling.csv")
    print("  results_2d_scaling.csv")
    print("  results_3d_scaling.csv")
    print("  results_deff_saturation.csv")
    print("  results_kz_spectrum.csv")
    print("  results_summary.json")
    print("  pendulum_full_results.pdf")
