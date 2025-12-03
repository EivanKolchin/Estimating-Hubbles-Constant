# -*- coding: utf-8 -*-
"""
@author: Eivan Kolchin-Rivas
"""
#importig the few libraries we can actually use for this...
import numpy as np
import matplotlib.pyplot as plt

# ======= HELPER FUNCTIONS ======

def weighted_linear_fit(x, y, sigma_y):
    """ function used to fit PL relation + to fit shift form
    computes a, b, chi2, dof, variance & standard deviation"""
    x, y, sigma_y = np.asanyarray(x), np.asanyarray(y), np.asanyarray(sigma_y)
    w = 1.0 / sigma_y ** 2
    S, Sx, Sy, Sxx, Sxy = np.sum(w), np.sum(w * x), np.sum(w * y), np.sum(w * x ** 2), np.sum(w * x * y)
    delta = S * Sxx - Sx ** 2
    a, b = ((S*Sxy - Sx*Sy) / delta), ((Sxx*Sy - Sx*Sxy) / delta)
    y_model = a * x + b
    var_a, var_b, cov_ab = (S / delta), (Sxx / delta), (-Sx / delta)
    sigma_a, sigma_b = np.sqrt(var_a), np.sqrt(var_b)
    chi2 = np.sum((y - y_model) ** 2 / sigma_y ** 2)
    dof = len(x) - 2

    return a, b, sigma_a, sigma_b, cov_ab, chi2, dof


def constant_fit(values, sigmas, add_intrinsic=False):
    """ function used to combine: distance moduli for NGC4527, hubble constant estimates -> 1
        assumes error in y onlyu and gaussian"""
    values, sigmas = np.asanyarray(values), np.asanyarray(sigmas)
    n = len(values)
    dof = n - 1
    w = 1.0 / sigmas ** 2
    mean = np.sum(w * values) / np.sum(w)
    chi2 = np.sum((values - mean) ** 2 / sigmas ** 2)
    chi2_red = chi2 / dof

    if not add_intrinsic:
        sigma_mean = np.sqrt(1.0 / np.sum(w))
        return mean, sigma_mean, chi2_red, 0.0

    def chi2_red_for_sigma_int(sigma_int):
        var_tot = sigmas ** 2 + sigma_int ** 2
        w_tot = 1.0 / var_tot
        mean_loc = np.sum(w_tot * values) / np.sum(w_tot)
        chi2_loc = np.sum((values - mean_loc) ** 2 / var_tot)
        return chi2_loc / dof, mean_loc

    max_sigma = 50.0
    sigma_grid = np.linspace(0.0, max_sigma, 1001)

    best_sigma_int = 0.0
    best_diff = 1e9
    best_mean = mean
    best_chi2_red = chi2_red

    for sigma_int in sigma_grid:
        chi2r_loc, mean_loc = chi2_red_for_sigma_int(sigma_int)
        diff = abs(chi2r_loc - 1.0)
        if diff < best_diff:
            best_diff, best_sigma_int, best_mean, best_chi2_red = (
                diff, sigma_int, mean_loc, chi2r_loc
            )
    if best_sigma_int >= max_sigma*0.99:
        print(f'intrinsic dispersioon hit grid limit {max_sigma} :/\n')

    var_tot = sigmas ** 2 + best_sigma_int ** 2
    w_tot = 1.0 / var_tot
    sigma_mean = np.sqrt(1.0 / np.sum(w_tot))

    return best_mean, sigma_mean, best_chi2_red, best_sigma_int

# =============== STEP 1 ===============
def fit_pl_relation(mw_file):
    """milkyway cepheids -> calibrated period luminosity relation"""
    data = np.loadtxt(
        mw_file,
        comments='#',
        dtype=[('f0', 'U32'), # f0 is string
               ('f1', float),
               ('f2', float),
               ('f3', float),
               ('f4', float),
               ('f5', float),
               ('f6', float)]
    )

    names = data['f0']
    parallax_mas = data['f1']
    sigma_parallax_mas = data['f2']
    period_days = data['f3']
    m_app = data['f4']
    A = data['f5']
    sigma_A = data['f6']

    # parallax -> distance in pasrsecs
    d_pc = 1000.0 / parallax_mas
    sigma_d_pc = 1000.0 * sigma_parallax_mas / parallax_mas ** 2

    # abs magnitude: (M = m-5log10(pc_d)+5-A)
    logd = np.log10(d_pc)
    M_abs = m_app - 5.0 * logd + 5.0 - A

    # Eror propagation for absolute magnitude (M)
    dM_dd = -5.0 / (np.log(10.0) * d_pc)

    # variance in M
    sigma_M = np.sqrt((dM_dd * sigma_d_pc) ** 2 + sigma_A ** 2)

    # x axis for PL relation: log10(P)
    logP = np.log10(period_days)

    # first shift: unshifted pl relation M= alpha*logP +beta
    alpha, beta, sigma_alpha, sigma_beta, cov_ab, chi2, dof = weighted_linear_fit(
        logP, M_abs, sigma_M)
    chi2_red = chi2 / dof
    rho = cov_ab / (sigma_alpha * sigma_beta)

    # reducing correlation: shifted x: M = a(log10(P)-x0)+b
    x0 = np.average(logP, weights=1.0 / sigma_M ** 2)
    x_shift = logP - x0
    a_shift, b_shift, sigma_a_shift, sigma_b_shift, cov_ab_shift, chi2_s, dof_s = (
        weighted_linear_fit(x_shift, M_abs, sigma_M))
    rho_shift = cov_ab_shift / (sigma_a_shift * sigma_b_shift)

    # bundling all into "results"
    results = {
        'names': names,
        'logP': logP,
        'M_abs': M_abs,
        'sigma_M': sigma_M,
        'alpha': alpha,
        'beta': beta,
        'sigma_alpha': sigma_alpha,
        'sigma_beta': sigma_beta,
        'cov_ab': cov_ab,
        'rho': rho,
        'chi2': chi2,
        'chi2_red': chi2_red,
        'x0': x0,
        'alpha_shift': a_shift,
        'beta_shift': b_shift,
        'sigma_alpha_shift': sigma_a_shift,
        'sigma_beta_shift': sigma_b_shift,
        'rho_shift': rho_shift,
    }

    return results


# =============== STEP 2 ===============
# ESTIMATING THE DISTANCE OF NEARBY GALAXY
def distance_ngc4527(ngc_file, pl_results, A_MW=0.0682, add_intrinsic=True):
    """Estimates distance of nearby galaxy"""

    #Reading file data
    data = np.loadtxt(
        ngc_file,
        comments='#',
        dtype=[('f0', 'U32'),
               ('f1', float),
               ('f2', float)]
    )
    names = data['f0']
    logP_ngc = data['f1']
    m_ngc = data['f2']

    # getitng absolute magnitudes from PL relation
    a, b, sigma_a, sigma_b, x0 = (
        pl_results['alpha_shift'],
        pl_results['beta_shift'],
        pl_results['sigma_alpha_shift'],
        pl_results['sigma_beta_shift'],
        pl_results['x0'],
    )

    # for each cepheid Mz = a(log10(Pz)-x0)+b
    delta_x = logP_ngc - x0
    M_ngc = a * delta_x + b

    # Uncertainty on each cepheids absolute magnitude
    sigma_M_ngc = np.sqrt(delta_x ** 2 * sigma_a ** 2 + sigma_b ** 2)

    # distance modulus for each cepheid
    mu_i = m_ngc - A_MW - M_ngc
    sigma_mu_i = sigma_M_ngc

    #combining all mu_i -> 1 for galaxy
    mu_mean, sigma_mu_mean, chi2_red, sigma_int = constant_fit(
        mu_i, sigma_mu_i, add_intrinsic=add_intrinsic
    )

    # defining distance from distance modulus: mu = 5log10(d)-5 => d = 10^(mu+5)/5
    d_pc = 10 ** ((mu_mean + 5.0) / 5.0)

    # erorr propagation
    sigma_d_pc = (np.log(10.0) / 5.0) * d_pc * sigma_mu_mean

    # bundling into "results" again
    results = {
        'names': names,
        'logP_ngc': logP_ngc,
        'm_ngc': m_ngc,
        'mu_i': mu_i,
        'sigma_mu_i': sigma_mu_i,
        'mu_mean': mu_mean,
        'sigma_mu_mean': sigma_mu_mean,
        'chi2_red_mu': chi2_red,
        'sigma_int_mu': sigma_int,
        'd_pc': d_pc,
        'sigma_d_pc': sigma_d_pc,
        'd_Mpc': d_pc / 1.0e6,
        'sigma_d_Mpc': sigma_d_pc / 1.0e6,
    }

    return results


# =============== STEP 3 ===============
# FINDING HUBBLES CONSTANT / EXPANSION OF UNIVERSE
def hubbles_constant(ngc_distance_results, other_file, v_rec_ngc=1152.0):
    """ using distance & recession velocities for computing hubbles constant"""

    # reads data from other galaxies files
    other = np.loadtxt(
        other_file,
        comments='#',
        dtype=[('f0', 'U32'),
               ('f1', float),
               ('f2', float),
               ('f3', float)]
    )

    names_other = other['f0']
    v_other = other['f1'].astype(float)
    D_other = other['f2']
    sigma_D_other = other['f3']

    # obtaining NGC4527 distance from other function
    D_ngc, sigma_D_ngc = ngc_distance_results['d_Mpc'], ngc_distance_results['sigma_d_Mpc']

    # combining all galaxies
    names_all = list(names_other) + ['NGC4527']
    v_all, D_all = np.append(v_other, v_rec_ngc), np.append(D_other, D_ngc)
    sigma_D_all = np.append(sigma_D_other, sigma_D_ngc)

    # # using hubble constant equation: hubbles constant = velocity of recession / distance
    H0_i = v_all / D_all

    # error propagation, assuming sigma_v~=0 relative to simga_D
    sigma_H0_i = (v_all / D_all ** 2) * sigma_D_all

    # combining H0 values with intrinsic scatter
    H0_mean, sigma_H0_mean, chi2_red, sigma_int = constant_fit(
        H0_i, sigma_H0_i, add_intrinsic=True)

    # bundling again
    results = {
        'names': names_all,
        'v_all': v_all,
        'D_all': D_all,
        'sigma_D_all': sigma_D_all,
        'H0_i': H0_i,
        'sigma_H0_i': sigma_H0_i,
        'H0_mean': H0_mean,
        'sigma_H0_mean': sigma_H0_mean,
        'chi2_red_H0': chi2_red,
        'sigma_int_H0': sigma_int,
    }

    return results


# =============== MAKING HUBBLE DIAGRAM ===============
def plot_hubble_diagram(hubble_results, H0_best, filename='hubble_plot.png'):
    """Creating hubble diagram (plotting v vs D)"""

    # makes hubble diagram
    D = hubble_results['D_all']
    sigma_D = hubble_results['sigma_D_all']
    v = hubble_results['v_all']
    names = hubble_results['names']

    plt.figure()
    plt.errorbar(D, v, xerr=sigma_D, fmt='o', capsize=5, color='red', markersize=3)

    for i, name in enumerate(names):
        plt.text(D[i] * 1.02, v[i] * 1.02, name, fontsize=4)

    # plots straight line with best fit Hubbles constant (line of regression)
    D_line = np.linspace(0, max(D) * 1.1, 200)
    v_line = H0_best * D_line
    plt.plot(D_line, v_line, color='green', label=f'H0 = {H0_best: .1f} kms^-1 Mpc^-1')

    plt.xlabel('Distance D (Mpc)')
    plt.ylabel('Recession velocity v (kms^-1)')
    plt.title('Hubble Diagram')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', color='0.85', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_pl_relation(pl_results, filename= 'pl_relation.png'):
    """plottion period luminosity relation, (absolute magnitude vs log10(period))"""
    logP = pl_results['logP']
    M = pl_results['M_abs']
    sigma_M = pl_results['sigma_M']
    names = pl_results['names']

    # Get fit parameters (shifted)
    a = pl_results['alpha_shift']
    b = pl_results['beta_shift']
    x0 = pl_results['x0']

    plt.figure()

    # plotting data points with error bars
    plt.errorbar(logP, M, yerr=sigma_M, fmt='o', capsize=5,
                 color='red', label='MW Cepheids', markersize=5)

    # labelling point s
    for i, name in enumerate(names):
        plt.text(logP[i], M[i] - 0.1, name, fontsize=6, ha='center')

    # plotting regression line
    x_plot = np.linspace(min(logP) - 0.1, max(logP) + 0.1, 100)
    # M = a(x - x0) + b
    y_plot = a * (x_plot - x0) + b

    plt.plot(x_plot, y_plot, color='green', linestyle='-', label='Best Fit')

    # Invert y-axis (magnitudes: lower is brighter)
    plt.gca().invert_yaxis()

    plt.xlabel('log10(P)[days]')
    plt.ylabel('absolute magnitude (M)')
    plt.title('cepheid period luminosity relation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f'PL relation plot saved as "{filename}"')


# =============== MAIN ===============

# assigning easier variable names to data files
mw_file, ngc_file, other_file = (
    'MW_Cepheids.dat',
    'ngc4527_cepheids.dat',
    'other_galaxies.dat'
)

# period luminosity relation
pl = fit_pl_relation(mw_file)

# prints period luminosity slope, intercept, uncertainties, correlation & shifted with less correlation
print('period luminosity relation (milky way cepheids)\n',
      f'  unshifted: M = alpha * log10(P) + beta\n',
      f'    alpha = {pl["alpha"]:.3f} ± {pl["sigma_alpha"]:.3f}\n',
      f'    beta  = {pl["beta"]:.3f} ± {pl["sigma_beta"]:.3f}\n',
      f'    corr(alpha, beta) = {pl["rho"]:.3f}\n',
      f'    reduced chi^2 = {pl["chi2_red"]:.3f}\n\n',
      '  shifted: M = a * (log10 P - x0) + b\n',
      f'    x0 = {pl["x0"]:.3f}\n',
      f'    a  = {pl["alpha_shift"]:.3f} ± {pl["sigma_alpha_shift"]:.3f}\n',
      f'    b  = {pl["beta_shift"]:.3f} ± {pl["sigma_beta_shift"]:.3f}\n',
      f'    corr(a, b) = {pl["rho_shift"]:.3f}\n\n'
      f'    corr: correlation coefficient between slope & intercept\n'
      f'    x0: reference value of log10(P) to reduce correlation\n\n')

# create pl relation plot
plot_pl_relation(pl, filename='pl_relation.png')

# distance to NGC4527
ngc = distance_ngc4527(ngc_file, pl, A_MW=0.0682, add_intrinsic=True)

# prints mean for galaxy, uncertainties, instrinsic scatter (mag), chi^2, distance + error
print('distance to NGC4527\n'
      f'  mean distance modulus = {ngc["mu_mean"]:.3f} ± {ngc["sigma_mu_mean"]:.3f} magnitudes\n'
      f'  intrinsic dispersion in distance modulus = {ngc["sigma_int_mu"]:.3f} magnitudes\n'
      f'  reduced chi-squared for distance modulus fit = {ngc["chi2_red_mu"]:.3f}\n'
      f'  galaxy distance = {ngc["d_Mpc"]:.2f} ± {ngc["sigma_d_Mpc"]:.2f} megaparsecs (Mpc)\n\n')

# hubble constant
hub = hubbles_constant(ngc, other_file, v_rec_ngc=1152.0)

# prints all Hubble constant estimates and the combined one
print('hubble constant from 8 galaxies\n'
      '  individual hubble constant values (km/s per Mpc):')
for name, H0_i, sH0 in zip(hub['names'], hub['H0_i'], hub['sigma_H0_i']):
    print(f'    {name:10s}: H0 = {H0_i:6.1f} ± {sH0:5.1f} km/s/Mpc')

print('  \ncombined hubble constant:\n'
      f'    H0 = {hub["H0_mean"]:.1f} ± {hub["sigma_H0_mean"]:.1f} km/s/Mpc\n'
      f'  intrinsic dispersion in H0 = {hub["sigma_int_H0"]:.1f} km/s/Mpc\n'
      f'  reduced chi-squared for H0 fit = {hub["chi2_red_H0"]:.3f}\n\n')


# saves hubble diagram and prints naem of file saved as
plot_hubble_diagram(hub, hub['H0_mean'], filename='hubble_diagram.png')
print('hubble diagram saved as "hubble_diagram.png"')
