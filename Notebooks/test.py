# MCMC Analysis and Plotting Functions
import numpy as np
import matplotlib.pyplot as plt
import corner  # pip install corner if not already installed


def plot_fitting_results(event, chi2_func, best_params, mcmc_results):
    """Plot transit and RV fitting results with MCMC uncertainty"""
    import matplotlib.pyplot as plt
    import numpy as np
    from exozippy.exozippy_tran import exozippy_tran
    from exozippy.exozippy_rv import exozippy_rv
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Get the data
    transit_data = event.get('transit_data', [])
    rv_data = event.get('rv_data', [])
    
    # Plot Transit Data
    if transit_data:
        print("Plotting transit light curve...")
        tdata = transit_data[0]  # Use first transit dataset
        
        # Plot data points
        if 'flux_err' in tdata and tdata['flux_err'] is not None:
            ax1.errorbar(tdata['bjd'], tdata['flux'], yerr=tdata['flux_err'], 
                        fmt='o', alpha=0.6, markersize=3, label='Data')
        else:
            ax1.scatter(tdata['bjd'], tdata['flux'], alpha=0.6, s=10, label='Data')
        
        # Generate model curve
        # Get parameters for transit model
        period = best_params.get('period_0', 2.899703)
        tc = best_params.get('tc_0', 2454218.76016)
        p = best_params.get('p_0', 0.114)
        inc = best_params.get('inc_0', 1.57)
        ar = best_params.get('ar_0', 11.9)
        e = best_params.get('e_0', 0.0)
        omega = best_params.get('omega_0', np.pi/2)
        u1 = best_params.get('u1_0', 0.3)
        u2 = best_params.get('u2_0', 0.2)
        f0 = best_params.get('f0_0', 1.0)

        # Create fine time grid for smooth model curve
        t_min, t_max = np.min(tdata['bjd']), np.max(tdata['bjd'])
        t_model = np.linspace(t_min, t_max, 1000)
        
        # Calculate model
        flux_model = exozippy_tran(t_model, inc, ar, tc, period, e, omega, p, u1, u2, f0)
        ax1.plot(t_model, flux_model, 'r-', linewidth=2, label='Best fit model')
        
        # Plot MCMC uncertainty envelope if available
        if mcmc_results['converged']:
            # Sample from posterior to show uncertainty
            cold_chains = mcmc_results['pars'][0, :, :, mcmc_results['burnndx']:]
            samples = cold_chains.reshape(-1, cold_chains.shape[1])
            n_samples = min(100, len(samples))  # Use up to 100 posterior samples
            
            transit_models = []
            for i in np.random.choice(len(samples), n_samples):
                sample_params = {mcmc_results['tofit'][j]: samples[i, j] for j in range(len(mcmc_results['tofit']))}
                # Use sample parameters, fill in fixed ones
                p_samp = sample_params.get('p_0', p)
                period_samp = sample_params.get('period_0', period)
                tc_samp = sample_params.get('tc_0', tc)
                f0_samp = sample_params.get('f0_0', f0)
                
                flux_samp = exozippy_tran(t_model, inc, ar, tc_samp, period_samp, e, omega, p_samp, u1, u2, f0_samp)
                transit_models.append(flux_samp)
            
            transit_models = np.array(transit_models)
            # Calculate percentiles for uncertainty envelope
            # flux_16 = np.percentile(transit_models, 16, axis=0)
            # flux_84 = np.percentile(transit_models, 84, axis=0)
            # ax1.fill_between(t_model, flux_16, flux_84, alpha=0.3, color='red', label='1σ uncertainty')
        
        ax1.set_xlabel('BJD')
        ax1.set_ylabel('Relative Flux')
        ax1.set_title('Transit Light Curve Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate and show residuals
        flux_data_model = exozippy_tran(tdata['bjd'], inc, ar, tc, period, e, omega, p, u1, u2, f0)
        residuals = tdata['flux'] - flux_data_model
        rms = np.std(residuals)
        ax1.text(0.02, 0.98, f'Transit RMS: {rms*1e6:.1f} ppm', 
                transform=ax1.transAxes, verticalalignment='top')
    
    # Plot RV Data
    if rv_data:
        print("Plotting radial velocity curve...")
        # Combine all RV datasets
        all_rv_times = []
        all_rv_values = []
        all_rv_errors = []
        
        for rvdata in rv_data:
            all_rv_times.extend(rvdata['bjd'])
            all_rv_values.extend(rvdata['rv'])
            if 'err' in rvdata and rvdata['err'] is not None:
                all_rv_errors.extend(rvdata['err'])
            else:
                all_rv_errors.extend([10.0] * len(rvdata['bjd']))  # Default error
        
        all_rv_times = np.array(all_rv_times)
        all_rv_values = np.array(all_rv_values)
        all_rv_errors = np.array(all_rv_errors)
        
        # Plot data points
        ax2.errorbar(all_rv_times, all_rv_values, yerr=all_rv_errors,
                    fmt='o', alpha=0.6, markersize=4, label='RV Data')
        
        # Generate model curve
        period = best_params.get('period_0', 2.899703)
        tc = best_params.get('tc_0', 2454218.76016)
        K = best_params.get('K_0', 89.0)
        gamma = best_params.get('gamma_0', -4500.0)
        e = event.get('e_0', 0.0)
        omega = event.get('omega_0', np.pi/2)
        
        # Create fine time grid for smooth model curve
        t_min, t_max = np.min(all_rv_times), np.max(all_rv_times)
        t_model_rv = np.linspace(t_min, t_max, 1000)
        
        # Calculate model
        rv_model = exozippy_rv(t_model_rv, tc, period, gamma, K, e, omega)
        ax2.plot(t_model_rv, rv_model, 'b-', linewidth=2, label='Best fit model')
        
        # Plot MCMC uncertainty envelope if available
        if mcmc_results['converged']:
            rv_models = []
            for i in np.random.choice(len(samples), n_samples):
                sample_params = {mcmc_results['tofit'][j]: samples[i, j] for j in range(len(mcmc_results['tofit']))}
                # Use sample parameters, fill in fixed ones
                K_samp = sample_params.get('K_0', K) 
                period_samp = sample_params.get('period_0', period)
                tc_samp = sample_params.get('tc_0', tc)
                gamma_samp = sample_params.get('gamma_0', gamma)
                
                rv_samp = exozippy_rv(t_model_rv, tc_samp, period_samp, gamma_samp, K_samp, e, omega)
                rv_models.append(rv_samp)
            
            rv_models = np.array(rv_models)
            # Calculate percentiles for uncertainty envelope
            # rv_16 = np.percentile(rv_models, 16, axis=0)
            # rv_84 = np.percentile(rv_models, 84, axis=0)
            # ax2.fill_between(t_model_rv, rv_16, rv_84, alpha=0.3, color='blue', label='1σ uncertainty')
        
        ax2.set_xlabel('BJD')
        ax2.set_ylabel('Radial Velocity (m/s)')
        ax2.set_title('Radial Velocity Curve Fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calculate and show residuals
        rv_data_model = exozippy_rv(all_rv_times, tc, period, gamma, K, e, omega)
        rv_residuals = all_rv_values - rv_data_model
        rv_rms = np.std(rv_residuals)
        ax2.text(0.02, 0.98, f'RV RMS: {rv_rms:.1f} m/s', 
                transform=ax2.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('/Users/wangxianyu/Program/Github/EXOZIPPy/Notebooks/mcmc_fitting_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'Notebooks/mcmc_fitting_results.png'")

def plot_traces(results, figsize=(12, 8)):
    """Plot MCMC trace plots"""
    
    cold_chains = results['pars'][0, :, :, :]  # All steps, not just post burn-in
    n_chains, n_params, n_steps = cold_chains.shape
    param_names = results['tofit']
    
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
    if n_params == 1:
        axes = [axes]
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        
        # Plot each chain
        for chain in range(n_chains):
            ax.plot(cold_chains[chain, i, :], alpha=0.7, linewidth=0.8)
        
        # Mark burn-in
        if 'burnndx' in results:
            ax.axvline(results['burnndx'], color='red', linestyle='--', alpha=0.7, label='Burn-in')
        
        ax.set_ylabel(param_name)
        ax.grid(True, alpha=0.3)
        
        if i == 0 and 'burnndx' in results:
            ax.legend()
    
    axes[-1].set_xlabel('Step')
    plt.suptitle('MCMC Trace Plots', fontsize=14)
    plt.tight_layout()
    return fig

def plot_corner(results, figsize=(10, 10)):
    """Create corner plot of posterior distributions"""
    
    # Extract post-burn-in samples
    cold_chains = results['pars'][0, :, :, results['burnndx']:]
    samples = cold_chains.reshape(-1, cold_chains.shape[1])  # Flatten
    param_names = results['tofit']
    
    # Create corner plot
    fig = corner.corner(
        samples,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],  # 68% confidence intervals
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12},
        figsize=figsize
    )
    
    return fig

def print_summary(results):
    """Print parameter summary statistics"""
    
    cold_chains = results['pars'][0, :, :, results['burnndx']:]
    samples = cold_chains.reshape(-1, cold_chains.shape[1])
    param_names = results['tofit']
    
    print("Parameter Summary (68% confidence intervals):")
    print("="*50)
    
    for i, param_name in enumerate(param_names):
        param_samples = samples[:, i]
        
        # Calculate statistics
        median = np.median(param_samples)
        std = np.std(param_samples)
        q16, q84 = np.percentile(param_samples, [16, 84])
        
        print(f"{param_name:12s}: {median:8.4f} +{q84-median:6.4f} -{median-q16:6.4f} (σ={std:.4f})")
    
    # Print convergence info
    if 'converged' in results:
        print(f"\\nConverged: {results['converged']}")
        print(f"Final G-R statistic: {results.get('max_gr', 'N/A')}")
        print(f"Independent samples: {results.get('tz', 'N/A')}")

def analyze_mcmc_results(results, save_plots=True, plot_dir="./"):
    """Complete MCMC analysis with plots and statistics"""
    
    # Print summary statistics
    print_summary(results)
    
    # Create plots
    trace_fig = plot_traces(results, figsize=(12, 10))
    corner_fig = plot_corner(results, figsize=(10, 10))
    
    # Save plots if requested
    if save_plots:
        trace_fig.savefig(f"{plot_dir}/trace_plot.png", dpi=150, bbox_inches='tight')
        corner_fig.savefig(f"{plot_dir}/corner_plot.png", dpi=150, bbox_inches='tight')
        print(f"Plots saved to {plot_dir}")
    
    plt.show()
    
    return trace_fig, corner_fig

print("MCMC analysis functions loaded successfully!")




# Complete EXOZIPPy MCMC Workflow Example
# Note: This is a demonstration - actual run may take significant time

# Step 1: Build model and create chi-squared function
from exozippy.build_simple_model import build_simple_model, create_default_parameters

event, chi2_func = build_simple_model(
    parfile="/Users/wangxianyu/Program/Github/EXOZIPPy/data/exofastv2/examples/hat3/HAT-3.priors",
    tranpath="/Users/wangxianyu/Program/Github/EXOZIPPy/data/exofastv2/examples/hat3/*.dat",
    rvpath="/Users/wangxianyu/Program/Github/EXOZIPPy/data/exofastv2/examples/hat3/*.rv",
    # sedfile="/Users/wangxianyu/Program/Github/EXOZIPPy/data/exofastv2/examples/hat3/HAT-3.sed",  # Optionally include SED
    debug=False  # Enable debug to see what data is being processed
)

# # Step 2: Set up starting parameters
params = create_default_parameters(event)

# # Override with better HAT-P-3 values for faster convergence
hat3_params = {
    'period_0': 2.899703,      # HAT-P-3b known period (days)
    'tc_0': 2454218.76016,     # Known transit time (BJD)
    'p_0': 0.114,              # Known radius ratio > 0
    'K_0': 89.0,               # Known RV amplitude (m/s)
    'gamma_0': 00.0,        # Systemic velocity (m/s)
    'inc_0': 1.57,             # Near edge-on
    'ar_0': 11.9,              # Semi-major axis ratio
    'omega_0': 1.57,              # Argument of periastron
    'f0_0': 1.0,                   # Baseline flux
    'u1_0': 0.3,                    # Linear limb darkening coefficient
    'u2_0': 0.2                     # Quadratic limb darkening coefficient
}

params.update(hat3_params)
print("Starting parameters:")
for key, val in params.items():
    print(f"  {key}: {val}")

# # Step 3: Set up MCMC with manual step sizes for better convergence
tofit_params = ['period_0', 'tc_0', 'p_0', 'gamma_0', 'inc_0', 'K_0', 'ar_0', 'f0_0', 'u1_0', 'u2_0']  # Start with just 4 well-constrained parameters

# Manual step sizes based on typical parameter uncertainties (MUCH larger for exploration)
manual_scales = {
    'period_0': 1e-2,      # Period uncertainty ~ 10 milliseconds (100x larger)
    'tc_0': 0.1,           # Transit time uncertainty ~ 2.4 hours (10x larger)
    'p_0': 0.01,           # Radius ratio uncertainty ~ 1% (same)
    'K_0': 10.0,           # RV amplitude uncertainty ~ 10 m/s
    'gamma_0': 0.0,       # Systemic velocity uncertainty ~ 50 m/s
    'inc_0': 0.1,          # Inclination uncertainty ~ 6 degrees
    'ar_0': 1.0,          # Semi-major axis ratio uncertainty ~ 1.0
    'f0_0': 0.1,           # Baseline flux uncertainty ~ 0.1
    'u1_0': 0.1,           # Linear limb darkening coefficient uncertainty ~ 0.1
    'u2_0': 0.1            # Quadratic limb darkening coefficient uncertainty ~ 0.1
}

scales = {param: manual_scales[param] for param in tofit_params}
bestpars = {param: hat3_params[param] for param in tofit_params}

# Debug: Test chi2 function at starting point (now includes priors penalty)
print("\\nTesting chi2 function (including priors penalty)...")
print(f"Best parameters: {bestpars}")
chi2_initial = chi2_func(bestpars)
print(f"Initial chi2 (with priors): {chi2_initial}")

# Test priors penalty separately by creating parameters outside priors bounds
print("\\nTesting priors penalty with out-of-bounds parameters...")
test_oob_params = bestpars.copy()
# Make some parameters way off to trigger large priors penalty
test_oob_params['period_0'] = 100.0  # Way off from HAT-P-3 period
chi2_oob = chi2_func(test_oob_params) 
priors_penalty = chi2_oob - chi2_initial
print(f"Chi2 with out-of-bounds period: {chi2_oob}")
print(f"Estimated priors penalty: {priors_penalty:.1f}")

# Let's try the EXOZIPPy automatic scale calculation like EXOFASTv2
print("\\nUsing EXOZIPPy automatic step size calculation (like EXOFASTv2)...")
from exozippy.exozippy_getmcmcscale import exozippy_getmcmcscale

try:
    auto_scales, auto_bestpars = exozippy_getmcmcscale(
        bestpars=bestpars,
        chi2func=chi2_func,
        tofit=tofit_params,
        debug=True,
        skipiter=False  # Don't skip iterations - let it optimize
    )
    print("\\n✅ EXOZIPPy automatic scales:", auto_scales)
    scales = auto_scales
    bestpars_optimized = auto_bestpars
    
    # Test that these automatic scales produce reasonable chi2 changes
    print("\\nTesting automatic scales:")
    for param in tofit_params:
        test_pars = bestpars_optimized.copy()
        original_chi2 = chi2_func(test_pars)
        test_pars[param] += scales[param]
        perturbed_chi2 = chi2_func(test_pars)
        delta_chi2 = perturbed_chi2 - original_chi2
        print(f"{param}: +1σ step → Δχ² = {delta_chi2:.2f} (target ≈ 1.0)")
        
except Exception as e:
    print(f"Automatic scale calculation failed: {e}")
    print("Falling back to manual scales...")
    scales = {param: manual_scales[param] for param in tofit_params}
    bestpars_optimized = bestpars

# Test the chi2 function performance and data size
import time
print("Testing chi2 function performance:")

# Time a single chi² evaluation
start = time.time()
original_chi2 = chi2_func(bestpars_optimized)
single_eval_time = time.time() - start

print(f"Single chi² evaluation: {single_eval_time*1000:.2f} ms")
print(f"Chi² value (with priors): {original_chi2:.2f}")

# Test multiple evaluations with LARGE perturbations to check for caching issues
print("\\nTesting for caching issues with LARGE perturbations:")
eval_times = []
chi2_values = []
transit_chi2_values = []

for i in range(5):
    # Add LARGE random perturbation that should definitely change chi2
    test_pars = bestpars_optimized.copy()
    perturbations = {}
    for param in tofit_params:
        # Much larger perturbations - 1000x the scale
        perturbation = np.random.normal(0, scales[param] * 1000)
        test_pars[param] += perturbation
        perturbations[param] = perturbation
    
    print(f"\\nTest {i+1} perturbations: {perturbations}")
    start = time.time()
    chi2_val = chi2_func(test_pars)
    eval_time = time.time() - start
    
    eval_times.append(eval_time * 1000)
    chi2_values.append(chi2_val)

print(f"5 evaluations: mean = {np.mean(eval_times):.2f} ms, std = {np.std(eval_times):.2f} ms")
print(f"Chi² values (with priors): min = {np.min(chi2_values):.1f}, max = {np.max(chi2_values):.1f}, range = {np.max(chi2_values)-np.min(chi2_values):.1f}")

if np.std(eval_times) < 0.01:  # Very low variance in timing
    print("⚠️  SUSPICIOUS: Evaluation times are too consistent - might be cached!")
if np.max(chi2_values) - np.min(chi2_values) < 1e-10:  # Identical chi2 values
    print("⚠️  SUSPICIOUS: Chi² values are identical - definitely cached!")
else:
    print("✅ Chi² values vary properly - priors penalty working correctly")
    
print("\\nStep sizes:", scales)

# Run MCMC with EXOFASTv2-style automatic scales
print("\\nStarting MCMC sampling with proper EXOFASTv2-style scales...")
from exozippy.exozippy_demcpt import exozippy_demcpt

# Choose MCMC backend: 'exozippy' or 'emcee'
backend = 'emcee'  # Switch to 'exozippy' to use the original implementation

if backend == 'emcee':
    print(f"Using emcee backend with Differential Evolution moves...")
    results = exozippy_demcpt(
        chi2func=chi2_func,
        bestpars=bestpars_optimized,
        scale=scales,
        backend='emcee',
        nchains=32,          # Number of emcee walkers
        maxsteps=10000,      # MCMC steps
        debug=True,          # Show progress
        seed=42              # For reproducibility
    )
else:
    print(f"Using EXOZIPPy backend with parallel tempering...")
    results = exozippy_demcpt(
        chi2func=chi2_func,
        bestpars=bestpars_optimized,
        scale=scales,
        backend='exozippy',
        nchains=4,           # 4 chains per temperature
        ntemps=3,            # 3 temperature levels
        maxsteps=10000,      # MCMC steps
        maxgr=1.1,           # Convergence threshold
        mintz=1000,          # Minimum independent samples
        debug=True,          # Show progress
        stretch=False        # Use DE (not stretch) like EXOFASTv2
    )

print(f"\\nMCMC completed! Converged: {results['converged']}")
print(f"Final G-R statistic: {results.get('max_gr', 'N/A')}")
print(f"Independent samples: {results.get('tz', 'N/A')}")
print("(Note: Chi² values now include priors penalty for proper Bayesian analysis)")

# # Print parameter summary
# if results['converged']:
if True:
    print("\\n=== Parameter Results ===")
    cold_chains = results['pars'][0, :, :, results['burnndx']:]
    samples = cold_chains.reshape(-1, cold_chains.shape[1])
    param_names = results['tofit']
    
    print(f"Samples shape: {samples.shape}, Parameters: {param_names}")
    
    for i, param_name in enumerate(param_names):
        param_samples = samples[:, i]
        median = np.median(param_samples)
        std = np.std(param_samples)
        q16, q84 = np.percentile(param_samples, [16, 84])
        print(f"{param_name:12s}: {median:12.8f} +{q84-median:10.8f} -{median-q16:10.8f}")
        
    # Also show the best chi² parameters found  
    print("\\n=== Best Chi² Point (including priors) ===")
    min_chi2_idx = np.unravel_index(np.argmin(results['chi2'][:, :, :]), results['chi2'].shape)
    best_pars = results['pars'][min_chi2_idx[0], min_chi2_idx[1], :, min_chi2_idx[2]]
    best_chi2 = results['chi2'][min_chi2_idx]
    print(f"Best chi² (with priors) = {best_chi2:.2f}")
    best_params_dict = {}
    for i, param_name in enumerate(param_names):
        best_params_dict[param_name] = best_pars[i]
        print(f"{param_name:12s}: {best_pars[i]:12.8f}")
        
    # Plot the fitting results
    print("\\n=== Plotting Results ===")
    plot_fitting_results(event, chi2_func, best_params_dict, results)
    
else:
    print("\\nMCMC did not converge - consider running longer or adjusting parameters")

