# Simple test to create the plots with the corrected format
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from exozippy.build_simple_model import build_simple_model, create_default_parameters
from exozippy.exozippy_demcpt import exozippy_demcpt
from exozippy.exozippy_getmcmcscale import exozippy_getmcmcscale
from exozippy.exozippy_tran import exozippy_tran
from exozippy.exozippy_rv import exozippy_rv

# Load the HAT-P-3 data using corrected paths
print("Loading HAT-P-3 data...")
event, chi2_func = build_simple_model(
    parfile="data/exofastv2/examples/hat3/HAT-3.priors",
    tranpath="data/exofastv2/examples/hat3/*.dat",
    rvpath="data/exofastv2/examples/hat3/*.rv",
    sedfile="data/exofastv2/examples/hat3/HAT-3.sed",
    debug=True
)

# Get default parameters
default_params = create_default_parameters(event)
print(f"\nParameters to fit: {list(default_params.keys())}")

# Test with just 1000 steps for quick plotting
tofit_params = ['period_0', 'tc_0', 'p_0']
reduced_params = {k: default_params[k] for k in tofit_params}

print(f"\nRunning short MCMC with {tofit_params} for plotting...")
results = exozippy_demcpt(
    chi2_func,
    reduced_params,
    backend='emcee',
    maxsteps=1000,
    nthin=10,
    debug=False
)

# Create plots
def plot_fitting_results_corrected(event, best_params, mcmc_results):
    """Plot transit and RV fitting results with MCMC uncertainty"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Get the data
    transit_data = event.get('transit_data', [])
    rv_data = event.get('rv_data', [])
    
    # Extract samples from MCMC results (emcee format)
    samples = mcmc_results.get('samples', None)
    param_names = mcmc_results.get('param_names', [])
    
    print(f"MCMC results keys: {list(mcmc_results.keys())}")
    print(f"Parameter names: {param_names}")
    if samples is not None:
        print(f"Samples shape: {samples.shape}")
    
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
        
        # Generate model curve - get parameters
        period = best_params.get('period_0', 2.899703)
        tc = best_params.get('tc_0', 2454218.76016) 
        p = best_params.get('p_0', 0.114)
        # Fixed parameters from HAT-P-3
        inc = np.radians(87.24)  # Convert to radians
        ar = 11.9
        e = 0.0
        omega = np.pi/2
        u1 = 0.3
        u2 = 0.2
        
        # Create fine time grid for smooth model curve
        t_min, t_max = np.min(tdata['bjd']), np.max(tdata['bjd'])
        t_model = np.linspace(t_min, t_max, 1000)
        
        # Calculate best fit model
        flux_model = exozippy_tran(t_model, inc, ar, tc, period, e, omega, p, u1, u2, 1.0)
        ax1.plot(t_model, flux_model, 'r-', linewidth=2, label='Best fit model')
        
        # Plot MCMC uncertainty envelope if available
        if samples is not None and len(samples) > 0:
            n_samples = min(50, len(samples))  # Use fewer samples for speed
            
            transit_models = []
            for i in np.random.choice(len(samples), n_samples):
                sample_params = {}
                for j, pname in enumerate(param_names):
                    sample_params[pname] = samples[i, j]
                
                # Use sample parameters, fill in fixed ones
                p_samp = sample_params.get('p_0', p)
                period_samp = sample_params.get('period_0', period)
                tc_samp = sample_params.get('tc_0', tc)
                
                try:
                    flux_samp = exozippy_tran(t_model, inc, ar, tc_samp, period_samp, e, omega, p_samp, u1, u2, 1.0)
                    transit_models.append(flux_samp)
                except:
                    continue
            
            if len(transit_models) > 0:
                transit_models = np.array(transit_models)
                # Calculate percentiles for uncertainty envelope
                flux_16 = np.percentile(transit_models, 16, axis=0)
                flux_84 = np.percentile(transit_models, 84, axis=0)
                ax1.fill_between(t_model, flux_16, flux_84, alpha=0.3, color='red', label='1σ uncertainty')
        
        ax1.set_xlabel('BJD')
        ax1.set_ylabel('Relative Flux')
        ax1.set_title('Transit Light Curve Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate and show residuals
        flux_data_model = exozippy_tran(tdata['bjd'], inc, ar, tc, period, e, omega, p, u1, u2, 1.0)
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
            all_rv_values.extend(rvdata['mnvel'])
            if 'errvel' in rvdata and rvdata['errvel'] is not None:
                all_rv_errors.extend(rvdata['errvel'])
            else:
                all_rv_errors.extend([2.0] * len(rvdata['bjd']))  # Default error from data
        
        all_rv_times = np.array(all_rv_times)
        all_rv_values = np.array(all_rv_values)
        all_rv_errors = np.array(all_rv_errors)
        
        # Plot data points
        ax2.errorbar(all_rv_times, all_rv_values, yerr=all_rv_errors,
                    fmt='o', alpha=0.6, markersize=4, label='RV Data')
        
        # Generate model curve - default HAT-P-3 parameters
        period = best_params.get('period_0', 2.899703)
        tc = best_params.get('tc_0', 2454218.76016)
        K = 89.0  # Fixed K from HAT-P-3 
        gamma = -4500.0  # Fixed gamma from HAT-P-3
        e = 0.0
        omega = np.pi/2
        
        # Create fine time grid for smooth model curve
        t_min, t_max = np.min(all_rv_times), np.max(all_rv_times)
        t_model_rv = np.linspace(t_min, t_max, 1000)
        
        # Calculate model
        rv_model = exozippy_rv(t_model_rv, tc, period, K, e, omega, gamma)
        ax2.plot(t_model_rv, rv_model, 'b-', linewidth=2, label='Best fit model')
        
        # Plot MCMC uncertainty envelope if available
        if samples is not None and len(samples) > 0:
            rv_models = []
            for i in np.random.choice(len(samples), n_samples):
                sample_params = {}
                for j, pname in enumerate(param_names):
                    sample_params[pname] = samples[i, j]
                
                # Use sample parameters for tc and period only (K, gamma are fixed for HAT-P-3)
                period_samp = sample_params.get('period_0', period)
                tc_samp = sample_params.get('tc_0', tc)
                
                try:
                    rv_samp = exozippy_rv(t_model_rv, tc_samp, period_samp, K, e, omega, gamma)
                    rv_models.append(rv_samp)
                except:
                    continue
            
            if len(rv_models) > 0:
                rv_models = np.array(rv_models)
                # Calculate percentiles for uncertainty envelope
                rv_16 = np.percentile(rv_models, 16, axis=0)
                rv_84 = np.percentile(rv_models, 84, axis=0)
                ax2.fill_between(t_model_rv, rv_16, rv_84, alpha=0.3, color='blue', label='1σ uncertainty')
        
        ax2.set_xlabel('BJD')
        ax2.set_ylabel('Radial Velocity (m/s)')
        ax2.set_title('Radial Velocity Curve Fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calculate and show residuals
        rv_data_model = exozippy_rv(all_rv_times, tc, period, K, e, omega, gamma)
        rv_residuals = all_rv_values - rv_data_model
        rv_rms = np.std(rv_residuals)
        ax2.text(0.02, 0.98, f'RV RMS: {rv_rms:.1f} m/s', 
                transform=ax2.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plot_path = 'Notebooks/mcmc_fitting_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as '{plot_path}'")

# Generate the plots
if results['converged']:
    # Get best fit parameters
    best_params_dict = {}
    if 'samples' in results:
        samples = results['samples']
        param_names = results['param_names']
        for i, param in enumerate(param_names):
            best_params_dict[param] = np.median(samples[:, i])
    
    print("\n=== Plotting Results ===")
    plot_fitting_results_corrected(event, best_params_dict, results)
else:
    print("MCMC did not converge - using default parameters for plotting")
    plot_fitting_results_corrected(event, reduced_params, {'samples': None, 'param_names': []})