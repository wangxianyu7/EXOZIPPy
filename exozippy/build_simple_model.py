"""
Simplified model building for EXOZIPPy without PyMC dependencies

This creates a simple event structure and combined chi-squared function
that integrates SED, transit, and RV modeling components.

Based on build_model2.py but using pure Python/NumPy approach.
"""

import numpy as np
import glob
import math
from astropy import units as u
import astropy.constants as const
from pathlib import Path

# EXOZIPPy imports
from .read_par import read_par
from .read_tran import read_tran
from .read_rv import read_rv
from .sed.utils import mistmultised
from .exozippy_tran import exozippy_tran
from .exozippy_rv import exozippy_rv


def build_simple_model(parfile=None,
                       tranpath=None, 
                       rvpath=None,
                       sedfile=None,
                       nstars=1,
                       nplanets=1,
                       debug=False):
    """
    Build a simplified model structure for EXOZIPPy fitting
    
    Parameters:
    -----------
    parfile : str
        Parameter file path (.priors format)
    tranpath : str
        Transit data file path or glob pattern
    rvpath : str  
        RV data file path or glob pattern
    sedfile : str
        SED data file path
    nstars : int
        Number of stars in system
    nplanets : int
        Number of planets in system
    debug : bool
        Enable debug output
        
    Returns:
    --------
    event : dict
        Event structure containing model components
    chi2_func : callable
        Combined chi-squared function
    """
    
    if debug:
        print("Building simplified model...")
    
    # Load parameter file
    if parfile is not None:
        user_params = read_par(parfile)
    else:
        user_params = {}
    
    # Load data files
    transit_data = []
    if tranpath is not None:
        tranfiles = glob.glob(tranpath)
        if debug:
            print(f"Found {len(tranfiles)} transit files")
        for tranfile in tranfiles:
            transit_data.append(read_tran(tranfile))
    
    rv_data = []
    if rvpath is not None:
        rvfiles = glob.glob(rvpath)
        if debug:
            print(f"Found {len(rvfiles)} RV files")
        for rvfile in rvfiles:
            rv_data.append(read_rv(rvfile))
    
    # Create event structure
    event = {
        'user_params': user_params,
        'transit_data': transit_data,
        'rv_data': rv_data,
        'sedfile': sedfile,
        'nstars': nstars,
        'nplanets': nplanets
    }
    
    # Extract parameter values with defaults
    def get_param_value(param_name, default_value, params_dict=user_params):
        """Get parameter value with fallback to default"""
        if param_name in params_dict:
            # Use mu if available, otherwise initval, otherwise default
            if 'mu' in params_dict[param_name] and params_dict[param_name]['mu'] is not None:
                return params_dict[param_name]['mu']
            elif 'initval' in params_dict[param_name] and params_dict[param_name]['initval'] is not None:
                return params_dict[param_name]['initval']
        return default_value
    
    # Create priors penalty function
    def calculate_priors_penalty(params_dict):
        """
        Calculate prior probability penalty (-2 * log(prior))
        
        Parameters:
        -----------
        params_dict : dict
            Dictionary of parameter values
            
        Returns:
        --------
        priors_penalty : float
            Total priors penalty (-2 * log(prior))
        """
        penalty = 0.0
        
        for param_name, param_value in params_dict.items():
            if param_name in user_params:
                param_info = user_params[param_name]
                
                # Gaussian prior
                if 'mu' in param_info and 'sigma' in param_info:
                    mu = param_info['mu']
                    sigma = param_info['sigma']
                    if mu is not None and sigma is not None and sigma > 0:
                        # -2 * log(Gaussian) = constant + (x-mu)^2/sigma^2
                        penalty += ((param_value - mu) / sigma) ** 2
                
                # Uniform prior with bounds
                elif 'lower' in param_info and 'upper' in param_info:
                    lower = param_info['lower']
                    upper = param_info['upper']
                    if lower is not None and upper is not None:
                        if not (lower <= param_value <= upper):
                            # Outside bounds - return very large penalty
                            penalty += 1e10
                
                # Scale parameter (Jeffreys prior for positive parameters)
                elif param_info.get('scale', False) and param_value > 0:
                    # -2 * log(1/x) = 2 * log(x)
                    penalty += 2 * np.log(param_value)
                
                # Derived parameter bounds check
                elif 'derived' in param_info and param_info['derived']:
                    if 'lower' in param_info and param_info['lower'] is not None:
                        if param_value < param_info['lower']:
                            penalty += 1e10
                    if 'upper' in param_info and param_info['upper'] is not None:
                        if param_value > param_info['upper']:
                            penalty += 1e10
        
        return penalty

    # Create combined chi-squared function  
    def combined_chi2_function(params_dict):
        """
        Combined chi-squared function for all data types including priors penalty
        
        Parameters:
        -----------
        params_dict : dict
            Dictionary of parameter values
            
        Returns:
        --------
        chi2_total : float
            Total chi-squared value including priors penalty
        """
        
        # Check for NaN/inf in input parameters
        for param_name, param_value in params_dict.items():
            if not np.isfinite(param_value):
                return np.inf
        
        # Physical parameter bounds checks
        if 'p_0' in params_dict and params_dict['p_0'] <= 0:
            return np.inf  # Planet radius ratio must be positive
        if 'period_0' in params_dict and params_dict['period_0'] <= 0:
            return np.inf  # Period must be positive
        if 'ar_0' in params_dict and params_dict['ar_0'] <= 0:
            return np.inf  # Semi-major axis must be positive
        if 'inc_0' in params_dict and not (0 <= params_dict['inc_0'] <= np.pi):
            return np.inf  # Inclination must be 0 to π
        
        chi2_total = 0.0
        
        # Add priors penalty
        priors_penalty = calculate_priors_penalty(params_dict)
        if not np.isfinite(priors_penalty):
            return np.inf
        chi2_total += priors_penalty
        
        if debug and priors_penalty > 0:
            print(f"Priors penalty: {priors_penalty:.3f}")
        
        try:
            # SED chi-squared
            if sedfile is not None:
                if debug:
                    print("Calculating SED chi-squared...")
                
                # Extract stellar parameters
                teff = params_dict.get('teff_0', get_param_value('teff_0', 5777))
                logg = params_dict.get('logg_0', get_param_value('logg_0', 4.4))
                feh = params_dict.get('feh_0', get_param_value('feh_0', 0.0))
                av = params_dict.get('av_0', get_param_value('av_0', 0.0))
                distance = params_dict.get('distance_0', get_param_value('distance_0', 10.0))
                lstar = params_dict.get('lstar_0', get_param_value('lstar_0', 1.0))
                
                # Calculate SED chi-squared
                sed_chi2, _, _, _ = mistmultised(
                    teff=np.atleast_1d(teff),
                    logg=np.atleast_1d(logg),
                    feh=np.atleast_1d(feh),
                    av=np.atleast_1d(av),
                    distance=np.atleast_1d(distance),
                    lstar=np.atleast_1d(lstar),
                    errscale=np.atleast_1d(1.0),
                    sedfile=sedfile,
                    debug=False
                )
                
                chi2_total += sed_chi2
                if debug:
                    print(f"  SED chi2: {sed_chi2:.3f}")
            
            # Transit chi-squared
            if len(transit_data) > 0:
                if debug:
                    print("Calculating transit chi-squared...")
                
                transit_chi2 = 0.0
                
                for i, tdata in enumerate(transit_data):
                    # Extract transit parameters
                    inc = params_dict.get('inc_0', get_param_value('inc_0', np.pi/2))
                    ar = params_dict.get('ar_0', get_param_value('ar_0', 15.0))
                    tc = params_dict.get('tc_0', get_param_value('tc_0', 0.0))
                    period = params_dict.get('period_0', get_param_value('period_0', 3.0))
                    e = params_dict.get('e_0', get_param_value('e_0', 0.0))
                    omega = params_dict.get('omega_0', get_param_value('omega_0', np.pi/2))
                    p = params_dict.get('p_0', get_param_value('p_0', 0.1))  # Rp/Rs
                    u1 = params_dict.get('u1_0', get_param_value('u1_0', 0.3))
                    u2 = params_dict.get('u2_0', get_param_value('u2_0', 0.2))
                    f0 = params_dict.get('f0_0', get_param_value('f0_0', 1.0))
                    
                    # Calculate transit model
                    model_flux = exozippy_tran(
                        tdata['bjd'], inc, ar, tc, period, e, omega, p, u1, u2, f0
                    )
                    
                    # Calculate chi-squared with overflow protection
                    if 'flux_err' in tdata and tdata['flux_err'] is not None:
                        residuals = (tdata['flux'] - model_flux) / tdata['flux_err']
                    else:
                        # Use 1% error as default if no errors provided
                        residuals = (tdata['flux'] - model_flux) / (0.01 * np.ones_like(tdata['flux']))
                    
                    # Check for NaN/inf in residuals
                    if not np.all(np.isfinite(residuals)):
                        return np.inf
                    
                    # Clip extreme residuals to prevent overflow
                    residuals = np.clip(residuals, -1e6, 1e6)
                    chi2_contribution = np.sum(residuals**2)
                    
                    if not np.isfinite(chi2_contribution):
                        return np.inf
                    
                    transit_chi2 += chi2_contribution
                    
                    if debug:
                        print(f"    Transit file: {len(tdata['bjd'])} points, chi2 = {np.sum(residuals**2):.3f}")
                
                chi2_total += transit_chi2
                if debug:
                    print(f"  Total transit chi2: {transit_chi2:.3f}")
            
            # RV chi-squared  
            if len(rv_data) > 0:
                if debug:
                    print("Calculating RV chi-squared...")
                
                rv_chi2 = 0.0
                
                for i, rdata in enumerate(rv_data):
                    # Extract RV parameters
                    tc = params_dict.get('tc_0', get_param_value('tc_0', 0.0))
                    period = params_dict.get('period_0', get_param_value('period_0', 3.0))
                    gamma = params_dict.get('gamma_0', get_param_value('gamma_0', 0.0))
                    K = params_dict.get('K_0', get_param_value('K_0', 50.0))
                    
                    # Calculate RV model
                    model_rv = exozippy_rv(rdata['bjd'], tc, period, gamma, K)
                    
                    # Calculate chi-squared with overflow protection
                    if 'rv_err' in rdata and rdata['rv_err'] is not None:
                        residuals = (rdata['rv'] - model_rv) / rdata['rv_err']
                    else:
                        # Use 1 m/s error as default if no errors provided
                        residuals = (rdata['rv'] - model_rv) / (1.0 * np.ones_like(rdata['rv']))
                    
                    # Check for NaN/inf in residuals
                    if not np.all(np.isfinite(residuals)):
                        return np.inf
                    
                    # Clip extreme residuals to prevent overflow
                    residuals = np.clip(residuals, -1e6, 1e6)
                    chi2_contribution = np.sum(residuals**2)
                    
                    if not np.isfinite(chi2_contribution):
                        return np.inf
                    
                    rv_chi2 += chi2_contribution
                
                chi2_total += rv_chi2
                if debug:
                    print(f"  RV chi2: {rv_chi2:.3f}")
            
            if debug:
                print(f"Total chi2 (including priors): {chi2_total:.3f}")
            
            return chi2_total
            
        except Exception as e:
            if debug:
                print(f"Error in chi2 calculation: {e}")
            return np.inf
    
    # Store chi2 function in event
    event['chi2_function'] = combined_chi2_function
    
    if debug:
        print("Simple model building complete!")
    
    return event, combined_chi2_function


def create_default_parameters(event):
    """
    Create default parameter dictionary from event structure
    
    Parameters:
    -----------
    event : dict
        Event structure from build_simple_model
        
    Returns:
    --------
    params : dict
        Default parameter dictionary
    """
    
    # Default stellar parameters
    params = {
        'teff_0': 5777.0,      # Effective temperature (K)
        'logg_0': 4.4,         # Surface gravity (log cm/s^2)
        'feh_0': 0.0,          # Metallicity [Fe/H]
        'av_0': 0.0,           # V-band extinction (mag)
        'distance_0': 10.0,    # Distance (pc)
        'lstar_0': 1.0,        # Stellar luminosity (Lsun)
    }
    
    # Default planet parameters
    if len(event.get('transit_data', [])) > 0 or len(event.get('rv_data', [])) > 0:
        params.update({
            'period_0': 3.0,       # Orbital period (days)
            'tc_0': 0.0,           # Time of conjunction (BJD)
            'inc_0': np.pi/2,      # Inclination (radians)
            'ar_0': 15.0,          # Semi-major axis in stellar radii
            'e_0': 0.0,            # Eccentricity  
            'omega_0': np.pi/2,    # Argument of periastron (radians)
        })
    
    # Transit-specific parameters
    if len(event.get('transit_data', [])) > 0:
        params.update({
            'p_0': 0.1,            # Planet-to-star radius ratio
            'u1_0': 0.3,           # Linear limb darkening
            'u2_0': 0.2,           # Quadratic limb darkening
        })
    
    # RV-specific parameters
    if len(event.get('rv_data', [])) > 0:
        params.update({
            'gamma_0': 0.0,        # Systemic velocity (m/s)
            'K_0': 50.0,           # RV semi-amplitude (m/s)
        })
    
    # Override with user parameters if provided
    user_params = event.get('user_params', {})
    for param_name, param_info in user_params.items():
        if 'mu' in param_info and param_info['mu'] is not None:
            params[param_name] = param_info['mu']
        elif 'initval' in param_info and param_info['initval'] is not None:
            params[param_name] = param_info['initval']
    
    return params


def test_simple_model():
    """Test the simple model building"""
    
    # Test with HAT-P-3 data
    parfile = "../data/exofastv2/examples/hat3/HAT-3.priors"
    tranpath = "../data/exofastv2/examples/hat3/*.dat" 
    rvpath = "../data/exofastv2/examples/hat3/*.rv"
    sedfile = "../data/exofastv2/examples/hat3/HAT-3.sed"
    
    try:
        event, chi2_func = build_simple_model(
            parfile=parfile,
            tranpath=tranpath,
            rvpath=rvpath, 
            sedfile=sedfile,
            debug=True
        )
        
        # Test with default parameters
        default_params = create_default_parameters(event)
        chi2_value = chi2_func(default_params)
        
        print(f"\nTest Results:")
        print(f"Chi-squared with default parameters: {chi2_value:.3f}")
        print(f"Model components loaded successfully!")
        
        return event, chi2_func
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None, None


if __name__ == "__main__":
    test_simple_model()