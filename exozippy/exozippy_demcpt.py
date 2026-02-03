"""
Python translation of EXOFAST_DEMCPT_MULTI.PRO
Differential Evolution Markov Chain Monte Carlo with Parallel Tempering

Translated from IDL implementation by Jason Eastman (EXOFASTv2)
Python version for EXOZIPPy

Author: Python translation for EXOZIPPy
Original: Jason Eastman
"""

import numpy as np
import time
import threading
from typing import Callable, Optional, Tuple, Dict, Any, List
import pickle
import h5py
from pathlib import Path

class EXOZIPPyDEMCPT:
    """
    Differential Evolution Markov Chain Monte Carlo with Parallel Tempering
    
    Python translation of the EXOFASTv2 DEMC-PT algorithm for robust 
    parameter estimation and uncertainty quantification.
    """
    
    def __init__(self, 
                 chi2func: Callable,
                 bestpars: Dict[str, float],
                 tofit: Optional[List[str]] = None,
                 scale: Optional[Dict[str, float]] = None,
                 nchains: int = 8,
                 ntemps: int = 8,
                 tf: float = 200.0,
                 maxsteps: int = 100000,
                 nthin: int = 1,
                 maxtime: Optional[float] = None,
                 maxgr: float = 1.01,
                 mintz: int = 1000,
                 burnndx: Optional[int] = None,
                 dontstop: bool = False,
                 nthreads: int = 1,
                 stretch: bool = False,
                 keephot: bool = False,
                 logname: Optional[str] = None,
                 debug: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize DEMC-PT sampler
        
        Parameters:
        -----------
        chi2func : callable
            Function to calculate chi-squared given parameters
        bestpars : dict
            Dictionary of best-fit parameters
        tofit : list, optional  
            List of parameter names to fit
        scale : dict, optional
            Step sizes for each parameter
        nchains : int
            Number of independent chains per temperature
        ntemps : int
            Number of temperature levels
        tf : float
            Temperature factor for parallel tempering ladder
        maxsteps : int
            Maximum number of MCMC steps
        nthin : int
            Thinning factor for output
        maxtime : float, optional
            Maximum runtime in seconds
        maxgr : float
            Maximum Gelman-Rubin statistic for convergence
        mintz : int
            Minimum independent samples for convergence test
        burnndx : int, optional
            Burn-in index (auto-determined if None)
        dontstop : bool
            Continue even if converged
        nthreads : int
            Number of threads for parallel computation
        stretch : bool
            Use stretch move instead of differential evolution
        keephot : bool
            Keep only hottest temperature chain
        logname : str, optional
            Log file name
        debug : bool
            Enable debug output
        seed : int, optional
            Random seed
        """
        
        self.chi2func = chi2func
        self.bestpars = bestpars.copy()
        self.nchains = nchains
        self.ntemps = ntemps
        self.tf = tf
        self.maxsteps = maxsteps
        self.nthin = nthin
        self.maxtime = maxtime
        self.maxgr = maxgr
        self.mintz = mintz
        self.burnndx = burnndx
        self.dontstop = dontstop
        self.nthreads = nthreads
        self.stretch = stretch
        self.keephot = keephot
        self.logname = logname
        self.debug = debug
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            
        # Determine which parameters to fit
        if tofit is None:
            self.tofit = list(bestpars.keys())
        else:
            self.tofit = tofit
            
        self.nfit = len(self.tofit)
        
        # Set up parameter scales
        if scale is None:
            self.scale = {key: 0.01 for key in self.tofit}
        else:
            self.scale = scale.copy()
            
        # Temperature ladder
        self.temps = self.tf ** (np.arange(self.ntemps) / (self.ntemps - 1))
        self.betas = 1.0 / self.temps
        
        # Initialize storage arrays
        self.pars = np.zeros((self.ntemps, self.nchains, self.nfit, self.maxsteps))
        self.chi2 = np.full((self.ntemps, self.nchains, self.maxsteps), np.inf)
        self.lnprob = np.zeros((self.ntemps, self.nchains, self.maxsteps))
        
        # Tracking variables
        self.naccept = np.zeros((self.ntemps, self.nchains))
        self.nswap = np.zeros(self.ntemps - 1)
        self.nswaptries = np.zeros(self.ntemps - 1)
        self.tz = 0
        self.converged = False
        
        # Threading setup
        if self.nthreads > 1:
            self.use_threading = True
        else:
            self.use_threading = False
            
    def initialize_chains(self) -> None:
        """Initialize all chains with random starting positions"""
        
        if self.debug:
            print(f"Initializing {self.ntemps} temperatures × {self.nchains} chains")
            
        # Convert bestpars to array for easier manipulation
        self.bestpars_array = np.array([self.bestpars[key] for key in self.tofit])
        
        for t in range(self.ntemps):
            for c in range(self.nchains):
                # Random starting position around best fit
                for i, key in enumerate(self.tofit):
                    scale = self.scale[key]
                    self.pars[t, c, i, 0] = (self.bestpars[key] + 
                                           np.random.normal(0, scale))
                
                # Calculate initial chi-squared
                pars_dict = {self.tofit[i]: self.pars[t, c, i, 0] 
                           for i in range(self.nfit)}
                self.chi2[t, c, 0] = self.chi2func(pars_dict)
                self.lnprob[t, c, 0] = -0.5 * self.chi2[t, c, 0] * self.betas[t]
                
    def differential_evolution_step(self, temp: int, chain: int, step: int) -> Tuple[np.ndarray, bool]:
        """
        Generate proposal using differential evolution
        
        Parameters:
        -----------
        temp : int
            Temperature index
        chain : int
            Chain index  
        step : int
            Current step
            
        Returns:
        --------
        proposal : ndarray
            Proposed parameters
        valid : bool
            Whether proposal is valid
        """
        
        current_pars = self.pars[temp, chain, :, step-1]
        
        if self.stretch:
            # Affine invariant stretch move
            # Choose random chain (not current one)
            other_chains = [c for c in range(self.nchains) if c != chain]
            other_chain = np.random.choice(other_chains)
            other_pars = self.pars[temp, other_chain, :, step-1]
            
            # Generate stretch factor
            a = 2.0  # stretch parameter
            z = ((a - 1.0) * np.random.random() + 1.0) ** 2 / a
            
            proposal = other_pars + z * (current_pars - other_pars)
            
        else:
            # Differential evolution
            # Choose two random chains (different from current)
            other_chains = [c for c in range(self.nchains) if c != chain]
            if len(other_chains) < 2:
                return current_pars, False
                
            chain1, chain2 = np.random.choice(other_chains, 2, replace=False)
            
            # DE/rand/1 scheme
            gamma = 2.38 / np.sqrt(2 * self.nfit)  # Optimal scaling
            pars1 = self.pars[temp, chain1, :, step-1] 
            pars2 = self.pars[temp, chain2, :, step-1]
            
            proposal = current_pars + gamma * (pars1 - pars2)
            
            # Add random perturbation to avoid getting stuck (increased for better exploration)
            for i in range(self.nfit):
                proposal[i] += np.random.normal(0, self.scale[self.tofit[i]] * 0.1)
        
        return proposal, True
    
    def evaluate_chi2(self, pars_array: np.ndarray) -> float:
        """
        Evaluate chi-squared for parameter array
        
        Parameters:
        -----------
        pars_array : ndarray
            Parameter values
            
        Returns:
        --------
        chi2 : float
            Chi-squared value
        """
        
        pars_dict = {self.tofit[i]: pars_array[i] for i in range(self.nfit)}
        return self.chi2func(pars_dict)
    
    def metropolis_hastings_step(self, temp: int, chain: int, step: int) -> None:
        """
        Perform Metropolis-Hastings accept/reject step
        
        Parameters:
        -----------
        temp : int
            Temperature index
        chain : int
            Chain index
        step : int
            Current step
        """
        
        # Generate proposal
        proposal, valid = self.differential_evolution_step(temp, chain, step)
        
        if not valid:
            # Keep current state
            self.pars[temp, chain, :, step] = self.pars[temp, chain, :, step-1]
            self.chi2[temp, chain, step] = self.chi2[temp, chain, step-1]
            self.lnprob[temp, chain, step] = self.lnprob[temp, chain, step-1]
            return
            
        # Evaluate proposal
        proposal_chi2 = self.evaluate_chi2(proposal)
        proposal_lnprob = -0.5 * proposal_chi2 * self.betas[temp]
        
        # Metropolis-Hastings ratio
        current_lnprob = self.lnprob[temp, chain, step-1]
        ln_alpha = proposal_lnprob - current_lnprob
        
        # Accept/reject
        if ln_alpha > 0 or np.log(np.random.random()) < ln_alpha:
            # Accept
            self.pars[temp, chain, :, step] = proposal
            self.chi2[temp, chain, step] = proposal_chi2
            self.lnprob[temp, chain, step] = proposal_lnprob
            self.naccept[temp, chain] += 1
        else:
            # Reject - keep current state
            self.pars[temp, chain, :, step] = self.pars[temp, chain, :, step-1]
            self.chi2[temp, chain, step] = self.chi2[temp, chain, step-1]
            self.lnprob[temp, chain, step] = self.lnprob[temp, chain, step-1]
    
    def parallel_tempering_swap(self, step: int) -> None:
        """
        Attempt parallel tempering swaps between adjacent temperatures
        
        Parameters:
        -----------
        step : int
            Current step
        """
        
        for t in range(self.ntemps - 1):
            # Randomly select chain to swap
            chain = np.random.randint(self.nchains)
            
            self.nswaptries[t] += 1
            
            # Calculate swap probability
            chi2_cold = self.chi2[t, chain, step]
            chi2_hot = self.chi2[t+1, chain, step]
            
            delta_beta = self.betas[t] - self.betas[t+1]
            ln_swap_prob = delta_beta * (chi2_hot - chi2_cold) / 2.0
            
            # Accept swap?
            if ln_swap_prob > 0 or np.log(np.random.random()) < ln_swap_prob:
                # Swap states
                self.pars[t, chain, :, step], self.pars[t+1, chain, :, step] = \
                    self.pars[t+1, chain, :, step].copy(), self.pars[t, chain, :, step].copy()
                    
                self.chi2[t, chain, step], self.chi2[t+1, chain, step] = \
                    self.chi2[t+1, chain, step], self.chi2[t, chain, step]
                    
                self.lnprob[t, chain, step], self.lnprob[t+1, chain, step] = \
                    self.lnprob[t+1, chain, step], self.lnprob[t, chain, step]
                    
                self.nswap[t] += 1
    
    def gelman_rubin_test(self, step: int) -> Tuple[float, int]:
        """
        Calculate Gelman-Rubin convergence diagnostic
        
        Parameters:
        -----------
        step : int
            Current step
            
        Returns:
        --------
        max_gr : float
            Maximum G-R statistic across parameters
        tz : int
            Number of independent samples
        """
        
        if step < 100:
            return np.inf, 0
            
        # Use cold temperature chains only
        temp = 0
        
        # Calculate for each parameter
        gr_stats = []
        
        for param_idx in range(self.nfit):
            chains_data = self.pars[temp, :, param_idx, :step+1]
            
            # Skip if insufficient data
            if step < 50:
                gr_stats.append(np.inf)
                continue
                
            # Calculate between and within chain variance
            chain_means = np.mean(chains_data, axis=1)
            overall_mean = np.mean(chain_means)
            
            # Between chain variance
            B = step * np.var(chain_means, ddof=1)
            
            # Within chain variance
            chain_vars = np.var(chains_data, axis=1, ddof=1)
            W = np.mean(chain_vars)
            
            # Gelman-Rubin statistic
            if W > 0:
                gr = np.sqrt(((step - 1) * W + B) / (step * W))
            else:
                gr = np.inf
                
            gr_stats.append(gr)
        
        max_gr = np.max(gr_stats) if gr_stats else np.inf
        
        # Estimate number of independent samples
        tz = min(step // 2, int(step / max_gr)) if max_gr > 1 else step
        
        return max_gr, tz
    
    def run(self) -> Dict[str, Any]:
        """
        Run the MCMC sampler
        
        Returns:
        --------
        results : dict
            Dictionary containing chains, chi2 values, and metadata
        """
        
        start_time = time.time()
        
        # Initialize chains
        self.initialize_chains()
        
        if self.debug:
            print(f"Starting MCMC with {self.ntemps} temperatures, {self.nchains} chains")
            print(f"Temperature ladder: {self.temps}")
            
        # Main MCMC loop
        for step in range(1, self.maxsteps):
            
            # MCMC steps for all chains
            if self.use_threading and self.nthreads > 1:
                # Threaded execution
                threads = []
                for t in range(self.ntemps):
                    for c in range(self.nchains):
                        thread = threading.Thread(
                            target=self.metropolis_hastings_step,
                            args=(t, c, step)
                        )
                        threads.append(thread)
                        thread.start()
                        
                        # Limit concurrent threads
                        if len(threads) >= self.nthreads:
                            for thread in threads:
                                thread.join()
                            threads = []
                            
                # Wait for remaining threads
                for thread in threads:
                    thread.join()
            else:
                # Sequential execution
                for t in range(self.ntemps):
                    for c in range(self.nchains):
                        self.metropolis_hastings_step(t, c, step)
            
            # Parallel tempering swaps
            if step % 10 == 0:  # Swap every 10 steps
                self.parallel_tempering_swap(step)
            
            # Check convergence
            if step % 100 == 0:
                max_gr, tz = self.gelman_rubin_test(step)
                self.tz = tz
                
                if self.debug and step % 1000 == 0:
                    elapsed = time.time() - start_time
                    accept_rate = np.mean(self.naccept[:, :] / step) * 100
                    swap_rate = np.mean(self.nswap / np.maximum(self.nswaptries, 1)) * 100
                    
                    # Find best (minimum) chi² across all chains and temperatures up to current step
                    best_chi2 = np.min(self.chi2[:, :, :step+1])
                    
                    print(f"Step {step:6d}: G-R = {max_gr:.4f}, "
                          f"tz = {tz:4d}, accept = {accept_rate:.1f}%, "
                          f"swap = {swap_rate:.1f}%, time = {elapsed:.1f}s, "
                          f"best_chi2 = {best_chi2:.2f}")
                
                # Check convergence
                if not self.dontstop and max_gr < self.maxgr and tz > self.mintz:
                    if self.debug:
                        print(f"Converged at step {step}")
                    self.converged = True
                    break
                    
                # Check time limit
                if self.maxtime and (time.time() - start_time) > self.maxtime:
                    if self.debug:
                        print(f"Time limit reached at step {step}")
                    break
        
        # Determine burn-in
        if self.burnndx is None:
            self.burnndx = max(step // 4, 100)  # Burn first 25% or 100 steps
            
        # Prepare results
        final_step = min(step, self.maxsteps - 1)
        
        results = {
            'pars': self.pars[:, :, :, :final_step+1],
            'chi2': self.chi2[:, :, :final_step+1],
            'lnprob': self.lnprob[:, :, :final_step+1],
            'tofit': self.tofit,
            'temps': self.temps,
            'betas': self.betas,
            'nsteps': final_step + 1,
            'burnndx': self.burnndx,
            'converged': self.converged,
            'tz': self.tz,
            'max_gr': max_gr if 'max_gr' in locals() else np.inf,
            'naccept': self.naccept,
            'nswap': self.nswap,
            'nswaptries': self.nswaptries,
            'runtime': time.time() - start_time
        }
        
        if self.debug:
            print(f"MCMC completed in {results['runtime']:.1f} seconds")
            print(f"Final acceptance rate: {np.mean(self.naccept / final_step * 100):.1f}%")
            
        return results
    
    def save_chains(self, filename: str, format: str = 'hdf5') -> None:
        """
        Save MCMC chains to file
        
        Parameters:
        -----------
        filename : str
            Output filename
        format : str
            File format ('hdf5', 'npz', 'pickle')
        """
        
        # Prepare data to save
        save_data = {
            'pars': self.pars,
            'chi2': self.chi2,
            'lnprob': self.lnprob,
            'tofit': self.tofit,
            'temps': self.temps,
            'betas': self.betas,
            'burnndx': self.burnndx,
            'converged': self.converged,
            'tz': self.tz,
            'naccept': self.naccept,
            'nswap': self.nswap,
            'nswaptries': self.nswaptries,
            'bestpars': self.bestpars,
            'scale': self.scale,
            'maxsteps': self.maxsteps,
            'nchains': self.nchains,
            'ntemps': self.ntemps
        }
        
        filepath = Path(filename)
        
        if format.lower() == 'hdf5':
            self._save_hdf5(filepath.with_suffix('.h5'), save_data)
        elif format.lower() == 'npz':
            self._save_npz(filepath.with_suffix('.npz'), save_data)
        elif format.lower() == 'pickle':
            self._save_pickle(filepath.with_suffix('.pkl'), save_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        if self.debug:
            print(f"Chains saved to {filepath}")
    
    def _save_hdf5(self, filepath: Path, data: Dict) -> None:
        """Save to HDF5 format (recommended for large chains)"""
        with h5py.File(filepath, 'w') as f:
            # Save arrays
            f.create_dataset('pars', data=data['pars'], compression='gzip')
            f.create_dataset('chi2', data=data['chi2'], compression='gzip')  
            f.create_dataset('lnprob', data=data['lnprob'], compression='gzip')
            f.create_dataset('temps', data=data['temps'])
            f.create_dataset('betas', data=data['betas'])
            f.create_dataset('naccept', data=data['naccept'])
            f.create_dataset('nswap', data=data['nswap'])
            f.create_dataset('nswaptries', data=data['nswaptries'])
            
            # Save scalars
            f.attrs['burnndx'] = data['burnndx']
            f.attrs['converged'] = data['converged'] 
            f.attrs['tz'] = data['tz']
            f.attrs['maxsteps'] = data['maxsteps']
            f.attrs['nchains'] = data['nchains']
            f.attrs['ntemps'] = data['ntemps']
            
            # Save string lists and dicts
            f.create_dataset('tofit', data=[s.encode() for s in data['tofit']])
            
            # Save parameter dictionaries
            bestpars_grp = f.create_group('bestpars')
            for key, val in data['bestpars'].items():
                bestpars_grp.attrs[key] = val
                
            scale_grp = f.create_group('scale')  
            for key, val in data['scale'].items():
                scale_grp.attrs[key] = val
    
    def _save_npz(self, filepath: Path, data: Dict) -> None:
        """Save to NumPy NPZ format"""
        # Convert string lists and dicts to saveable format
        tofit_array = np.array(data['tofit'], dtype='U50')
        bestpars_keys = np.array(list(data['bestpars'].keys()), dtype='U50')
        bestpars_vals = np.array(list(data['bestpars'].values()))
        scale_keys = np.array(list(data['scale'].keys()), dtype='U50') 
        scale_vals = np.array(list(data['scale'].values()))
        
        np.savez_compressed(
            filepath,
            pars=data['pars'],
            chi2=data['chi2'],
            lnprob=data['lnprob'],
            temps=data['temps'],
            betas=data['betas'],
            tofit=tofit_array,
            bestpars_keys=bestpars_keys,
            bestpars_vals=bestpars_vals,
            scale_keys=scale_keys,
            scale_vals=scale_vals,
            naccept=data['naccept'],
            nswap=data['nswap'],
            nswaptries=data['nswaptries'],
            burnndx=data['burnndx'],
            converged=data['converged'],
            tz=data['tz'],
            maxsteps=data['maxsteps'],
            nchains=data['nchains'],
            ntemps=data['ntemps']
        )
    
    def _save_pickle(self, filepath: Path, data: Dict) -> None:
        """Save to Pickle format (preserves all Python objects)"""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_chains(filename: str) -> Dict[str, Any]:
    """
    Load MCMC chains from file
    
    Parameters:
    -----------
    filename : str
        Input filename
        
    Returns:
    --------
    data : dict
        Loaded chain data
    """
    
    filepath = Path(filename)
    
    if filepath.suffix == '.h5':
        return _load_hdf5(filepath)
    elif filepath.suffix == '.npz':
        return _load_npz(filepath)
    elif filepath.suffix == '.pkl':
        return _load_pickle(filepath)
    else:
        # Try to auto-detect
        if filepath.with_suffix('.h5').exists():
            return _load_hdf5(filepath.with_suffix('.h5'))
        elif filepath.with_suffix('.npz').exists():
            return _load_npz(filepath.with_suffix('.npz'))
        elif filepath.with_suffix('.pkl').exists():
            return _load_pickle(filepath.with_suffix('.pkl'))
        else:
            raise FileNotFoundError(f"No chain file found for {filename}")


def _load_hdf5(filepath: Path) -> Dict[str, Any]:
    """Load from HDF5 format"""
    with h5py.File(filepath, 'r') as f:
        data = {}
        
        # Load arrays
        data['pars'] = f['pars'][:]
        data['chi2'] = f['chi2'][:]
        data['lnprob'] = f['lnprob'][:]
        data['temps'] = f['temps'][:]
        data['betas'] = f['betas'][:]
        data['naccept'] = f['naccept'][:]
        data['nswap'] = f['nswap'][:]
        data['nswaptries'] = f['nswaptries'][:]
        
        # Load scalars
        data['burnndx'] = f.attrs['burnndx']
        data['converged'] = f.attrs['converged']
        data['tz'] = f.attrs['tz']
        data['maxsteps'] = f.attrs['maxsteps'] 
        data['nchains'] = f.attrs['nchains']
        data['ntemps'] = f.attrs['ntemps']
        
        # Load string lists
        data['tofit'] = [s.decode() for s in f['tofit'][:]]
        
        # Load parameter dictionaries
        data['bestpars'] = dict(f['bestpars'].attrs)
        data['scale'] = dict(f['scale'].attrs)
        
    return data


def _load_npz(filepath: Path) -> Dict[str, Any]:
    """Load from NumPy NPZ format"""
    with np.load(filepath) as f:
        data = {}
        
        # Load arrays
        data['pars'] = f['pars']
        data['chi2'] = f['chi2']
        data['lnprob'] = f['lnprob']
        data['temps'] = f['temps']
        data['betas'] = f['betas']
        data['naccept'] = f['naccept']
        data['nswap'] = f['nswap']
        data['nswaptries'] = f['nswaptries']
        
        # Load scalars
        data['burnndx'] = int(f['burnndx'])
        data['converged'] = bool(f['converged'])
        data['tz'] = int(f['tz'])
        data['maxsteps'] = int(f['maxsteps'])
        data['nchains'] = int(f['nchains'])
        data['ntemps'] = int(f['ntemps'])
        
        # Load parameter lists and dicts
        data['tofit'] = f['tofit'].tolist()
        data['bestpars'] = dict(zip(f['bestpars_keys'], f['bestpars_vals']))
        data['scale'] = dict(zip(f['scale_keys'], f['scale_vals']))
        
    return data


def _load_pickle(filepath: Path) -> Dict[str, Any]:
    """Load from Pickle format"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def exozippy_demcpt(chi2func: Callable,
                    bestpars: Dict[str, float],
                    backend: str = 'exozippy',
                    **kwargs) -> Dict[str, Any]:
    """
    Convenience function for running MCMC with different backends
    
    Parameters:
    -----------
    chi2func : callable
        Function to calculate chi-squared
    bestpars : dict
        Best-fit parameters
    backend : str
        MCMC backend to use ('exozippy' or 'emcee')
    **kwargs
        Additional arguments passed to the sampler
        
    Returns:
    --------
    results : dict
        MCMC results
    """
    
    if backend.lower() == 'emcee':
        return _run_emcee_de(chi2func, bestpars, **kwargs)
    else:
        sampler = EXOZIPPyDEMCPT(chi2func, bestpars, **kwargs)
        return sampler.run()


def _run_emcee_de(chi2func: Callable,
                  bestpars: Dict[str, float],
                  nchains: int = 32, 
                  maxsteps: int = 10000,
                  scale: Dict[str, float] = None, 
                  debug: bool = False, 
                  seed: int = None,
                  **kwargs) -> Dict[str, Any]:
    """
    Run MCMC using emcee with Differential Evolution moves
    
    Parameters:
    -----------
    chi2func : callable
        Function that calculates chi-squared given parameter dictionary
    bestpars : dict
        Dictionary of best-fit parameters
    nchains : int
        Number of walkers (chains)
    maxsteps : int
        Maximum number of steps
    scale : dict
        Parameter step sizes (used for walker initialization)
    debug : bool
        Enable debug output
    seed : int
        Random seed
    **kwargs : dict
        Additional arguments (ignored for emcee compatibility)
        
    Returns:
    --------
    results : dict
        MCMC results in EXOZIPPy-compatible format
    """
    
    try:
        import emcee
        from emcee.moves import DEMove
    except ImportError:
        raise ImportError("emcee is required for the 'emcee' backend. Install with: pip install emcee")
    
    import numpy as np
    import time
    
    if seed is not None:
        np.random.seed(seed)
    
    # Convert bestpars to arrays for emcee
    tofit = list(bestpars.keys())
    ndim = len(tofit)
    
    def log_prob(pars_array):
        """Convert chi2 to log probability for emcee"""
        pars_dict = {tofit[i]: pars_array[i] for i in range(ndim)}
        try:
            chi2 = chi2func(pars_dict)
            return -0.5 * chi2
        except:
            return -np.inf
    
    # Initialize walker positions around best fit
    if scale is None:
        # Default scale - 1% of parameter value
        scale = {param: abs(val) * 0.01 if val != 0 else 0.01 
                for param, val in bestpars.items()}
    
    # Convert to array format
    bestpars_array = np.array([bestpars[param] for param in tofit])
    scale_array = np.array([scale.get(param, 0.01) for param in tofit])
    
    # Initialize walkers with small perturbations around best fit
    pos = bestpars_array + np.random.normal(0, scale_array, (nchains, ndim))
    
    # Set up emcee with DE moves
    moves = [DEMove()]  # Use Differential Evolution moves
    sampler = emcee.EnsembleSampler(nchains, ndim, log_prob, moves=moves)
    
    if debug:
        print(f"Starting emcee MCMC with {nchains} walkers, {ndim} parameters")
        print(f"Using Differential Evolution moves")
    
    start_time = time.time()
    
    # Initialize variables for convergence tracking
    autocorr_times = []
    
    # Run MCMC with emcee's built-in progress bar
    if debug:
        print("Running MCMC with autocorrelation-based convergence monitoring...")
        
        # We'll implement our own progress monitoring with autocorr analysis
        old_tau = np.inf
        converged = False
        
        # Start MCMC
        for sample in sampler.sample(pos, iterations=maxsteps, progress=True):
            # Check convergence every 1000 steps
            if sampler.iteration % 1000:
                continue
                
            # Compute the autocorrelation time so far
            try:
                tau = sampler.get_autocorr_time(tol=0)
                autocorr_times.append(tau.copy())
                
                # Check convergence - we need:
                # 1. Chain length > 50 * tau
                # 2. tau hasn't changed much
                converged_chain_length = np.all(sampler.iteration > 50 * tau)
                converged_tau_stable = np.all(np.abs(old_tau - tau) / tau < 0.01)
                
                if converged_chain_length and converged_tau_stable:
                    converged = True
                    if debug:
                        print(f"\n🎉 Convergence achieved at step {sampler.iteration}!")
                        print(f"   Autocorr times: {tau}")
                        print(f"   Chain length / tau: {sampler.iteration / np.max(tau):.1f}")
                    break
                    
                old_tau = tau
                
                # Progress update
                accept_rate = np.mean(sampler.acceptance_fraction) * 100
                elapsed = time.time() - start_time
                print(f"\nStep {sampler.iteration:6d}: accept = {accept_rate:.1f}%, "
                      f"time = {elapsed:.1f}s")
                print(f"   Autocorr times: {tau}")
                print(f"   Chain length / max(tau): {sampler.iteration / np.max(tau):.1f}")
                
            except Exception as e:
                # Autocorr time estimation can fail early in the run
                if sampler.iteration < 100:
                    continue
                else:
                    if debug:
                        print(f"   Autocorr time estimation failed: {e}")
        
        if not converged:
            print(f"\n⚠️  MCMC completed {maxsteps} steps without full convergence")
            print("   Consider running longer or checking your model")
            
    else:
        # Simple run without detailed monitoring
        sampler.run_mcmc(pos, maxsteps, progress=True)
    
    runtime = time.time() - start_time
    
    if debug:
        print(f"emcee MCMC completed in {runtime:.1f} seconds")
        print(f"Final acceptance rate: {np.mean(sampler.acceptance_fraction)*100:.1f}%")
    
    # Convert emcee results to EXOZIPPy format
    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    chi2_chain = -2 * sampler.get_log_prob()  # Convert back to chi2
    
    # Reshape to EXOZIPPy format (add temperature dimension)
    pars_exozippy = np.transpose(chain, (1, 2, 0))[np.newaxis, :, :, :]  # (1, nwalkers, ndim, nsteps)
    chi2_exozippy = chi2_chain.T[np.newaxis, :, :]  # (1, nwalkers, nsteps)
    lnprob_exozippy = sampler.get_log_prob().T[np.newaxis, :, :]
    
    # Calculate burn-in (use first 25% as burn-in)
    burnndx = maxsteps // 4
    
    # Calculate convergence statistics using autocorrelation analysis
    try:
        # Get final autocorr time
        final_tau = sampler.get_autocorr_time(quiet=True)
        
        # Calculate effective sample size (independent samples)
        # After burn-in, we have (maxsteps - burnndx) samples
        # But they're correlated with timescale tau
        # So effective samples ≈ (maxsteps - burnndx) / (2 * tau)
        eff_samples_per_param = (sampler.iteration - burnndx) / (2 * final_tau)
        tz = int(np.min(eff_samples_per_param))  # Most conservative estimate
        
        # Convergence assessment:
        # 1. Did we run long enough? (> 50 * tau)
        # 2. Are autocorr times reasonable?
        chain_length_adequate = np.all(sampler.iteration > 50 * final_tau)
        tau_reasonable = np.all(final_tau < sampler.iteration / 10)  # tau shouldn't be > 10% of chain
        
        # Use a convergence metric based on autocorr time stability
        if len(autocorr_times) > 1:
            # Compare last few tau estimates
            recent_tau = np.array(autocorr_times[-3:]) if len(autocorr_times) >= 3 else autocorr_times
            tau_variation = np.std(recent_tau, axis=0) / np.mean(recent_tau, axis=0)
            max_gr = np.max(tau_variation)  # Use tau stability as convergence metric
        else:
            max_gr = np.max(final_tau) / sampler.iteration  # Fraction of chain length
            
        # Overall convergence
        if debug:
            converged_overall = chain_length_adequate and tau_reasonable and (max_gr < 0.1)
        else:
            # If we ran without monitoring, use simpler criteria
            converged_overall = chain_length_adequate and tau_reasonable
        
        if debug:
            print(f"\n=== Convergence Analysis ===")
            print(f"Final autocorr times: {final_tau}")
            print(f"Chain length / tau: {sampler.iteration / final_tau}")
            print(f"Chain length adequate (>50τ): {chain_length_adequate}")
            print(f"Tau reasonable (<10% chain): {tau_reasonable}")
            print(f"Effective samples per param: {eff_samples_per_param}")
            print(f"Convergence metric: {max_gr:.4f}")
            
    except Exception as e:
        if debug:
            print(f"Autocorr analysis failed: {e}")
        # Fallback to simple estimates
        final_tau = np.ones(ndim) * 10  # Conservative estimate
        max_gr = 1.0
        tz = (sampler.iteration - burnndx) // 10  # Conservative
        converged_overall = sampler.iteration > 1000  # Simple criterion
    
    results = {
        'pars': pars_exozippy,
        'chi2': chi2_exozippy, 
        'lnprob': lnprob_exozippy,
        'tofit': tofit,
        'temps': np.array([1.0]),  # Only one temperature for emcee
        'betas': np.array([1.0]),
        'nsteps': maxsteps,
        'burnndx': burnndx,
        'converged': converged_overall,
        'tz': tz,
        'max_gr': max_gr,
        'naccept': sampler.acceptance_fraction.sum().reshape(1, -1),  # Reshape for compatibility
        'runtime': runtime,
        'backend': 'emcee'
    }
    
    return results


# Example usage and testing
if __name__ == "__main__":
    
    # Simple test case - 1D Gaussian
    def chi2_gaussian(pars):
        x = pars['x']
        return (x - 3.0) ** 2  # Target at x = 3
        
    bestpars = {'x': 2.5}
    scale = {'x': 0.1}
    
    results = exozippy_demcpt(
        chi2_gaussian,
        bestpars,
        scale=scale,
        nchains=4,
        ntemps=4,
        maxsteps=5000,
        debug=True,
        seed=42
    )
    
    # Analyze results
    cold_chains = results['pars'][0, :, 0, results['burnndx']:]  # Cold temp, all chains, param 0, post burn-in
    samples = cold_chains.flatten()
    
    print(f"\nResults:")
    print(f"True value: 3.0")
    print(f"Recovered mean: {np.mean(samples):.3f}")
    print(f"Recovered std: {np.std(samples):.3f}")
    print(f"Converged: {results['converged']}")