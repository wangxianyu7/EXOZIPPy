"""
High-level driver for EXOZIPPy (analogous to EXOFASTv2.pro).
Provides an all-in-one interface: read priors, run optimizer,
optional MCMC, and emit plots/logs.
"""

from pathlib import Path
from datetime import datetime
import argparse
import numpy as np

from .fit_exoplanet import fit_exoplanet, run_mcmc
from .massradius_mist import plot_mist_track
from .plottran import plottran
from .plotrv import plotrv
from .plotsed import plotsed


def _log(msg, verbose=True):
    if verbose:
        print(msg)


def _log_section(title, verbose=True):
    if verbose:
        bar = '=' * len(title)
        print(f'\n{bar}\n{title}\n{bar}')


def exozippy(
    parfile,
    tranpath,
    rvpath,
    sedfile=None,
    prefix='fitresults/planet.',
    circular=True,
    nomist=False,
    run_mcmc_flag=False,
    mcmc_steps=2000,
    mcmc_burn=None,
    mcmc_walkers=32,
    mcmc_threads=None,
    mcmc_backend='emcee',
    verbose=True,
    **kwargs,
):
    """
    Parameters mirror the EXOFASTv2 interface; unused keywords are accepted
    for compatibility but ignored in this lightweight Python port.
    """
    use_mist = not nomist
    if sedfile is None:
        mistsed = kwargs.get('mistsedfile')
        fluxfile = kwargs.get('fluxfile')
        sedfile = mistsed or fluxfile
    if sedfile is None:
        raise ValueError("sedfile (or mistsedfile/fluxfile) must be provided")

    parfile = Path(parfile).expanduser().resolve()
    tranpath = Path(tranpath).expanduser().resolve()
    rvpath = Path(rvpath).expanduser().resolve()
    sedfile = Path(sedfile).expanduser().resolve()

    prefix_path = Path(prefix).expanduser()
    if prefix_path.is_dir() or str(prefix_path).endswith('/'):
        prefix_path.mkdir(parents=True, exist_ok=True)
        prefix_path = prefix_path / 'planet.'
    else:
        prefix_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime.utcnow()
    _log_section('EXOZIPPy Global Fit', verbose)
    _log(f'Start time (UTC): {start_time:%Y-%m-%d %H:%M:%S}', verbose)
    _log(f'Prior file       : {parfile}', verbose)
    _log(f'Transit file     : {tranpath}', verbose)
    _log(f'RV file          : {rvpath}', verbose)
    _log(f'SED file         : {sedfile}', verbose)
    _log(f'Output prefix    : {prefix_path}', verbose)
    _log(f'Circular orbit   : {circular}', verbose)
    _log(f'Use MIST         : {use_mist}', verbose)

    e = 0.0 if circular else 0.0  # placeholder until eccentric support added
    omega = np.pi / 2

    bestfit = fit_exoplanet(
        str(parfile), str(tranpath), str(rvpath), str(sedfile),
        e=e, omega=omega, verbose=verbose, use_mist=use_mist,
    )

    samples = None
    if run_mcmc_flag and mcmc_steps and mcmc_steps > 0:
        _log_section('Running MCMC', verbose)
        nburn = mcmc_burn if mcmc_burn is not None else max(mcmc_steps // 5, 1)
        _log(f'Walkers : {mcmc_walkers}', verbose)
        _log(f'Steps   : {mcmc_steps}', verbose)
        _log(f'Burn-in : {nburn}', verbose)
        samples, labels, summary = run_mcmc(
            str(parfile), str(tranpath), str(rvpath), str(sedfile),
            bestfit=bestfit, e=e, omega=omega,
            nwalkers=mcmc_walkers, nsteps=mcmc_steps, nburn=nburn,
            verbose=verbose, use_mist=use_mist, nthreads=mcmc_threads,
            backend=mcmc_backend,
        )
        np.savez(
            f'{prefix_path}mcmc_samples.npz',
            samples=samples, labels=labels,
            **{k: np.array(v) for k, v in summary.items()},
        )
        _log(f"Saved posterior samples to {prefix_path}mcmc_samples.npz", verbose)

    _log_section('Generating Plots', verbose)
    tran_png = f'{prefix_path}transit.png'
    rv_png = f'{prefix_path}rv.png'
    sed_png = f'{prefix_path}sed.png'
    plottran(str(tranpath), bestfit, samples=samples, e=e, omega=omega, outfile=tran_png)
    plotrv(str(rvpath), bestfit, samples=samples, e=e, omega=omega, outfile=rv_png)
    plotsed(str(sedfile), bestfit, outfile=sed_png)
    _log(f'Transit plot: {tran_png}', verbose)
    _log(f'RV plot     : {rv_png}', verbose)
    _log(f'SED plot    : {sed_png}', verbose)
    if use_mist:
        mist_png = f'{prefix_path}mist.png'
        plot_mist_track(bestfit, outfile=mist_png)
        _log(f'MIST plot   : {mist_png}', verbose)

    end_time = datetime.utcnow()
    _log_section('Done', verbose)
    _log(f'End time (UTC): {end_time:%Y-%m-%d %H:%M:%S}', verbose)
    _log(f'Elapsed       : {end_time - start_time}', verbose)
    return bestfit, samples


def _cli():
    parser = argparse.ArgumentParser(description='EXOZIPPy global fitting driver')
    parser.add_argument('--prior', required=True, help='Path to prior file')
    parser.add_argument('--tran', required=True, help='Transit light curve file')
    parser.add_argument('--rv', required=True, help='Radial velocity file')
    parser.add_argument('--sed', help='SED definition file')
    parser.add_argument('--prefix', default='fitresults/planet.', help='Output prefix')
    parser.add_argument('--nomist', action='store_true', help='Disable MIST evolutionary prior')
    parser.add_argument('--noncircular', action='store_true', help='Allow eccentric orbit (placeholder)')
    parser.add_argument('--mcmc', action='store_true', help='Run MCMC after optimizer')
    parser.add_argument('--steps', type=int, default=2000, help='MCMC steps per walker')
    parser.add_argument('--burn', type=int, help='MCMC burn-in (defaults to 20%% of steps)')
    parser.add_argument('--walkers', type=int, default=32, help='Number of MCMC walkers')
    parser.add_argument('--workers', type=int, default=None, help='Processes for parallel MCMC')
    parser.add_argument('--demcpt', action='store_true', help='Use DEMC-PT backend instead of emcee')
    parser.add_argument('--quiet', action='store_true', help='Reduce console output')
    args = parser.parse_args()

    exozippy(
        parfile=args.prior,
        tranpath=args.tran,
        rvpath=args.rv,
        sedfile=args.sed,
        prefix=args.prefix,
        circular=not args.noncircular,
        nomist=args.nomist,
        run_mcmc_flag=args.mcmc,
        mcmc_steps=args.steps,
        mcmc_burn=args.burn,
        mcmc_walkers=args.walkers,
        mcmc_threads=args.workers,
        mcmc_backend='demcpt' if args.demcpt else 'emcee',
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    _cli()
