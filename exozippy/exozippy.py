"""
High-level driver for EXOZIPPy (analogous to EXOFASTv2.pro).
Provides an all-in-one interface: read priors, run optimizer,
optional MCMC, and emit plots/logs.

Like EXOFASTv2, plots are generated at each stage:
  1. Initial guess  (prefix + 'start.')
  2. After amoeba   (prefix + 'amoeba.')
  3. After MCMC     (prefix + 'mcmc.')
"""

from pathlib import Path
from datetime import datetime
import argparse
import glob as _glob
import numpy as np

from .fit_exoplanet import fit_exoplanet, run_mcmc, build_initial_guess, parse_priors
from .massradius_mist import plot_mist_track
from .plottran import plottran
from .plotrv import plotrv
from .plotsed import plotsed
from .plotmcmc import plot_corner, plot_trace
from .derivepars import derivepars
from .exozippy_latextab import summarize_samples, write_csv, exozippy_latextab
from .mkprior2 import mkprior2


def _log(msg, verbose=True):
    if verbose:
        print(msg)


def _log_section(title, verbose=True):
    if verbose:
        bar = '=' * len(title)
        print(f'\n{bar}\n{title}\n{bar}')


def _make_plots(tranpath, rvpath, sedfile, bestfit, prefix_path,
                e, omega, use_mist, verbose, tag=None, samples=None):
    """Generate transit / RV / SED / MIST plots with an optional tag."""
    if tag:
        pfx = f'{prefix_path}{tag}.'
    else:
        pfx = str(prefix_path)

    tran_png = f'{pfx}transit.png'
    rv_png   = f'{pfx}rv.png'
    sed_png  = f'{pfx}sed.png'

    # Expand glob patterns to actual file paths (pass all files to plotters)
    tranfiles = sorted(_glob.glob(str(tranpath)))
    rvfiles = sorted(_glob.glob(str(rvpath)))

    if tranfiles:
        plottran(tranfiles, bestfit, samples=samples, e=e, omega=omega,
                 outfile=tran_png)
    if rvfiles:
        plotrv(rvfiles, bestfit, samples=samples, e=e, omega=omega,
               outfile=rv_png)
    if sedfile is not None:
        plotsed(str(sedfile), bestfit, outfile=sed_png)

    _log(f'Transit plot: {tran_png}', verbose)
    _log(f'RV plot     : {rv_png}', verbose)
    if sedfile is not None:
        _log(f'SED plot    : {sed_png}', verbose)

    if use_mist:
        mist_png = f'{pfx}mist.png'
        plot_mist_track(bestfit, outfile=mist_png)
        _log(f'MIST plot   : {mist_png}', verbose)


def exozippy(
    parfile,
    tranpath,
    rvpath,
    sedfile=None,
    prefix='fitresults/planet.',
    circular=True,
    nomist=False,
    skipopt=False,
    run_mcmc_flag=False,
    mcmc_steps=2000,
    mcmc_nchains=None,
    mcmc_ntemps=1,
    mcmc_threads=None,
    mcmc_checkpoint=None,
    mcmc_checkpoint_every=100,
    nstars=1,
    fitjittervar=False,
    fitvariance=False,
    fitthermal=False,
    fitreflect=False,
    fitbeam=False,
    fitellip=False,
    fitttv=False,
    novcve=False,
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

    parfile = Path(parfile).expanduser().resolve()
    tranpath = Path(tranpath).expanduser().resolve()
    rvpath = Path(rvpath).expanduser().resolve()
    if sedfile is not None:
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
    _log(f'SED file         : {sedfile or "(none)"}', verbose)
    _log(f'Output prefix    : {prefix_path}', verbose)
    _log(f'Circular orbit   : {circular}', verbose)
    _log(f'Use MIST         : {use_mist}', verbose)

    # Initial e/omega from priors (when non-circular, sesinw/secosw handle it internally)
    priors_parsed = parse_priors(str(parfile))
    if circular:
        e = 0.0
        omega = np.pi / 2
    else:
        e = priors_parsed.get('e', {}).get('value', 0.0)
        omega = priors_parsed.get('omega', {}).get('value', np.pi / 2)

    # Auto-detect vcve mode: transit-only + non-circular + not suppressed
    import glob as _gl
    _has_rv = bool(sorted(_gl.glob(str(rvpath))))
    usevcve = (not circular) and (not _has_rv) and (not novcve)
    if usevcve:
        _log('Using Vc/Ve eccentricity parameterization (transit-only)', verbose)

    pc_kwargs = dict(fitjittervar=fitjittervar, fitvariance=fitvariance,
                     fitttv=fitttv,
                     fitthermal=fitthermal, fitreflect=fitreflect,
                     fitbeam=fitbeam, fitellip=fitellip,
                     usevcve=usevcve)

    # --- Stage 1: start (initial guess plots) ---
    _log_section('Start Plots', verbose)
    init_guess = build_initial_guess(
        str(parfile), str(tranpath), str(rvpath),
        e=e, omega=omega, circular=circular,
        use_mist=use_mist, nstars=nstars,
        **pc_kwargs,
    )
    _make_plots(tranpath, rvpath, sedfile, init_guess, prefix_path,
                e, omega, use_mist, verbose, tag='start')

    # --- Stage 2: amoeba (optimizer) ---
    if skipopt:
        _log('Skipping optimizer (--skipopt)', verbose)
        bestfit = init_guess
    else:
        bestfit = fit_exoplanet(
            str(parfile), str(tranpath), str(rvpath),
            str(sedfile) if sedfile is not None else None,
            e=e, omega=omega, circular=circular,
            verbose=verbose, use_mist=use_mist,
            nstars=nstars, **pc_kwargs,
        )

        _log_section('Amoeba Plots', verbose)
        _make_plots(tranpath, rvpath, sedfile, bestfit, prefix_path,
                    e, omega, use_mist, verbose, tag='amoeba')

    # --- Stage 3: MCMC (DEMC-PT) ---
    samples = None
    if run_mcmc_flag and mcmc_steps and mcmc_steps > 0:
        _log_section('Running DEMC-PT', verbose)
        _log(f'Chains  : {mcmc_nchains or "auto (2*ndim)"}', verbose)
        _log(f'Temps   : {mcmc_ntemps}', verbose)
        _log(f'Steps   : {mcmc_steps}', verbose)
        if mcmc_checkpoint is None:
            mcmc_checkpoint = f'{prefix_path}mcmc.h5'
        if mcmc_checkpoint_every and mcmc_checkpoint_every > 0:
            _log(f'Checkpoint: {mcmc_checkpoint} (every {mcmc_checkpoint_every} steps)', verbose)
        samples, labels, summary, bestfit_mcmc, chain_info = run_mcmc(
            str(parfile), str(tranpath), str(rvpath),
            str(sedfile) if sedfile is not None else None,
            bestfit=bestfit, e=e, omega=omega, circular=circular,
            nchains=mcmc_nchains, nsteps=mcmc_steps, ntemps=mcmc_ntemps,
            verbose=verbose, use_mist=use_mist, nthreads=mcmc_threads,
            checkpoint=mcmc_checkpoint, checkpoint_every=mcmc_checkpoint_every,
            nstars=nstars, **pc_kwargs,
        )
        _log_section('MCMC Plots', verbose)
        if bestfit_mcmc is None:
            bestfit_mcmc = bestfit

        _make_plots(tranpath, rvpath, sedfile, bestfit_mcmc, prefix_path,
                    e, omega, use_mist, verbose, tag='mcmc',
                    samples=samples)

        # Corner plot and trace plot (arviz)
        corner_png = f'{prefix_path}mcmc.corner.png'
        trace_png = f'{prefix_path}mcmc.trace.png'
        try:
            plot_corner(chain_info, outfile=corner_png)
            _log(f'Corner plot : {corner_png}', verbose)
        except Exception as exc:
            _log(f'Corner plot failed: {exc}', verbose)
        try:
            plot_trace(chain_info, outfile=trace_png)
            _log(f'Trace plot  : {trace_png}', verbose)
        except Exception as exc:
            _log(f'Trace plot failed: {exc}', verbose)

        # Derived parameter tables (CSV + LaTeX)
        priors = parse_priors(str(parfile))
        derived = derivepars(samples, labels, priors, e=e, omega=omega)
        summary_d = summarize_samples(derived)
        csv_path = f"{prefix_path}median.csv"
        tex_path = f"{prefix_path}median.tex"
        caption = f"Median values and 68\\% confidence interval for {prefix_path}, created using EXOZIPPy"
        label = f"tab:{prefix_path.name}"
        write_csv(summary_d, csv_path, order=None)
        exozippy_latextab(summary_d, tex_path, caption=caption, label=label)
        _log(f"Saved table: {tex_path}", verbose)
        _log(f"Saved table: {csv_path}", verbose)

    # --- Write updated prior file with best-fit starting values ---
    final_ss = bestfit
    if run_mcmc_flag and samples is not None and bestfit_mcmc is not None:
        final_ss = bestfit_mcmc
    try:
        mkprior2(str(parfile), final_ss, verbose=verbose)
    except Exception as exc:
        _log(f'Updated priors failed: {exc}', verbose)

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
    parser.add_argument('--skipopt', action='store_true', help='Skip optimizer, go straight to MCMC')
    parser.add_argument('--mcmc', action='store_true', help='Run DEMC-PT after optimizer')
    parser.add_argument('--steps', type=int, default=2000, help='MCMC steps per chain')
    parser.add_argument('--nchains', type=int, default=None, help='Number of DEMC chains (default: 2*ndim)')
    parser.add_argument('--ntemps', type=int, default=1, help='Number of parallel tempering rungs (default: 1)')
    parser.add_argument('--workers', type=int, default=None, help='Processes for parallel MCMC')
    parser.add_argument('--checkpoint', type=str, default=None, help='HDF5 checkpoint path (default: <prefix>mcmc.h5)')
    parser.add_argument('--checkpoint-every', type=int, default=100, help='Checkpoint interval in steps (default: 100)')
    parser.add_argument('--ttv', action='store_true', help='Fit transit timing variations (requires >=3 transits)')
    parser.add_argument('--novcve', action='store_true', help='Disable Vc/Ve eccentricity parameterization')
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
        fitttv=args.ttv,
        novcve=args.novcve,
        skipopt=args.skipopt,
        run_mcmc_flag=args.mcmc,
        mcmc_steps=args.steps,
        mcmc_nchains=args.nchains,
        mcmc_ntemps=args.ntemps,
        mcmc_threads=args.workers,
        mcmc_checkpoint=args.checkpoint,
        mcmc_checkpoint_every=args.checkpoint_every,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    _cli()
