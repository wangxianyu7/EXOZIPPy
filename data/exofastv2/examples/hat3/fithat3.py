"""
Python equivalent of fithat3_modern.pro — joint SED + MIST + Transit + RV fit.
Usage examples:
    python fithat3.py
    python fithat3.py --mcmc --nsteps 4000
    python fithat3.py --nomist --prior hat3.torres.priors
"""
import argparse
import os

from exozippy.exozippy import exozippy


def main():
    base = os.path.dirname(os.path.abspath(__file__)) + '/'

    parser = argparse.ArgumentParser(description="High-level EXOZIPPy run for HAT-3b.")
    parser.add_argument('--prior', default='HAT-3.priors', help='Relative path to prior file.')
    parser.add_argument('--nomist', action='store_true', help='Disable MIST evolutionary prior.')
    parser.add_argument('--mcmc', action='store_true', help='Run MCMC after optimizer.')
    parser.add_argument('--demcpt', action='store_true', help='Use DEMC-PT sampler instead of emcee.')
    parser.add_argument('--nsteps', type=int, default=2000, help='MCMC steps.')
    parser.add_argument('--nburn', type=int, help='MCMC burn-in (defaults to 20%% of steps).')
    parser.add_argument('--nwalkers', type=int, default=32, help='MCMC walkers.')
    parser.add_argument('--workers', type=int, default=None, help='Processes for parallel MCMC.')
    parser.add_argument('--prefix', default=None, help='Output prefix (defaults to fitresults/HAT-3b.<tag>.)')
    parser.add_argument('--quiet', action='store_true', help='Reduce console output.')
    args = parser.parse_args()

    priorfile = os.path.join(base, args.prior)
    tranpath = base + 'n20070428.Sloani.KepCam.dat'
    rvpath = base + 'HAT-3b.HIRES.rv'
    sedfile = base + 'HAT-3.sed'
    tag = 'Torres' if args.nomist else 'MIST'
    prefix = args.prefix or os.path.join(base, 'fitresults', f'HAT-3b.{tag}.')

    exozippy(
        parfile=priorfile,
        tranpath=tranpath,
        rvpath=rvpath,
        sedfile=sedfile,
        prefix=prefix,
        nomist=args.nomist,
        run_mcmc_flag=args.mcmc,
        mcmc_steps=args.nsteps,
        mcmc_burn=args.nburn,
        mcmc_walkers=args.nwalkers,
        mcmc_threads=args.workers,
        mcmc_backend='demcpt' if args.demcpt else 'emcee',
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
