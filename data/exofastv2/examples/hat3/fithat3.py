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
    parser.add_argument('--skipopt', action='store_true', help='Skip optimizer, go straight to MCMC.')
    parser.add_argument('--mcmc', action='store_true', help='Run DEMC-PT after optimizer.')
    parser.add_argument('--nsteps', type=int, default=2000, help='MCMC steps per chain.')
    parser.add_argument('--nchains', type=int, default=None, help='DEMC chains (default: 2*ndim).')
    parser.add_argument('--ntemps', type=int, default=1, help='Parallel tempering rungs (default: 1).')
    parser.add_argument('--workers', type=int, default=None, help='Processes for parallel MCMC.')
    parser.add_argument('--prefix', default=None, help='Output prefix (defaults to fitresults/HAT-3b.<tag>.)')
    parser.add_argument('--quiet', action='store_true', help='Reduce console output.')
    args = parser.parse_args()

    priorfile = os.path.join(base, args.prior)
    tranpath = base + 'n20070428.Sloani.KepCam.dat'
    rvpath = base + 'HAT-3b.HIRES.rv'
    sedfile = base + 'HAT-3.sed'
    tag = 'Torres' if args.nomist else 'MIST'
    prefix = args.prefix or os.path.expanduser(
        f'~/modeling/hat3/fitresults/HAT-3b.{tag}.')

    exozippy(
        parfile=priorfile,
        tranpath=tranpath,
        rvpath=rvpath,
        sedfile=sedfile,
        prefix=prefix,
        nomist=args.nomist,
        skipopt=args.skipopt,
        run_mcmc_flag=args.mcmc,
        mcmc_steps=args.nsteps,
        mcmc_nchains=args.nchains,
        mcmc_ntemps=args.ntemps,
        mcmc_threads=args.workers,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
