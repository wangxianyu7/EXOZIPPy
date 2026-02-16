import os
from exozippy.exozippy import exozippy
base = os.path.dirname(os.path.abspath(__file__)) + '/'
ncpus = os.cpu_count()

def main():

    priorfile = 'HAT-3.priors'
    runmcmc = True
    nomist = False
    skipopt = False
    mcmc_steps = 2000
    mcmc_nchains = None
    mcmc_ntemps = 5
    mcmc_workers = ncpus - 1 if ncpus > 1 else None
    prefix = 'fitresults/HAT-3b.'

    priorfile = os.path.join(base, priorfile)
    tranpath = base + 'n20070428.*.dat' # support for glob patterns in transit files
    rvpath = base + 'HAT-3b.*.rv' # support for glob patterns in RV files
    sedfile = base + 'HAT-3.sed'
    tag = 'Torres' if nomist else 'MIST'
    prefix = prefix or os.path.expanduser(
        f'~/modeling/hat3/fitresults/HAT-3b.{tag}.')


    exozippy(
        parfile=priorfile,
        tranpath=tranpath,
        rvpath=rvpath,
        sedfile=sedfile,
        prefix=prefix,
        nomist=nomist,
        skipopt=skipopt,
        run_mcmc_flag=runmcmc,
        mcmc_steps=mcmc_steps,
        mcmc_nchains=mcmc_nchains,
        mcmc_ntemps=mcmc_ntemps,
        mcmc_threads=mcmc_workers,
        verbose=True,
    )


if __name__ == "__main__":
    main()