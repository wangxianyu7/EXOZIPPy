import os
from exozippy.exozippy import exozippy
base = os.path.dirname(os.path.abspath(__file__)) + '/'
ncpus = os.cpu_count()

def main():
    priorfile = 'wasp18.priors'
    runmcmc = True
    nomist = False
    skipopt = False
    mcmc_steps = 20000
    mcmc_nchains = None
    mcmc_ntemps = 5
    mcmc_workers = ncpus - 1 if ncpus > 1 else None
    prefix = 'fitresults/WASP-18b.'

    priorfile = os.path.join(base, priorfile)
    # 2 TESS sectors (s02 + s03), ~31k data points total
    tranpath = base + 'n20*.dat'
    # 1 RV telescope: CORALIE (9 points)
    rvpath = base + 'WASP-18.*.rv'
    sedfile = base + 'wasp18.sed'
    tag = 'Torres' if nomist else 'MIST'
    prefix = prefix or os.path.expanduser(
        f'~/modeling/wasp18/fitresults/WASP-18b.{tag}.')

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
        # Phase curve fitting (IDL: fitthermal=['TESS'], fitreflect=['TESS'],
        #                           fitellip=['TESS'], fitbeam=[1])
        fitthermal=True,
        fitreflect=True,
        fitbeam=True,
        fitellip=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()
