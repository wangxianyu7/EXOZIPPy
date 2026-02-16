import os
from exozippy.exozippy import exozippy
base = os.path.dirname(os.path.abspath(__file__)) + '/'
ncpus = os.cpu_count()

def main():
    priorfile = 'kelt17.priors'
    runmcmc = True
    nomist = False
    skipopt = False
    mcmc_steps = 2000
    mcmc_nchains = None
    mcmc_ntemps = 5
    mcmc_workers = ncpus - 1 if ncpus > 1 else None
    prefix = 'fitresults/KELT-17b.'

    priorfile = os.path.join(base, priorfile)
    # 12 transit light curves across 7 bands (B, I, V, g', i', r', z')
    tranpath = base + 'DT/n20??????.*.dat'
    # 2 RV telescopes: TRES (orbital) + TRESRM (Rossiter-McLaughlin)
    rvpath = base + 'DT/KELT-17b.*.rv'
    sedfile = base + 'DT/kelt17.sed'
    tag = 'Torres' if nomist else 'MIST'
    prefix = prefix or os.path.expanduser(
        f'~/modeling/kelt17/fitresults/KELT-17b.{tag}.')

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
