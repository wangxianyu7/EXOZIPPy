import os
from exozippy.exozippy import exozippy
base = os.path.dirname(os.path.abspath(__file__)) + '/'
ncpus = os.cpu_count()

def main():
    priorfile = 'kelt14.priors'
    runmcmc = True
    nomist = True            # IDL: /nomist, /torres
    skipopt = False
    circular = False         # IDL: circular=[0] — eccentric orbit
    mcmc_steps = 5000
    mcmc_nchains = None
    mcmc_ntemps = 5
    mcmc_workers = ncpus - 1 if ncpus > 1 else None
    prefix = 'fitresults/KELT-14b.'

    priorfile = os.path.join(base, priorfile)
    tranpath = base + 'n20*.dat'           # glob: EulerCam light curves
    rvpath = base + 'KELT-14b.*.rv'       # glob: AAT + CORALIE RVs
    sedfile = None  # no SED file for this target, nomist, torres is used instead for stellar parameters

    tag = 'Torres' if nomist else 'MIST'
    prefix = prefix or os.path.expanduser(
        f'~/modeling/kelt14/fitresults/KELT-14b.{tag}.')


    exozippy(
        parfile=priorfile,
        tranpath=tranpath,
        rvpath=rvpath,
        sedfile=sedfile,
        prefix=prefix,
        nomist=nomist,
        circular=circular,
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