import os
from exozippy.exozippy import exozippy
base = os.path.dirname(os.path.abspath(__file__)) + '/'
ncpus = os.cpu_count()

def main():

    priorfile = 'kelt17.priors'
    runmcmc = True
    nomist = False
    skipopt = False
    mcmc_steps = 5000
    mcmc_nchains = None
    mcmc_ntemps = 4
    mcmc_workers = ncpus - 1 if ncpus > 1 else None
    prefix = 'fitresults/KELT-17b.'

    priorfile = os.path.join(base, priorfile)
    tranpath  = base + 'n20??????.*.dat'
    rvpath    = base + 'KELT-17b.*.rv'
    sedfile   = base + 'kelt17.sed'

    # rmbands: sorted alphabetically by filename
    #   KELT-17b.TRES.rv     → 'notrm'  (orbital RV, skip RM model)
    #   KELT-17b.TRES_RM0.rv → 'V'      (RM night 1, V band)
    #   KELT-17b.TRES_RM1.rv → 'V'      (RM night 2, V band)
    rmbands = ['notrm', 'V', 'V']

    exozippy(
        parfile          = priorfile,
        tranpath         = tranpath,
        rvpath           = rvpath,
        sedfile          = sedfile,
        prefix           = prefix,
        circular         = True,
        nomist           = nomist,
        skipopt          = skipopt,
        run_mcmc_flag    = runmcmc,
        mcmc_steps       = mcmc_steps,
        mcmc_nchains     = mcmc_nchains,
        mcmc_ntemps      = mcmc_ntemps,
        mcmc_threads     = mcmc_workers,
        rossiter         = True,
        rmbands          = rmbands,
        verbose          = True,
    )


if __name__ == "__main__":
    main()
