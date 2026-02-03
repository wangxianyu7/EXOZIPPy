'''
The high level code for exozippy (equivalent to exofastv2.pro)
'''


import numpy as np
# import pymc as pm
import matplotlib as plt
import ipdb
from astropy import units as u
import astropy.constants as const
from .read_par import read_par
from .read_tran import read_tran
from .read_rv import read_rv
# from .build_model import build_model
# from .build_latex_table import build_latex_table

def exozippy(parfile=None, \
             prefix='fitresults/planet.',\
             # data file inputs
             rvpath=None, tranpath=None, \
             astrompath=None, dtpath=None, \
             # SED model inputs
             fluxfile=None,mistsedfile=None, \
             sedfile=None,specphotpath=None, \
             noavprior=False,\
             fbolsedfloor=0.024,teffsedfloor=0.02,\
             fehsedfloor=0.08, oned=False,\
             # evolutionary model inputs
             yy=False, nomist=False, parsec=False, \
             torres=False, mannrad=False,mannmass=False, \
             teffemfloor=0.02, fehemfloor=0.08, \
             rstaremfloor=0.042,ageemfloor=0.01,\
             # BEER model inputs
             fitthermal=False, fitellip=False, \
             fitreflect=False, fitphase=False, \
             fitbeam=False, derivebeam=False, \
             # star inputs
             nstars=1, starndx=0, \
             diluted=False, fitdilute=False, \
             # planet inputs
             nplanets=1, \
             fittran=None, fitrv=None, \
             rossiter=False, fitdt=None, \
             circular=False, tides=False, \
             alloworbitcrossing=False, \
             chen=False, i180=False, \
             # RV inputs
             fitslope=False, fitquad=False, rvepoch=None,\
             # transit inputs
             noclaret=False, \
             ttvs=False, tivs=False, tdeltavs=False,\
             longcadence=False, exptime=False, ninterp=False, \
             rejectflatmodel=False,\
             fitspline=False, splinespace=0.75, \
             fitwavelet=False, \
             # reparameterization inputs
             fitlogmp=False,\
             novcve=False, nochord=False, fitsign=False, \
             fittt=False, earth=False, \
             # plotting inputs
             transitrange=[None,None],rvrange=[None,None],\
             sedrange=[None,None],emrange=[None,None], \
             # debugging inputs
             debug=False, verbose=False, delay=0.0, \
             # MCMC inputs
             maxsteps=None, nthin=None, maxtime=None, \
             maxgr=1.01, mintz=1000, \
             dontstop=False, \
             ntemps=1, tf=200, keephot=False, \
             seed=None, \
             stretch=False, \
             nthreads=None, \
             # General inputs
             skiptt=False, \
             usernote=None, \
             mksummarypg=False, \
             nocovar=False, \
             plotonly=False, bestonly=False, \
             badstart=False):
    param = read_par(parfile)
    trandata = read_tran(tranpath)
    rvdata = read_rv(rvpath)
    print(param)
    print(trandata)
    print(rvdata)

if __name__ == "__main__":
    exozippy(parfile="../data/exofastv2/examples/hat3/HAT-3.priors",
             tranpath="../data/exofastv2/examples/hat3/n20070428.Sloani.KepCam.dat",
             rvpath="../data/exofastv2/examples/hat3/HAT-3b.HIRES.rv")
