# jupyter lab --no-browser
import pyoculus
from pyoculus.solvers import PoincarePlot
from pyoculus.solvers import FixedPoint
from pyoculus.solvers import LyapunovExponent
from pyoculus.solvers import FluxSurfaceGR

import numpy as np


# Inbuilt problem taken from Hudson 2004
ps = pyoculus.problems.TwoWaves(k=0.0018)


# set up the integrator
iparams = dict()
iparams['rtol'] = 1e-9

# set up the Poincare plot
pparams = dict()
pparams['Nfp'] = 1
pparams['sbegin'] = 0.58
pparams['send'] = 0.66
pparams['nPtrj'] = 50
pparams['nPpts'] = 500
pparams['zeta'] =0.0

pplot = PoincarePlot(ps,pparams,integrator_params=iparams)
pdata=pplot.compute()
pplot.plot(ylim=[0.58,0.66],s=0.5)


# 

