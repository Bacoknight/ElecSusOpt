import numpy as np
from bayes_opt import BayesianOptimization
from scipy.integrate import simps as integrate
from elecsus import elecsus_methods as elecsus
import time

def FoMGen(bField):

        # Detuning range.
        d = np.arange(-10000, 10000, 10) # MHz

        # Input parameters. Those represented by the input array x will be optimised.
        elecsus_params = {'Bfield':bField, 'rb85frac':72.17, 'Btheta':83, 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}

        [Iy] = elecsus.calculate(d, [1,0,0], elecsus_params, outputs=['Iy'])
        maxIy = max(Iy)
        ENBW = integrate(Iy, x=d/1e3)/maxIy
        FoM = maxIy/ENBW

        return(FoM)

# Define parameter bounds.
pbounds = {'bField': (0, 1e4)}
startTime = time.time()
# Create the optimiser.
optimiser = BayesianOptimization(f=FoMGen, pbounds=pbounds, verbose=2, random_state=1)

optimiser.maximize(init_points=20, n_iter=180)

print(optimiser.max)
elapsedTime = time.time() - startTime
print("Algorithm complete. Elapsed time: {} seconds".format(elapsedTime))