"""
Similar setup to original three split, however there is now a term which punishes polarisations
from the first two filters for being too different.
"""

import numpy as np
from elecsus import elecsus_methods as elecsus
from mpi4py import MPI
from scipy.integrate import simps as integrate
from os.path import isfile
from skopt import Optimizer, forest_minimize, load
from skopt.callbacks import CheckpointSaver
import gc
from tqdm import tqdm
import scipy.optimize as opt
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# Define some global parameters. NOTE: The two base filters have the same length in this version.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def ThreeFilterPunish(inputParams):
    """
    Uses the Stokes variables to punish off-orthogonal states in the combination of the fields.
    """

    filterRParams = baseParamsFilter1.copy()
    filterLParams = baseParamsFilter2.copy()
    filterFinParams = baseParamsFilter1.copy()

    filterRParams["Etheta"] = np.deg2rad(inputParams[0])
    filterRParams["Bfield"] = inputParams[1]
    filterRParams["T"] = inputParams[2]
    filterRParams["Btheta"] = np.deg2rad(inputParams[3])
    filterRParams["Bphi"] = np.deg2rad(inputParams[4])
    filterLParams["Bfield"] = inputParams[5]
    filterLParams["T"] = inputParams[6]
    filterLParams["Btheta"] = np.deg2rad(inputParams[7])
    filterLParams["Bphi"] = np.deg2rad(inputParams[8])    
    filterFinParams["Bfield"] = inputParams[9]
    filterFinParams["T"] = inputParams[10]
    filterFinParams["Btheta"] = np.deg2rad(inputParams[11])
    filterFinParams["Bphi"] = np.deg2rad(inputParams[12])

    # Both filters have the same input field.
    inputE = np.array([np.cos(filterRParams["Etheta"]), np.sin(filterRParams["Etheta"]), 0])/2

    # Put each field through their own filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [S0R, S1R, S2R, S3R, outputER] = elecsus.calculate(globalDetuning, inputE, filterRParams, outputs = ["S0", "S1", "S2", "S3", "E_out"])
    except:
        print("There was an issue in ElecSus for the right filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterRParams))
        print("Input field: " + str(inputE))
        return 0.0

    print(outputER[0][504])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [S0L, S1L, S2L, S3L, outputEL] = elecsus.calculate(globalDetuning, inputE, filterLParams, outputs = ["S0", "S1", "S2", "S3", "E_out"])
    except:
        print("There was an issue in ElecSus for the left filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterLParams))
        print("Input field: " + str(inputE))
        return 0.0

    print(outputEL[0][504])

    print(np.divide(outputER, outputEL)[0][504])

    plt.plot(globalDetuning, S1R, label = "S1R")
    plt.plot(globalDetuning, S2R, label = "S2R")
    plt.plot(globalDetuning, S3R, label = "S3R")
    plt.plot(globalDetuning, S1L, label = "S1L")
    plt.plot(globalDetuning, S2L, label = "S2L")
    plt.plot(globalDetuning, S3L, label = "S3L")
    plt.legend()
    plt.show()

    # Generate the required angles for plotting on the Poincare sphere. Psi is the angle in the S1-S2 plane, whilst zeta is the azimuthal angle.
    # NOTE: This generates lists for the angles of each detuning.
    psiRight = np.arctan(np.divide(S2R, S1R))
    zetaRight = np.arctan(np.divide(S3R, np.sqrt(np.square(S1R) + np.square(S2R))))
    psiLeft = np.arctan(np.divide(S2L, S1L))
    zetaLeft = np.arctan(np.divide(S3L, np.sqrt(np.square(S1L) + np.square(S2L))))

    deltaPsi = np.subtract(psiRight, psiLeft)
    deltaZeta = np.subtract(zetaRight, zetaLeft)

    # Recombine the two fields to form the total output field. Here we apply the punishment for 'bad' polarisations.
    combinedField = np.multiply(np.array(outputER) + np.array(outputEL), np.abs(np.multiply(np.cos(deltaPsi), np.cos(deltaZeta))))
    print(combinedField[0][504])

    # Pass the combined field through a final filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEFin] = elecsus.calculate(globalDetuning, combinedField, filterFinParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the final filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterFinParams))
        print("Input field: " + str(combinedField))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filterRParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * outputEFin)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    #assert filterTransmission.max() <= 1., "Maximal transmission is greater than 1, ensure your electric fields are correct in magnitude."

    ENBW = ((integrate(filterTransmission, globalDetuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print("Filter R parameters:")
        print(str(filterRParams))
        print("Filter L parameters:")
        print(str(filterLParams))
        print("Final filter parameters:")
        print(str(filterFinParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

if __name__ == "__main__":
    # Test the function.
    print(ThreeFilterPunish([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90.]))