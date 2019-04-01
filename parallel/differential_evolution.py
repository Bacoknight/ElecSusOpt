"""
Optimises the parallel filter using differential evolution.
"""
import numpy as np
from elecsus import elecsus_methods as elecsus
from mpi4py import MPI
from scipy.integrate import simps as integrate
from os.path import isfile
from skopt import Optimizer, forest_minimize, load, dump
import gc
from tqdm import tqdm
import scipy.optimize as opt
from sklearn.externals import joblib
import pandas as pd

# Define some global parameters. NOTE: The two base filters have the same length.
globalDetuning = np.sort(np.append(np.linspace(-20000, 20000, 1000), np.linspace(-500, 500, 10000)))
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def ThreeFilter(inputParams):
    """
    This fitness function has two independent beams which go through a single filter each. It is
    assumed that these beams are parallel and have the same linear polarisation angle before starting.
    The input variables for this function are:
    - E theta
    - B field 1
    - Temp 1
    - B theta 1
    - B field 2
    - Temp 2
    - B theta 2
    - B field 3
    - Temp 3
    - B theta 3
    - Extra polariser angle
    After the filter, the electric fields are combined additively (hence being naive), and pass through a final filter and a polariser
    which is perpendicular to the input polarisation to ensure a convergent integral.
    """

    filterRParams = baseParamsFilter1.copy()
    filterLParams = baseParamsFilter2.copy()
    filterFinParams = baseParamsFilter1.copy()

    filterRParams["Etheta"] = np.deg2rad(inputParams[0])
    filterRParams["Bfield"] = inputParams[1]
    filterRParams["T"] = inputParams[2]
    filterRParams["Btheta"] = np.deg2rad(inputParams[3])
    filterLParams["Bfield"] = inputParams[4]
    filterLParams["T"] = inputParams[5]
    filterLParams["Btheta"] = np.deg2rad(inputParams[6])  
    filterFinParams["Bfield"] = inputParams[7]
    filterFinParams["T"] = inputParams[8]
    filterFinParams["Btheta"] = np.deg2rad(inputParams[9])

    # Both filters have the same input field. Normalised for intensity to be 1.
    # NOTE: The scaling is 0.5, from comparing with known results.
    inputE = np.array([np.cos(filterRParams["Etheta"]), np.sin(filterRParams["Etheta"]), 0])/2

    # Put each field through their own filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputER] = elecsus.calculate(globalDetuning, inputE, filterRParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the right filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterRParams))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEL] = elecsus.calculate(globalDetuning, inputE, filterLParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the left filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterLParams))
        print("Input field: " + str(inputE))
        return 0.0

    # Recombine the two fields to form the total output field. This is where the fitness function is naive.
    combinedField = np.array(outputER) + np.array(outputEL)

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
    polariserAnglePreFin = np.deg2rad(inputParams[10])

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    preFinalPolariser = np.matrix([[np.cos(polariserAnglePreFin)**2, np.sin(polariserAnglePreFin)*np.cos(polariserAnglePreFin), 0],
								[np.sin(polariserAnglePreFin)*np.cos(polariserAnglePreFin), np.sin(polariserAnglePreFin)**2, 0],
                                [0, 0, 1]])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    finalPolariserAngle = filterRParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    finalPolariser = np.matrix([[np.cos(finalPolariserAngle)**2, np.sin(finalPolariserAngle)*np.cos(finalPolariserAngle), 0],
								[np.sin(finalPolariserAngle)*np.cos(finalPolariserAngle), np.sin(finalPolariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(finalPolariser * preFinalPolariser * outputEFin)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    assert filterTransmission.max() <= 1., "Maximal transmission is greater than 1, ensure your electric fields are correct in magnitude."

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

def Optimise(numIters):
    """
    Optimise the fitness function.
    """

    # Set up the problem.
    problemBounds = [(0., 180.), (0., 1300.), (0., 230.), (0., 180.), (0., 1300.), (0., 230.), (0., 180.), (0., 1300.), (0., 230.), (0., 180.), (0., 180.)]

    optResult = opt.differential_evolution(ThreeFilter, problemBounds, disp = True)

    print(optResult.fun, optResult.x)

    return

if __name__ == "__main__":
    Optimise(100)