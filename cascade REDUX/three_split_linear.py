"""
Given the difficulty of playing with elliptical waves, this module instead linearly polarises the resultant wave along the semimajor
axis of the peak of the resultant field from one of the filters. This gives the best chance of keeping a high figure of merit.
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
from mpmath import findroot
from functools import partial
import math

# Define some global parameters. NOTE: The two base filters have the same length in this version.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def ThreeFilterLinear(inputParams):
    """
    Polarises the outputs of the parallel filters along the semimajor axis of the 'better' filter peak.
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
        [S1R, S2R, outputER] = elecsus.calculate(globalDetuning, inputE, filterRParams, outputs = ["S1", "S2", "E_out"])
    except:
        print("There was an issue in ElecSus for the right filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterRParams))
        print("Input field: " + str(inputE))
        return 0.0

    print(outputER[0][504])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [S1L, S2L, outputEL] = elecsus.calculate(globalDetuning, inputE, filterLParams, outputs = ["S1", "S2", "E_out"])
    except:
        print("There was an issue in ElecSus for the left filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterLParams))
        print("Input field: " + str(inputE))
        return 0.0

    print(outputEL[0][504])

    # Find the angle the semimajor axis makes with the x axis. Field from right filter is chosen as it gives a better shape.
    semimajorTheta = 0.5 * np.arctan(np.divide(S2R, S1R))[504]
    print(semimajorTheta)
    rightPolariser = np.matrix([[np.cos(semimajorTheta)**2, np.sin(semimajorTheta)*np.cos(semimajorTheta), 0],
								[np.sin(semimajorTheta)*np.cos(semimajorTheta), np.sin(semimajorTheta)**2, 0],
                                [0, 0, 1]])
    leftTheta = semimajorTheta + np.pi/2
    leftPolariser = np.matrix([[np.cos(leftTheta)**2, np.sin(leftTheta)*np.cos(leftTheta), 0],
								[np.sin(leftTheta)*np.cos(leftTheta), np.sin(leftTheta)**2, 0],
                                [0, 0, 1]])

    # Pass both fields through the same polariser.
    combinedField = np.array(rightPolariser * outputER) + np.array(leftPolariser * outputEL)

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

    assert filterTransmission.max() <= 1., "Maximal transmission is greater than 1, ensure your electric fields are correct in magnitude."
    plt.plot(globalDetuning, filterTransmission)
    plt.show()

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

def ThreeFilterAngFree(inputParams):
    """
    Polarises the outputs of the parallel filters along the given angle.
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
        [S1R, S2R, outputER] = elecsus.calculate(globalDetuning, inputE, filterRParams, outputs = ["S1", "S2", "E_out"])
    except:
        print("There was an issue in ElecSus for the right filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterRParams))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [S1L, S2L, outputEL] = elecsus.calculate(globalDetuning, inputE, filterLParams, outputs = ["S1", "S2", "E_out"])
    except:
        print("There was an issue in ElecSus for the left filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterLParams))
        print("Input field: " + str(inputE))
        return 0.0

    # Find the angle the semimajor axis makes with the x axis. Field from right filter is chosen as it gives a better shape.
    # semimajorTheta = inputParams[13]
    # leftTheta = inputParams[14]

    print(outputER[:,504])

    def LinearAngleFitness(wavAngle, jonesVec):
        """
        The fitness function to find the roots of to turn the elliptical wave linear.
        """

        sinCos = math.sin(wavAngle.real) * math.cos(wavAngle.real)
        sinSq = math.sin(wavAngle.real) ** 2
        cosSq = math.cos(wavAngle.real) ** 2

        part1 = 1.j * (jonesVec[0].conjugate() * (cosSq - 1.j * sinSq) + jonesVec[1].conjugate() * (1. + 1.j) * sinCos)
        return [part1 - part1.conjugate()]

    semimajorTheta = findroot(partial(LinearAngleFitness, jonesVec = outputER[:,504]), 0.)[0].real
    leftTheta = findroot(partial(LinearAngleFitness, jonesVec = outputEL[:,504]), 0.)[0].real

    print(semimajorTheta)
    print(leftTheta)

    rightWaveplate = np.exp(-1.j * np.pi/4.) * np.matrix([[math.cos(semimajorTheta)**2 + 1.j * math.sin(semimajorTheta)**2, (1. - 1.j) * math.sin(semimajorTheta)*math.cos(semimajorTheta), 0],
								[(1. - 1.j) * math.sin(semimajorTheta)*math.cos(semimajorTheta), math.sin(semimajorTheta)**2 + 1.j * math.cos(semimajorTheta)**2, 0],
                                [0, 0, 1]])

    leftWaveplate = np.exp(-1.j * np.pi/4.) * np.matrix([[math.cos(leftTheta)**2 + 1.j * math.sin(leftTheta)**2, (1. - 1.j) * math.sin(leftTheta) * math.cos(leftTheta), 0],
								[(1. - 1.j) * math.sin(leftTheta)*math.cos(leftTheta), math.sin(leftTheta)**2 + 1.j * math.cos(leftTheta)**2, 0],
                                [0, 0, 1]])

    # Pass both fields through their waveplates.
    combinedField = np.array(rightWaveplate * outputER) + np.array(leftWaveplate * outputEL)

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

if __name__ == "__main__":
    # Test the function.
    # for theta1 in np.linspace(0., 90., num = 100):
    #     for theta2 in np.linspace(0., 90., num = 100):
    #         val = ThreeFilterAngFree([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    #         143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    #         87.23663536, 90., theta1, theta2])
    #         if val < -1.7:
    #             print(val)

    print(ThreeFilterAngFree([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
            143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
            87.23663536, 90., 0, 0]))