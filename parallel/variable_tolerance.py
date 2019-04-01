"""
For the best parallel result determined, determine the change in a variable value needed to create
a change of 1% in the figure of merit. Whether this is an increase or decrease currently doesn't matter.
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
import matplotlib.pyplot as plt
plt.rc("text", usetex = True)

import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

# Define some global parameters. NOTE: The two base filters have the same length.
globalDetuning = np.sort(np.append(np.linspace(-20000, 20000, 10), np.linspace(-500, 500, 10)))
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def ParallelFitness(inputParams):
    """
    The fitness function that will be used in this module.
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
    polariserAnglePreFin = np.deg2rad(inputParams[13])

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

def VariableTolerance(optimalParams):
    """
    Determines the shift required in each variable from the optimum to induce a change of 1% in the figure of merit.
    """

    # Determine the reference FoM.
    optimalFoM = ParallelFitness(optimalParams)
    print("Optimal figure of merit: {}".format(str(optimalFoM)))

    variableNames = ["Etheta", "Bfield 1", "Temperature 1", "Btheta 1", "Bphi 1", "Bfield 2", "Temperature 2", "Btheta 2", "Bphi 2", "Bfield 3", "Temperature 3", "Btheta 3", "Bphi 3", "Extra Polariser Angle"]

    # List the perturbations.
    perturbationsPositive = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    perturbationsNegative = np.multiply(-1, perturbationsPositive)

    perturbationPosDict = {}
    print("Testing positive perturbations:")
    for index, param in enumerate(optimalParams):
        for perturbation in perturbationsPositive:
            # Apply the perturbation.
            perturbParams = optimalParams.copy()
            perturbParams[index] = param + perturbation
            perturbFoM = ParallelFitness(perturbParams)

            if perturbFoM/optimalFoM <= 0.99 or perturbFoM/optimalFoM >= 1.01:
                print("Sensitivity of " + str(variableNames[index]) + ": Change of 1% from increase by " + str(perturbation))
                perturbationPosDict[str(variableNames[index])] = perturbation
                break

    perturbationNegDict = {}
    print("Testing negative perturbations:")
    for index, param in enumerate(optimalParams):
        for perturbation in perturbationsNegative:
            # Apply the perturbation.
            perturbParams = optimalParams.copy()
            perturbParams[index] = param + perturbation
            perturbFoM = ParallelFitness(perturbParams)

            if perturbFoM/optimalFoM <= 0.99 or perturbFoM/optimalFoM >= 1.01:
                print("Sensitivity of " + str(variableNames[index]) + ": Change of 1% from decrease by " + str(perturbation))
                perturbationNegDict[str(variableNames[index])] = perturbation
                break

    return perturbationPosDict, perturbationNegDict

def PlotTolerance(optimalParams, variableList):
    """
    Plots a graph of the tolerance for a given variable in each filter.
    """

    allVariables = ["Etheta", "Bfield 1", "Temperature 1", "Btheta 1", "Bphi 1", "Bfield 2", "Temperature 2", "Btheta 2", "Bphi 2", "Bfield 3", "Temperature 3", "Btheta 3", "Bphi 3", "Extra Polariser Angle"]

    perturbationPosDict, perturbationNegDict = VariableTolerance(optimalParams)

    for variableName in perturbationPosDict.keys():
        if variableName in variableList:
            # Get the index of the variable.
            varIndex = None
            for index, varName in enumerate(allVariables):
                if variableName in varName:
                    varIndex = index
                    break

            # Plot the tolerance curve.
            posTol = perturbationPosDict[variableName]
            negTol = perturbationNegDict[variableName]
            perturbVals = np.sort(np.append(np.linspace(negTol, 0, 10), np.linspace(0, posTol, 10)))
            fomVals = []
            testParams = optimalParams.copy()
            for perturbation in perturbVals:
                testParams[varIndex] += perturbation
                fomVals.append(abs(ParallelFitness(testParams)))
            perturbValsPlot = np.sort(np.append(np.linspace(negTol, 0, 10)/abs(negTol), np.linspace(0, posTol, 10)/abs(negTol)))
            print(perturbValsPlot)
            plt.plot(perturbValsPlot, fomVals)
            
    plt.show()

    return

if __name__ == "__main__":
    # Determine the variable tolerance.
    # VariableTolerance([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    # 143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    # 87.23663536, 90., 175.90283232701586])

    variableList = ["Temperature 1", "Temperature 2", "Temperature 3"]
    PlotTolerance([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    43.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90., 175.90283232701586], variableList)