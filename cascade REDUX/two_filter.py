"""
The 'main' module for this project. This module will optimise the dual filter setup, with all filter parameters.
Note that this therefore doesn't allow for the intermediary polariser yet.
"""

import skopt
import numpy as np
from elecsus import elecsus_methods as elecsus
from scipy.integrate import simps as integrate

# Define some global parameters.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 50e-3, "Dline": "D2", "rb85frac": 72.17}

def TwoFilterFitness(inputParams):
    """
    The fitness function used to determine the figure of merit for a dual filter setup. The input is a list with the following values (in this order):
    - B field 1
    - Temp 1
    - E theta
    - B theta 1
    - B phi 1
    - B field 2
    - Temp 2
    - B theta 2
    - B phi 2
    This fitness function does not allow for the presence of a polariser in between, so those complexities are removed (for now).
    """

    filter1Params = baseParamsFilter1.copy()
    filter2Params = baseParamsFilter2.copy()

    filter1Params["Bfield"] = inputParams[0]
    filter1Params["T"] = inputParams[1]
    filter1Params["Etheta"] = np.deg2rad(inputParams[2])
    filter1Params["Btheta"] = np.deg2rad(inputParams[3])
    filter1Params["Bphi"] = np.deg2rad(inputParams[4])
    filter2Params["Bfield"] = inputParams[5]
    filter2Params["T"] = inputParams[6]
    filter2Params["Btheta"] = np.deg2rad(inputParams[7])
    filter2Params["Bphi"] = np.deg2rad(inputParams[8])

    # First generate the output transmission for the first filter.
    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field from the first filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE1] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the first filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    # Call ElecSus to obtain the output field from the second filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE2] = elecsus.calculate(globalDetuning, outputE1, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the second filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE1))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filter1Params["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * outputE2)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    ENBW = ((integrate(filterTransmission, globalDetuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real
    
    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print("Filter 1 parameters:")
        print(str(filter1Params))
        print("Filter 2 parameters:")
        print(str(filter2Params))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def LiteratureTest():
    """
    Reproduce the figure of merit from the dual filter paper using the fitness function.
    """
    print("Reproducing literature value, we're looking for around 1.2:")

    inputParams = [270, 86.7, 6, 0, 0, 240, 79, 90, 0]
    
    print(TwoFilterFitness(inputParams))

    return

def Optimise(numIters):
    """
    Optimise the fitness function.
    """

    problemBounds = [(10, 1300), (40, 230), (0, 90), (0, 90), (0, 90), (10, 1300), (40, 230), (0, 90), (0, 90)]

    result = skopt.forest_minimize(TwoFilterFitness, problemBounds, verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)))

    print("Result determined! Here are the stats:")
    print("Best figure of merit: " + str(-1 * result.fun))
    print("Parameters giving this result: " + str(result.x))

    return

if __name__ == "__main__":

    # Test the fitness function against literature.
    #LiteratureTest()

    # Run the optimisation.
    Optimise(10000)