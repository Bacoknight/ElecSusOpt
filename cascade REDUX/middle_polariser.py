"""
This module is similar to the two_filter.py module, but there exists a polariser in between the two filters.
If this polariser is perpendicular to the input polarisation, the final polariser is free to be any angle.
"""

import skopt
import numpy as np
from elecsus import elecsus_methods as elecsus
from scipy.integrate import simps as integrate

# Define some global parameters.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 50e-3, "Dline": "D2", "rb85frac": 72.17}

def TwoFilterFitnessPol(inputParams):
    """
    The fitness function used to determine the figure of merit for a dual filter setup. The input is a list with the following values (in this order):
    - B field 1
    - Temp 1
    - E theta
    - B theta 1
    - B phi 1
    - Middle polariser angle
    - B field 2
    - Temp 2
    - B theta 2
    - B phi 2
    """

    filter1Params = baseParamsFilter1.copy()
    filter2Params = baseParamsFilter2.copy()

    filter1Params["Bfield"] = inputParams[0]
    filter1Params["T"] = inputParams[1]
    filter1Params["Etheta"] = np.deg2rad(inputParams[2])
    filter1Params["Btheta"] = np.deg2rad(inputParams[3])
    filter1Params["Bphi"] = np.deg2rad(inputParams[4])
    filter2Params["Bfield"] = inputParams[6]
    filter2Params["T"] = inputParams[7]
    filter2Params["Btheta"] = np.deg2rad(inputParams[8])
    filter2Params["Bphi"] = np.deg2rad(inputParams[9])

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

    # Apply the action of the middle polariser.
    jonesMatrixMid = np.matrix([[np.cos(np.deg2rad(inputParams[5]))**2, np.sin(np.deg2rad(inputParams[5]))*np.cos(np.deg2rad(inputParams[5])), 0],
								[np.sin(np.deg2rad(inputParams[5]))*np.cos(np.deg2rad(inputParams[5])), np.sin(np.deg2rad(inputParams[5]))**2, 0],
                                [0, 0, 1]])

    outputP1 = np.array(jonesMatrixMid * outputE1)

    # Call ElecSus to obtain the output field from the second filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE2] = elecsus.calculate(globalDetuning, outputP1, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the second filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE1))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the second polariser.
    # Must be perpendicular to middle filter.
    polariserAngle = np.deg2rad(inputParams[5]) + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrixFin = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrixFin * outputE2)

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

def TwoFilterFitnessSpec(inputParams):
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
    - Final polariser angle (free in the special case of the intermediate polariser being perpendicular to the input)
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

    # Apply the action of the middle polariser.
    midPAngle = filter1Params["Etheta"] + np.pi/2
    jonesMatrixMid = np.matrix([[np.cos(midPAngle)**2, np.sin(midPAngle)*np.cos(midPAngle), 0],
								[np.sin(midPAngle)*np.cos(midPAngle), np.sin(midPAngle)**2, 0],
                                [0, 0, 1]])

    outputP1 = np.array(jonesMatrixMid * outputE1)

    # Call ElecSus to obtain the output field from the second filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE2] = elecsus.calculate(globalDetuning, outputP1, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the second filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE1))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the second polariser.
    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrixFin = np.matrix([[np.cos(np.deg2rad(inputParams[9]))**2, np.sin(np.deg2rad(inputParams[9]))*np.cos(np.deg2rad(inputParams[9])), 0],
								[np.sin(np.deg2rad(inputParams[9]))*np.cos(np.deg2rad(inputParams[9])), np.sin(np.deg2rad(inputParams[9]))**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrixFin * outputE2)

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

def OptimiseNorm(numIters):
    """
    Optimise the fitness function.
    The input is a list with the following values (in this order):
    - B field 1
    - Temp 1
    - E theta
    - B theta 1
    - B phi 1
    - Middle polariser angle
    - B field 2
    - Temp 2
    - B theta 2
    - B phi 2
    """

    problemBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]

    result = skopt.forest_minimize(TwoFilterFitnessPol, problemBounds, verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET")

    print("Result determined! Here are the stats:")
    print("Best figure of merit: " + str(-1 * result.fun))
    print("Parameters giving this result: " + str(result.x))
    print("Feature Importance: " + str(result.models[-1].feature_importances_))

    return

def OptimiseSpec(numIters):
    """
    Optimise the fitness function.
    The input is a list with the following values (in this order):
    - B field 1
    - Temp 1
    - E theta
    - B theta 1
    - B phi 1
    - B field 2
    - Temp 2
    - B theta 2
    - B phi 2
    - Final polariser angle.
    """

    problemBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.)]

    result = skopt.forest_minimize(TwoFilterFitnessSpec, problemBounds, verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET")

    print("Result determined! Here are the stats:")
    print("Best figure of merit: " + str(-1 * result.fun))
    print("Parameters giving this result: " + str(result.x))
    print("Feature Importance: " + str(result.models[-1].feature_importances_))

    return

if __name__ == "__main__":
    # Start the optimisation.
    OptimiseNorm(1000)

    # Optimise the special case of a perpendicular middle polariser.
    OptimiseSpec(1000)