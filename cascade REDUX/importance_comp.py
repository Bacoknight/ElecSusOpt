"""
This module will generate feature importances for each filter optimisation.
This will be used to see how feature importances change as you introduce more complex filters.
"""

import skopt
import numpy as np
from elecsus import elecsus_methods as elecsus
from scipy.integrate import simps as integrate
import matplotlib.pyplot as plt
import json

# Define some global parameters.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 50e-3, "Dline": "D2", "rb85frac": 72.17}
baseParams = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

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

def SingleFilterFitness(inputParams):
    """
    The figure of merit generation function as required by skopt. The input is a list of each parameter. This is for a single filter, including bPhi.
    """

    elecsusParams = baseParams.copy()

    paramDict = {'Bfield': inputParams[0], "T": inputParams[1], 'Btheta': np.deg2rad(inputParams[2]), 'Etheta': np.deg2rad(inputParams[3]), 'Bphi': np.deg2rad(inputParams[4])}

    # This is the full dictionary to use on ElecSus.
    elecsusParams.update(paramDict)

    # First generate the output transmission as before.
    inputE = np.array([np.cos(elecsusParams["Etheta"]), np.sin(elecsusParams["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, inputE, elecsusParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(elecsusParams))
        print("Input field: " + str(inputE))
        return 0.0
    
    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = elecsusParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    singleFilterOutputE = np.array(jonesMatrix * outputE)

    # Get the transmission.
    singleFilterTransmission = (singleFilterOutputE * singleFilterOutputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(singleFilterTransmission, globalDetuning)/singleFilterTransmission.max().real)/1e3).real

    figureOfMerit = (singleFilterTransmission.max()/ENBW).real
    
    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print(str(elecsusParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def CompareImportance(numIters):
    """
    Run the ExtraTrees algorithm and use the model to generate feature importances for each filter type. 
    Prints all of this into a JSON file for later plotting.
    """

    # Generate a random seed to use for all optimisers.
    randomSeed = np.random.randint(1e6)
    print("Random seed: " + str(randomSeed))

    # Define bounds for each problem.
    sffBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.)]
    tffBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]
    tffpBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]
    tffsBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.)]

    # Optimise each filter.
    print("Optimising the single filter...")
    sffResult = skopt.forest_minimize(SingleFilterFitness, sffBounds, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET", random_state = randomSeed)
    
    print("Optimising full cascaded filter, no intermediate polariser...")
    tffResult = skopt.forest_minimize(TwoFilterFitness, tffBounds, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET", random_state = randomSeed)

    print("Optimising full cascaded filter, intermediate polariser present...")
    tffpResult = skopt.forest_minimize(TwoFilterFitnessPol, tffpBounds, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET", random_state = randomSeed)

    print("Optimising full cascaded filter, intermediate polariser present (special case)...")
    tffsResult = skopt.forest_minimize(TwoFilterFitnessSpec, tffsBounds, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET", random_state = randomSeed)

    print("Optimisations complete! Compiling JSON file...")

    sffImportance = sffResult.models[-1].feature_importances_
    tffImportance = tffResult.models[-1].feature_importances_
    tffpImportance = tffpResult.models[-1].feature_importances_
    tffsImportance = tffsResult.models[-1].feature_importances_
    # Set up a dictionary with all the wanted values.
    dataDict = {"Single Filter": {"B field 1": sffImportance[0], "Temperature 1": sffImportance[1], "B theta 1": sffImportance[2], "E theta": sffImportance[3], "B phi 1": sffImportance[4]},
    
    "Cascaded Filter, No Middle Polariser": {"B field 1": tffImportance[0], "Temperature 1": tffImportance[1], "E theta": tffImportance[2], "B theta 1": tffImportance[3], "B phi 1": tffImportance[4],
    "B field 2": tffImportance[5], "Temperature 2": tffImportance[6], "B theta 2": tffImportance[7], "B phi 2": tffImportance[8]},
    
    "Cascaded Filter, Middle Polariser General": {"B field 1": tffpImportance[0], "Temperature 1": tffpImportance[1], "E theta": tffpImportance[2], "B theta 1": tffpImportance[3], "B phi 1": tffpImportance[4],
    "Free polariser angle": tffpImportance[5], "B field 2": tffpImportance[6], "Temperature 2": tffpImportance[7], "B theta 2": tffpImportance[8], "B phi 2": tffpImportance[9]},
    
    "Cascaded Filter, Middle Polariser Perpendicular": {"B field 1": tffsImportance[0], "Temperature 1": tffsImportance[1], "E theta": tffsImportance[2], "B theta 1": tffsImportance[3], "B phi 1": tffsImportance[4],
    "B field 2": tffsImportance[5], "Temperature 2": tffsImportance[6], "B theta 2": tffsImportance[7], "B phi 2": tffsImportance[8], "Free polariser angle": tffsImportance[9]}}

    # Write this data to a file.
    with open("variable_importance.txt", "w") as outputFile:
        # NOTE: This is in write mode, so will erase all previous data on that file (if it exists).
        json.dump(dataDict, outputFile)

    return

if __name__ == "__main__":
    # Run the variable importance generator.
    CompareImportance(2000)

