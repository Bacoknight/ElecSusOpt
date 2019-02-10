"""
The 'main' module for this project. This module will optimise the dual filter setup, with all filter parameters.
Note that this therefore doesn't allow for the intermediary polariser yet.
"""

import skopt
import numpy as np
from elecsus import elecsus_methods as elecsus
from sklearn.externals import joblib
from scipy.integrate import simps as integrate
from sklearn.tree import export_graphviz
from tqdm import tqdm
import pandas as pd
import shap

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

    problemBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]

    #result = skopt.forest_minimize(TwoFilterFitness, problemBounds, verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET")
    optimizer = skopt.Optimizer(dimensions = problemBounds, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET")

    for _ in tqdm(range(numIters)):

        # Ask for the next point.
        nextPoint = optimizer.ask()

        if len(optimizer.models) != 0:
            # Clear old models, to prevent memory overload.
            optimizer.models = [optimizer.models[-1]]

        pointVal = TwoFilterFitness(nextPoint)

        result = optimizer.tell(nextPoint, pointVal)

    print("Result determined for cascaded filters! Here are the stats:")
    print("Best figure of merit: " + str(-1 * result.fun))
    print("Parameters giving this result: " + str(result.x))
    # tree = result.models[-1].estimators_[5]
    # print("Feature Importance: " + str(result.models[-1].feature_importances_))
    # export_graphviz(tree, out_file = "tree_output.dot", precision = 2, impurity = False)

    return

def Sensitivity(optimalParams):
    """
    Determine the sensitivity of each continuous filter parameter to perturbation.
    """

    # Determine the reference FoM.
    optimalFoM = TwoFilterFitness(optimalParams)
    print("Optimal figure of merit: " + str(optimalFoM))

    # Match the list values to variable names.
    variableNames = ["Bfield 1", "Temperature 1", "Etheta", "Btheta 1", "Bphi 1", "Bfield 2", "Temperature 2", "Btheta 2", "Bphi 2"]

    # List the perturbations.
    perturbationsPositive = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    perturbationsNegative = np.multiply(-1, perturbationsPositive)

    print("Testing positive perturbations:")
    for index, param in enumerate(optimalParams):
        for perturbation in perturbationsPositive:
            # Apply the perturbation.
            perturbParams = optimalParams.copy()
            perturbParams[index] = param + perturbation
            perturbFoM = TwoFilterFitness(perturbParams)

            if perturbFoM/optimalFoM <= 0.99 or perturbFoM/optimalFoM >= 1.01:
                print("Sensitivity of " + str(variableNames[index]) + ": Change of 1% from increase by " + str(perturbation))
                break

    print("Testing negative perturbations:")
    for index, param in enumerate(optimalParams):
        for perturbation in perturbationsNegative:
            # Apply the perturbation.
            perturbParams = optimalParams.copy()
            perturbParams[index] = param + perturbation
            perturbFoM = TwoFilterFitness(perturbParams)

            if perturbFoM/optimalFoM <= 0.99 or perturbFoM/optimalFoM >= 1.01:
                print("Sensitivity of " + str(variableNames[index]) + ": Change of 1% from decrease by " + str(perturbation))
                break

    return

def ShapImportance(numIters):
    """
    Determine the SHAP importance of the variables.
    """

    # First create the model.
    problemBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]

    #result = skopt.forest_minimize(TwoFilterFitness, problemBounds, verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET")
    optimizer = skopt.Optimizer(dimensions = problemBounds, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET")

    for _ in tqdm(range(numIters)):

        # Ask for the next point.
        nextPoint = optimizer.ask()

        if len(optimizer.models) != 0:
            # Clear old models, to prevent memory overload.
            optimizer.models = [optimizer.models[-1]]

        pointVal = TwoFilterFitness(nextPoint)

        result = optimizer.tell(nextPoint, pointVal)

    print("Result determined for cascaded filters! Here are the stats:")
    print("Best figure of merit: " + str(-1 * result.fun))
    print("Parameters giving this result: " + str(result.x))

    bestIndex = np.argwhere(result.func_vals == result.fun)[0][0]
    # Create a Pandas dataframe to hold the optimiser results.
    resultDataframe = pd.DataFrame(optimizer.Xi, columns = [r"$|\textbf{B}_{\textrm{1}}|$", r"$T_{\textrm{1}}$", r"$\theta_{\textrm{E}}$", 
    r"$\theta_{\textrm{B}_1}$", r"$\phi_{\textrm{B}_1}$", r"$|\textbf{B}_{\textrm{2}}|$", r"$T_{\textrm{2}}$", r"$\theta_{\textrm{B}_2}$", r"$\phi_{\textrm{B}_2}$"])

    # Save the model in a pickle file.
    joblib.dump((optimizer.models[-1], resultDataframe, bestIndex), "shap_data.pkl")
    print("Data saved to shap_data.pkl.")

    # Predict the best value.
    expectMinLoc, expectMinVal = skopt.expected_minimum(result)

    print("Predicted best point: {}".format(expectMinLoc))
    print("Predicted (abs) value at this point: {}".format(abs(expectMinVal)))
    print("Actual (abs) value at this point: {}".format(abs(TwoFilterFitness(expectMinLoc))))

    return

if __name__ == "__main__":

    # Test the fitness function against literature.
    #LiteratureTest()

    # Run the optimisation.
    #Optimise(6000)

    # Determine the sensitivity of variables.
    #Sensitivity([314, 109, 50, 86, 59, 199, 77, 77, 2])

    # Obtain SHAP importance.
    ShapImportance(1000)