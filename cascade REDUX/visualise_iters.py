"""
This module will generate a model for a simple 1D (or 2D) problem, showing the structure of the model after a certain numbers of iterations.
"""

import numpy as np
from elecsus import elecsus_methods as elecsus
import skopt
from scipy.integrate import simps as integrate
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz    
import json
import shap
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("text", usetex = True)
import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")


# Global parameters.
baseParams = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
globalDetuning = np.linspace(-25000, 25000, 1000)

def Fitness1D(inputParams):
    """
    A 1D fitness function for optimising.
    NOTE: The returned value is negative due to the nature of skopt.
    """

    elecsusParams = baseParams.copy()

    # NOTE: No bPhi as of yet.
    paramDict = {'Bfield': inputParams[0], "T": 126, 'Btheta': np.deg2rad(83), 'Etheta': np.deg2rad(6), 'Bphi': np.deg2rad(0)}

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

def Fitness2D(inputParams):
    """
    A 1D fitness function for optimising.
    NOTE: The returned value is negative due to the nature of skopt.
    """

    elecsusParams = baseParams.copy()

    # NOTE: No bPhi as of yet.
    paramDict = {'Bfield': inputParams[0], "T": 126, 'Btheta': np.deg2rad(inputParams[1]), 'Etheta': np.deg2rad(6), 'Bphi': np.deg2rad(0)}

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

def GenerateModels1D(numIters):
    """
    Runs the optimisation and saves the model function to a JSON file. Also saves the plot of the real function, and the images of the decision tree sizes.
    """

    # Set up run seed for reproducibility.
    seed = np.random.randint(1e6)
    print("Random seed: " + str(seed))

    # First, run the optimisation.
    result = skopt.forest_minimize(Fitness1D, [(10., 1000.)], n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), random_state = seed)
    print("Optimisation complete! Best value: " + str(result.fun))

    # Obtain models after the random seeding, half the maximum iterations, and the maximum iterations.
    modelRandom = result.models[int(np.ceil(numIters/10)) - 1]
    modelHalf = result.models[int(np.ceil(numIters/2)) - 1]
    modelEnd = result.models[-1]

    # Generate a graph of the objective function values as well as the model predictions of the function.
    bRange = np.linspace(10, 1000, num = 100)
    objVals = []
    for b in bRange:
        objVals.append(abs(Fitness1D([b])))
    
    # Objective function mapping complete.
    print("Objective function mapping complete! Moving on to models...")
    randomVals = np.abs(modelRandom.predict(bRange.reshape(-1, 1)))
    halfVals = np.abs(modelHalf.predict(bRange.reshape(-1, 1)))
    endVals = np.abs(modelEnd.predict(bRange.reshape(-1, 1)))

    print("Model predictions complete! Sending to JSON...")
    # Create the data dictionary. Numpy arrays cannot be serialised to JSON so we must use .tolist() to make them so.
    dataDict = {"numIters": numIters, "bRange": bRange.tolist(), "objVals": objVals, "randomVals": randomVals.tolist(), "halfVals": halfVals.tolist(), "endVals": endVals.tolist()}
    with open("visualise_iters.txt", "w") as outputFile:
        # NOTE: This is in write mode, so will erase all previous data on that file.
        json.dump(dataDict, outputFile)

    # JSON dump complete. Create the tree DOT files.
    print("Printing a tree from each forest to a DOT file...")
    randomTree = modelRandom.estimators_[-1]
    export_graphviz(randomTree, out_file = "random_tree.dot", precision = 2, impurity = False)
    halfTree = modelHalf.estimators_[-1]
    export_graphviz(halfTree, out_file = "half_tree.dot", precision = 2, impurity = False)
    endTree = modelEnd.estimators_[-1]
    export_graphviz(endTree, out_file = "end_tree.dot", precision = 2, impurity = False)

    return

def GenerateModels2D(numIters):
    """
    Runs the optimisation and saves the model function to a JSON file. Also saves the plot of the real function, and the images of the decision tree sizes.
    """

    # Set up run seed for reproducibility.
    seed = np.random.randint(1e6)
    print("Random seed: " + str(seed))

    # First, run the optimisation.
    result = skopt.forest_minimize(Fitness2D, [(10., 1300.), (0., 180.)], n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), random_state = seed)
    print("Optimisation complete! Best value: " + str(result.fun))

    # Obtain models after the random seeding, half the maximum iterations, and the maximum iterations.
    # modelRandom = result.models[int(np.ceil(numIters/10)) - 1]
    # modelHalf = result.models[int(np.ceil(numIters/2)) - 1]
    modelEnd = result.models[-1]

    # # Generate a graph of the objective function values as well as the model predictions of the function.
    # bRange = np.linspace(10, 1000, num = 500)
    # tRange = np.linspace(40, 230, num = 500)

    # objVals = []
    # for B in bRange:
    #     for T in tRange:
    #         print(B)
    #         objVals.append(abs(Fitness2D([B, T])))

    
    # Objective function mapping complete.
    # print("Objective function mapping complete! Moving on to models...")

    # # Predict the function at every step of the model.
    # predictVals = []
    # for model in result.models:
    #     modelVals = []
    #     for B in bRange:
    #         for T in tRange:
    #             modelVals.append(np.abs(model.predict(np.array([B, T]).reshape(1, -1))).tolist())

    #     predictVals.append(modelVals)

    # print("Model predictions complete! Sending to JSON...")
    # # Create the data dictionary. Numpy arrays cannot be serialised to JSON so we must use .tolist() to make them so.
    # dataDict = {"numIters": numIters, "bRange": bRange.tolist(), "tRange": tRange.tolist(), "objVals": objVals, "predictVals": predictVals}
    # with open("visualise_iters_2d.txt", "w") as outputFile:
    #     # NOTE: This is in write mode, so will erase all previous data on that file.
    #     json.dump(dataDict, outputFile)

    # JSON dump complete. Create the tree DOT files.
    print("Printing a tree from each (important) forest to a DOT file...")
    # randomTree = modelRandom.estimators_[-1]
    # export_graphviz(randomTree, out_file = "random_tree_2d.dot", precision = 2, impurity = False)
    # halfTree = modelHalf.estimators_[-1]
    # export_graphviz(halfTree, out_file = "half_tree_2d.dot", precision = 2, impurity = False)
    endTree = modelEnd.estimators_[-1]
    export_graphviz(endTree, out_file = "end_tree_2d.dot", precision = 2, impurity = False)

    resultDataframe = pd.DataFrame(result.x_iters, columns = [r"$|\textbf{B}_{\textrm{1}}|$", r"$\theta_{\textrm{B}}$"])
    explainer = shap.TreeExplainer(modelEnd)
    shapValues = explainer.shap_values(resultDataframe)

    shap.summary_plot(shapValues, resultDataframe)

    return

if __name__ == "__main__":
    #GenerateModels1D(100)

    GenerateModels2D(1000)