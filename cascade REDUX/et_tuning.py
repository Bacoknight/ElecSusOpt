"""
ExtraTrees is now being used as the main optimisation algorithm for ElecSus. However, it is currently
being used with its default parameters. This module seeks to find the best set of parameters to use.
"""

from elecsus import elecsus_methods as elecsus
import numpy as np
import skopt
from matplotlib.pyplot import cm
from scipy.integrate import simps as integrate
from matplotlib import pyplot as plt
import time
from functools import partial
from tqdm import tqdm
from mpi4py import MPI
from random import shuffle
import json

# Global parameters.
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 50e-3, "Dline": "D2", "rb85frac": 72.17}
globalDetuning = np.linspace(-25000, 25000, 1000)

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

def TwoFilterFitnessTime(inputParams):
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
    Also returns time for algorithms which seek time optimisation.
    """

    startTime = time.time()

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
        return 0.0, time.time() - startTime

    # Call ElecSus to obtain the output field from the second filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE2] = elecsus.calculate(globalDetuning, outputE1, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the second filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE1))
        return 0.0, time.time() - startTime

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
        return 0.0, time.time() - startTime
    else:
        return -1.0 * figureOfMerit, time.time() - startTime

def PlotConvergence(numIters, numRuns, toPlot = True):
    """
    Compares the acquisition functions of the ExtraTrees algorithm by plotting their optimisation path.
    """

    seedList = np.random.randint(1e6, size = numRuns)
    print("Seed list:")
    print(seedList)

    # Define the bounds of the problem.
    problemBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]

    # Give each test its own colour.
    colours = cm.viridis(np.linspace(0.25, 1.0, 5)).tolist()

    # Define each test. Each test has its own list for storing its own values.
    optimiserList = [(partial(skopt.forest_minimize, base_estimator = "ET", acq_func = "PI", func = TwoFilterFitness), "PI", colours[0], [], []), 
    (partial(skopt.forest_minimize, base_estimator = "ET", acq_func = "EI", func = TwoFilterFitness), "EI", colours[1], [], []), 
    (partial(skopt.forest_minimize, base_estimator = "ET", acq_func = "LCB", func = TwoFilterFitness), "LCB", colours[2], [], []), 
    (partial(skopt.forest_minimize, base_estimator = "ET", acq_func = "EIps", func = TwoFilterFitnessTime), "EIps", colours[3], [], []), 
    (partial(skopt.forest_minimize, base_estimator = "ET", acq_func = "PIps", func = TwoFilterFitnessTime), "PIps", colours[4], [], [])]

    # Define things used for plotting.
    realIters = range(1, numIters + 1)
    if toPlot:
        fig = plt.figure("Acquisition function comparison")
        fig.set_size_inches(19.20, 10.80)
        pathPlot = plt.subplot(121)
        timePlot = plt.subplot(122)
    timeList = []
    nameList = []
    barColourList = []

    for run in tqdm(range(numRuns)):
        # Shuffle the list.
        shuffle(optimiserList)
        for optimiser in tqdm(optimiserList):
            # Complete a single run.
            startTime = time.time()
            result = optimiser[0](dimensions = problemBounds, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), random_state = seedList[run])
            timeElapsed = time.time() - startTime

            # Plot the process.
            bestVal = [np.max(abs(result.func_vals[:i])) for i in realIters]
            if toPlot:
                pathPlot.plot(realIters, bestVal, c = optimiser[2], alpha = 0.2)

            # Append the results to the optimisers own lists.
            optimiser[4].append(bestVal)
            optimiser[3].append(timeElapsed)

    # All runs complete.
    print("\n")
    for optimiser in optimiserList:
        avgTime = np.mean(optimiser[3])
        print(optimiser[1] + " average FoM per unit time: " + str(np.mean(optimiser[4], axis = 0)[-1]/avgTime))
        if toPlot:
            # TODO: Plot max/min times error bars for bar chart.
            timeList.append(avgTime)
            nameList.append(optimiser[1])
            barColourList.append(optimiser[2])
            # Plot the average result.
            pathPlot.plot(realIters, np.mean(optimiser[4], axis = 0), c = optimiser[2], marker = ".", markersize = 12, markevery = int(np.ceil(numIters/10)), lw = 2, label = optimiser[1])

    if toPlot:
        # Plotting.
        timePlot.bar(nameList, timeList, align = "center", color = barColourList)
        timePlot.set_xlabel("Acquisition Function")
        timePlot.set_ylabel("Average runtime for " + str(numIters) + " iterations (s)")
        
        pathPlot.legend()
        pathPlot.set_xlim(1, numIters)
        pathPlot.set_ylim(bottom = 0)
        pathPlot.set_xlabel("Iteration number")
        pathPlot.set_ylabel(r"Best Figure of Merit (GHz$^{-1}$)")
        
        plt.tight_layout()
        plt.savefig("acq_plot.pdf")
        plt.show()
    else:
        # Save everything to a JSON file.
        # Remove the non-serializable objects.
        optimiserList = [optimiser[1:] for optimiser in optimiserList]
        dataDict = {"seedList": seedList.tolist(), "optimiserList": optimiserList, "numIters": numIters}
        with open("acq_plot.txt", "w") as outputFile:
            # Write to a file.
            # NOTE: This is in write mode, so will erase all previous data on that file.
            json.dump(dataDict, outputFile)
    
    return

def ParameterTuning(numIters, numRuns, numPoints, toPlot = True):
    """
    Plots a colourmap of the acquisition function parameter (kappa or xi depending on the type) and the fraction of iterations that are random probes
    and the resulting maximal figure of merit. 
    numPoints is the number of different fractions and xi values to test (will be square).
    NOTE: Minimum number of bootstrap iterations is 1.
    """
    
    seedList = np.random.randint(1e6, size = numRuns)
    print("Seed list:")
    print(seedList)

    # Define the bounds of the problem.
    problemBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]

    fracList = np.linspace(0, 1, num = numPoints)
    kappaList = np.linspace(0, 5, num = numPoints)

    avgList = []

    for kappa in tqdm(kappaList):
        for frac in tqdm(fracList):
            bestFoMs = [abs(skopt.forest_minimize(base_estimator = "ET", acq_func = "LCB", func = TwoFilterFitness, dimensions = problemBounds, n_calls = numIters, n_random_starts = np.max([int(np.ceil(numIters * frac)), 1]), 
            kappa = kappa, random_state = seedList[run]).fun) for run in range(numRuns)]
            avgFoM = np.mean(bestFoMs, axis = 0)
            avgList.append(avgFoM)

    kappas, fracs = np.meshgrid(kappaList, fracList)

    print("\nMaximum average FoM found: " + str(np.max(avgList)))
    print("Value of Kappa: " + str(np.array(kappas).flatten()[np.argmax(avgList)]))
    print("Fraction of numIters: " + str(np.array(fracs).flatten()[np.argmax(avgList)]))

    avgList = np.reshape(avgList, (numPoints, numPoints))

    if toPlot:
        plt.pcolormesh(kappas, fracs, avgList, shading = "gouraud")
        plt.xlabel("Kappa")
        plt.ylabel("Fraction of iterations used for random sampling")
        cb = plt.colorbar()
        cb.set_label(r"Average Figure of Merit (GHz$^{-1}$)", rotation = 270, labelpad = 15)
        plt.tight_layout()
        plt.savefig("kappa_frac_plot.pdf")
        plt.show()
    else:
        # Save everything onto a JSON file.
        dataDict = {"seedList": seedList.tolist(), "kappas": kappas.tolist(), "fracs": fracs.tolist(), "avgList": avgList.tolist()}
        with open("kappa_frac_plot.txt", "w") as outputFile:
            # NOTE: This is in write mode, so will erase all previous data on that file.
            json.dump(dataDict, outputFile)

    return

def ParameterFitness(inputParams):
    """
    The fitness function to optimise the hyperparameters of the optimiser.
    The seeds are 0, 1, 2, 3, 4.
    """
    
    # Explicitly define number of iterations.
    numIters = 100

    # Define the bounds of the problem.
    problemBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]

    results = [skopt.forest_minimize(base_estimator = "ET", acq_func = "LCB", func = TwoFilterFitness, dimensions = problemBounds, n_calls = numIters, n_random_starts = np.max([int(np.ceil(numIters * inputParams[0])), 1]), 
            kappa = inputParams[1], random_state = run) for run in range(5)]

    minIters = [np.argwhere(result.func_vals == result.fun)[0][0] + 1 for result in results]
    bestFoMs = [result.fun for result in results]

    return  np.mean(np.divide(bestFoMs, minIters), axis = 0)

def BayesianTuning(numIters):
    """
    Determine the best kappa and fractions using Gaussian Processes.
    """

    # Define the bounds of the problem. First is fraction, second is kappa.
    problemBounds = [(0., 1.), (0., 20.)]

    result = skopt.gp_minimize(ParameterFitness, problemBounds, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), verbose = True)

    print("Optimisation Complete! Here are the stats:")
    print(result.x)
    print(result.fun)

    return

if __name__ == "__main__":
    # Plot the convergence paths.
    #PlotConvergence(1000, 10, toPlot = True)
    #PlotConvergence(10, 2, False)

    # Plot parameter dependence.
    #ParameterTuning(50, 2, 25, toPlot = True)
    #ParameterTuning(50, 2, 10, toPlot = False)

    # Test the Bayesian fitness function.
    #ParameterFitness([0.1, 1.92])

    # Use Bayesian Optimisation to tune the algorithm.
    BayesianTuning(500)
