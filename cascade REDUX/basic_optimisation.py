"""
This module is designed to ensure that the optimisation algorithms being used can actually optimise the filters in a reasonable time.
This is done by tasking the optimisation algorithm with obtaining the figure of merits obtained in the literature.
"""

from elecsus import elecsus_methods as elecsus
import numpy as np
import skopt
from scipy.integrate import simps as integrate
from matplotlib import pyplot as plt
from skopt.plots import plot_convergence, plot_objective
from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
from functools import partial
from tqdm import tqdm

# Global parameters.
baseParams = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
globalDetuning = np.linspace(-25000, 25000, 1000)

def ElecsusFitnessSingle(inputParams, toTest = False):
    """
    The figure of merit generation function as required by skopt. The input is a list of each parameter. This is for a single filter.
    """

    elecsusParams = baseParams.copy()

    # NOTE: No bPhi as of yet.
    paramDict = {'Bfield': inputParams[0], "T": inputParams[1], 'Btheta': np.deg2rad(inputParams[2]), 'Etheta': np.deg2rad(inputParams[3])}

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

    if toTest:
        print("ENBW obtained! Value: " + str(ENBW))
        print("FoM obtained! Value: " + str(figureOfMerit))
        print("Maximum transmission: " + str(singleFilterTransmission.max()))

        # Plot the output.
        plt.plot(globalDetuning/1e3, singleFilterTransmission.real)

        # Make the graph pretty.
        plt.xlabel("Detuning (GHz)")
        plt.ylabel("Transmission")

        plt.show()
    
    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print(str(elecsusParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def TestFitness():
    """
    Ensure the fitness function is correct by reproducing literature value for rubidium.
    """

    # First define the input parameters. It is a list with the following values:
    # [Bfield, T, Btheta, Etheta], with the angles in degrees.
    inputParams = [230, 126, 83, 6]

    ElecsusFitnessSingle(inputParams, toTest = True)

    return

def bStrengthFitness(inputParams):
    """
    A 1D fitness function for optimising.
    NOTE: The returned value is negative due to the nature of skopt.
    """

    elecsusParams = baseParams.copy()

    # NOTE: No bPhi as of yet.
    paramDict = {'Bfield': inputParams[0], "T": 126, 'Btheta': np.deg2rad(83), 'Etheta': np.deg2rad(6)}

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

def tFitness(inputParams):
    """
    A 1D fitness function for optimising.
    NOTE: The returned value is negative due to the nature of skopt.
    """

    elecsusParams = baseParams.copy()

    # NOTE: No bPhi as of yet.
    paramDict = {'Bfield': 230, "T": inputParams[0], 'Btheta': np.deg2rad(83), 'Etheta': np.deg2rad(6)}

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

def bThetaFitness(inputParams):
    """
    A 1D fitness function for optimising.
    NOTE: The returned value is negative due to the nature of skopt.
    """

    elecsusParams = baseParams.copy()

    # NOTE: No bPhi as of yet.
    paramDict = {'Bfield': 230, "T": 126, 'Btheta': np.deg2rad(inputParams[0]), 'Etheta': np.deg2rad(6)}

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
    
def eThetaFitness(inputParams):
    """
    A 1D fitness function for optimising.
    NOTE: The returned value is negative due to the nature of skopt.
    """

    elecsusParams = baseParams.copy()

    # NOTE: No bPhi as of yet.
    paramDict = {'Bfield': 230, "T": 126, 'Btheta': np.deg2rad(83), 'Etheta': np.deg2rad(inputParams[0])}

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

def Optimise1D(numIters):
    """
    Runs optimisation algorithms for the 4 variables used in the literature, with the aim of reproducing the literature value.
    """

    # Define the problems to optimise, and their bounds.
    problemList = [(bStrengthFitness, (10, 1300)), (tFitness, (50, 230)), (bThetaFitness, (0, 90)), (eThetaFitness, (0, 90))]

    for problem in problemList:
        print("Optimising: " + str(problem[0].__name__))
        result = skopt.gp_minimize(problem[0], [problem[1]], verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)))

    return

def FullOptimisation(numIters):
    """
    Runs the full (4D) optimisation of the rubidium filter with the aim of reproducing the literature value.
    """

    # Define the problem bounds.
    problemBounds = [(10, 1300), (40, 230), (0, 90), (0, 90)]

    print("Optimising a single filter with 4 variables.")
    result = skopt.forest_minimize(ElecsusFitnessSingle, problemBounds, verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)))

    return
    
def ConvergencePlot(numIters):
    """
    Compare the different strategies available in the skopt library. These are:
    - Random Sampling.
    - Bayesian Optimisation.
    - Gradient Boosted Trees.
    - Random Forest.
    - Extra Trees
    """
        
    # Define the problem bounds.
    problemBounds = [(10, 1300), (40, 230), (0, 90), (0, 90)]

    def run(minimizer, n_iter = 2):
        return [minimizer(ElecsusFitnessSingle, problemBounds, n_calls = numIters, random_state = n, verbose = True) 
            for n in range(n_iter)]

    # Random search.
    dummy_res = run(skopt.dummy_minimize) 

    # Gaussian processes.
    gp_res = run(skopt.gp_minimize)

    # Random forest.
    rf_res = run(partial(skopt.forest_minimize, base_estimator="RF"))

    # Extra trees .
    et_res = run(partial(skopt.forest_minimize, base_estimator="ET"))

    # Gradient boosted trees.
    gb_res = run(skopt.gbrt_minimize)

    plot = plot_convergence(("Random search", dummy_res),
                        ("Bayesian Optimisation", gp_res),
                        ("Random Forest", rf_res),
                        ("Extra Trees", et_res),
                        ("Gradient Boosted", gb_res), 
                        true_minimum = -1.04)

    plot.legend(loc = "best", prop = {'size': 6}, numpoints=1)

    plt.show()

    return

def AcquisitionPlot(numIters):
    """
    A plot showing the acquisition function and what it predicts as the next best point.
    """

    # Set up the frame.
    fig = plt.figure()
    objPlot = fig.add_subplot(2, 1, 1)
    objPlot.set_xlim(10, 1300)
    acqPlot = fig.add_subplot(2, 1, 2)
    acqPlot.set_xlim(10, 1300)

    # Generate a model for each acquisition function.
    # Define the problem bounds.
    problemBounds = [(10, 1300)]

    # Give each the same random state.
    ranState = np.random.randint(1e6)
    print("Random seed:")
    print(ranState)

    def run(minimizer):
        return minimizer(bStrengthFitness, problemBounds, n_calls = numIters, random_state = ranState, verbose = True) 

    # Generate the objective function.
    bRange = np.linspace(10 ,1300, num = 1000)
    funcVals = []
    print("Plotting objective function:")
    for b in tqdm(bRange):
        funcVals.append(bStrengthFitness([b]))

    objPlot.plot(bRange, np.abs(funcVals))

    # Reshape the input due to it having a single variable (feature).
    bRange = bRange.reshape(-1, 1)

    # Plot the LCB acquisition function.
    lcbRes = run(partial(skopt.forest_minimize, base_estimator = "ET", acq_func = "LCB"))
    lcbModel = lcbRes.models[-1]
    lcbFunc = gaussian_lcb(bRange, lcbModel)
    acqPlot.plot(bRange, np.divide(lcbFunc, lcbFunc.min()), label = "LCB")

    # Plot the EI acquisition function.
    eiRes = run(partial(skopt.forest_minimize, base_estimator = "ET", acq_func = "EI"))
    eiModel = eiRes.models[-1]
    eiFunc = gaussian_ei(bRange, eiModel)
    acqPlot.plot(bRange, np.divide(eiFunc, eiFunc.max()), label = "EI")

    # Plot the PI acquisition function.
    piRes = run(partial(skopt.forest_minimize, base_estimator = "ET", acq_func = "PI"))
    piModel = piRes.models[-1]
    piFunc = gaussian_pi(bRange, piModel)
    acqPlot.plot(bRange, np.divide(piFunc, piFunc.max()), label = "PI")

    acqPlot.legend()
    plt.show()

    return

def VisualiseResults(numIters):
    """
    Uses the built in plot_objective function in skopt to plot the optimisation process for a single filter.
    """

    # Define the problem bounds.
    problemBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.)]

    # Run the optimisation.
    result = skopt.forest_minimize(ElecsusFitnessSingle, problemBounds, verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET")

    # Plot the result.
    plot_objective(result)

    plt.show()

    return

if __name__ == "__main__":
    # Test the fitness function.
    TestFitness()

    # Run the optimisation algorithms in 1D.
    #Optimise1D(50)

    # Run the optimisation of the 'full' single filter problem.
    FullOptimisation(1000)

    # Create a convergence plot of the strategies available on skopt.
    #ConvergencePlot(50)

    # Show what the acquisition functions see.
    #AcquisitionPlot(50)

    # Visualise the results of a single optimisation.
    #VisualiseResults(1000)