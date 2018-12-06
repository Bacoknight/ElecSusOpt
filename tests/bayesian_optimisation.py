import numpy as np
from scipy.integrate import simps as integrate
from elecsus import elecsus_methods as elecsus
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import time
from tqdm import tqdm

"""
This module will use the Bayesian optimisation algorithm, given by an external python library, to determine the 
optimal experimental parameters for a given atomic filter.
"""

# Here we define some global variables so that it is easier to change it for all functions.
# Detuning used for all tests.
globalDetuning = np.arange(-10000, 10000, 10) # MHz

def ProduceSpectrum(detuning, params, toPlot=False):
        """
        Produce a simple transmission output using ElecSus.
        We always assume that the polariser after the filter is perpendiucular to the input
        angle of the light.
        """

        # Use the input of the function to determine the polarisation of the input light.
        E_in = np.array([np.cos(params["Etheta"]), np.sin(params["Etheta"]), 0])

        # Determine the effect of the final polariser on the output field using a Jones matrix.
        outputAngle = params['Etheta'] + np.pi/2
        J_out = np.matrix([[np.cos(outputAngle)**2, np.sin(outputAngle)*np.cos(outputAngle)],
                        [np.sin(outputAngle)*np.cos(outputAngle), np.sin(outputAngle)**2]])

        # Call ElecSus to find the output electric field from the cell.
        try:
	        [E_out] = elecsus.calculate(detuning, E_in, params, outputs=['E_out'])

        except:
                # There was an issue obtaining the field from ElecSus.
	        return 0

        transmittedE =  np.array(J_out * E_out[:2])
        transmission =  (transmittedE * transmittedE.conjugate()).sum(axis=0)

        if toPlot:
                # Plot the result.
                plt.plot(detuning, transmission)
                plt.show()

        return transmission

def CalculateFoM(detuning, params):

        # Get the overall transmission.
        transmission = ProduceSpectrum(detuning, params, False)

        maxTransmission = np.max(transmission)
        ENBW = integrate(transmission, detuning)/maxTransmission
        FOM = maxTransmission/ENBW # This is in 1/MHz, so we multiply by 1000 for 1/GHz

        if np.isnan(FOM):
                # Occurs if there is just a flat line for the transmission. Usually occurs for high temp and high B field.
                return 0
        else:
                return FOM.real * 1000

def TestFoM():
        """
        This function aims to reproduce the values of the Table 1 in the paper in Opt. Lett. 2018
        Doc ID: 335953.
        """
        # Define the parameters for each test.
        paramsList = [{'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'},
        {'Bfield':695, 'rb85frac':72.17, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(45), 'lcell':5e-3, 'T':147, 'Dline':'D1', 'Elem':'Rb'},
        {'Bfield':224, 'rb85frac':100, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(90), 'lcell':5e-3, 'T':121, 'Dline':'D2', 'Elem':'Rb'},
        {'Bfield':310, 'rb85frac':100, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(46), 'lcell':5e-3, 'T':139, 'Dline':'D1', 'Elem':'Rb'},
        {'Bfield':849, 'rb85frac':0, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(81), 'lcell':5e-3, 'T':121, 'Dline':'D2', 'Elem':'Rb'},
        {'Bfield':315, 'rb85frac':0, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(45), 'lcell':5e-3, 'T':141, 'Dline':'D1', 'Elem':'Rb'},
        {'Bfield':1120, 'Btheta':np.deg2rad(87), 'Etheta':np.deg2rad(89), 'lcell':5e-3, 'T':127, 'Dline':'D2', 'Elem':'Cs'},
        {'Bfield':338, 'Btheta':np.deg2rad(89), 'Etheta':np.deg2rad(46), 'lcell':5e-3, 'T':121, 'Dline':'D1', 'Elem':'Cs'},
        {'Bfield':88, 'Btheta':np.deg2rad(1), 'Etheta':np.deg2rad(87), 'lcell':5e-3, 'T':150, 'Dline':'D2', 'Elem':'K'},
        {'Bfield':460, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(47), 'lcell':5e-3, 'T':177, 'Dline':'D1', 'Elem':'K'},
        {'Bfield':144, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(0), 'lcell':5e-3, 'T':245, 'Dline':'D2', 'Elem':'Na'},
        {'Bfield':945, 'Btheta':np.deg2rad(88), 'Etheta':np.deg2rad(41), 'lcell':5e-3, 'T':279, 'Dline':'D1', 'Elem':'Na'}]

        for elecsusParams in paramsList:
                print("FoM for " + str(elecsusParams["Elem"]) + ", " + str(elecsusParams["Dline"]) + " line: " + str(CalculateFoM(globalDetuning, elecsusParams)))

        return

def BayesianObjective1D(Bfield):
        """
        This function is the function which will be optimised using the Bayesian optimisation algorithm.
        TODO: See if it is faster to put the entire function in here rather than calling other functions?
        """
        inputParams = {'Bfield':Bfield, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}

        return CalculateFoM(globalDetuning, inputParams)


def TestOpt1D():
        """
        This function uses Bayesian optimisation to reproduce the D2 rubidium entry in Table 1 of the paper defined in TestFoM.
        To do this, all parameters (bar 1 for 1D, 2 for 2D etc..) are fixed at their optimum as given in the literature, and the Bayesian
        optimisation algorithm is tasked with finding the remaining variable(s) such that the FoM is maximised. At worst, this should
        return the same figure of merit as shown in the literature (FoM = 1.04). Note that the FoM has units of 1/GHz.
        """

        # First we must define the bounds over which the algorithm will search.
        inputBounds = {"Bfield": (0, 1e4)}

        # Start timing.
        print("Beginning optimisation for 1D...")
        startTime = time.time()

        # Start the optimisation algorithm.
        optimiser = BayesianOptimization(f = BayesianObjective1D, pbounds = inputBounds, random_state = np.random.randint(1, 1e4), verbose = 2)
        optimiser.maximize(init_points = 50, n_iter = 50)

        # Print results.
        elapsedTime = time.time() - startTime
        print("Optimisation complete!")
        print("Elapsed time: " + str(elapsedTime))
        print("Optimal result: " + str(optimiser.max["target"]))
        print("Obtained with values: " + str(optimiser.max["params"]))

        indexList = []
        paramList = []
        for index, result in enumerate(optimiser.res):
                indexList.append(index + 1)
                paramList.append(result["target"])

        plt.plot(indexList, paramList)

        plt.xlabel("Iteration number")
        plt.ylabel(r"FoM (GHz$^{-1}$)")

        plt.show()

        return

def OptGraph1D():
        """
        This function creates a graph showing the area we want to optimise, and the result of the optimisation.
        """

        # Begin by generating the optimiser model.
        inputBounds = {"Bfield": (0, 1e4)}
        optimiser = BayesianOptimization(BayesianObjective1D, inputBounds, random_state = np.random.randint(1, 1e4), verbose = 2)
        optimiser.maximize(init_points = 10, n_iter = 40)

        # Use the model to show how the model sees the objective function.
        bRange = np.linspace(0, 1e4, num = 500)
        mean, stdDev = optimiser._gp.predict(bRange.reshape(-1, 1), return_std = True)

        # Plot the objective function.
        FoMList = []
        for b in tqdm(bRange):
                FoMList.append(BayesianObjective1D(b))

        # Plot all the graphs.
        plt.plot(bRange, FoMList)
        plt.plot(bRange, mean)
        plt.fill_between(bRange, mean + stdDev, mean - stdDev, alpha=0.3)
        plt.scatter(optimiser.space.params.flatten(), optimiser.space.target, c = "red", s = 50, zorder = 10)

        # Plot the value used in the literature.
        plt.axvline(230, c = "m")

        plt.xlim(0, 1e4)
        plt.ylim(ymin = 0)
        plt.xlabel("Magnetic field strength (Gauss)")
        plt.ylabel(r"FoM (GHz$^{-1}$)")

        plt.show()

        return

def BayesianObjective2D(Bfield, temp):
        """
        This function is the function which will be optimised using the Bayesian optimisation algorithm.
        TODO: See if it is faster to put the entire function in here rather than calling other functions?
        """
        inputParams = {'Bfield':Bfield, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':temp, 'Dline':'D2', 'Elem':'Rb'}

        return CalculateFoM(globalDetuning, inputParams)

def TestOpt2D():
        """
        This function uses Bayesian optimisation to reproduce the D2 rubidium entry in Table 1 of the paper defined in TestFoM.
        To do this, all parameters (bar 1 for 1D, 2 for 2D etc..) are fixed at their optimum as given in the literature, and the Bayesian
        optimisation algorithm is tasked with finding the remaining variable(s) such that the FoM is maximised. At worst, this should
        return the same figure of merit as shown in the literature (FoM = 1.04). Note that the FoM has units of 1/GHz.
        """

        # First we must define the bounds over which the algorithm will search.
        inputBounds = {"Bfield": (0, 0.6e4), "temp": (20, 400)}

        # Start timing.
        print("Beginning optimisation for 2D...")
        startTime = time.time()

        # Start the optimisation algorithm.
        optimiser = BayesianOptimization(f = BayesianObjective2D, pbounds = inputBounds, random_state = np.random.randint(1, 1e4), verbose = 2)
        optimiser.maximize(init_points = 200, n_iter = 300)

        # Print results.
        elapsedTime = time.time() - startTime
        print("Optimisation complete!")
        print("Elapsed time: " + str(elapsedTime))
        print("Optimal result: " + str(optimiser.max["target"]))
        print("Obtained with values: " + str(optimiser.max["params"]))

        indexList = []
        paramList = []
        for index, result in enumerate(optimiser.res):
                indexList.append(index + 1)
                paramList.append(result["target"])

        plt.plot(indexList, paramList)

        plt.xlabel("Iteration number")
        plt.ylabel(r"FoM (GHz$^{-1}$)")

        plt.show()

        return

def BayesianObjective3D(Bfield, Btheta, temp):
        """
        This function is the function which will be optimised using the Bayesian optimisation algorithm.
        TODO: See if it is faster to put the entire function in here rather than calling other functions?
        """
        inputParams = {'Bfield':Bfield, 'rb85frac':72.17, 'Btheta':np.deg2rad(Btheta), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':temp, 'Dline':'D2', 'Elem':'Rb'}

        return CalculateFoM(globalDetuning, inputParams)

def TestOpt3D():
        """
        This function uses Bayesian optimisation to reproduce the D2 rubidium entry in Table 1 of the paper defined in TestFoM.
        To do this, all parameters (bar 1 for 1D, 2 for 2D etc..) are fixed at their optimum as given in the literature, and the Bayesian
        optimisation algorithm is tasked with finding the remaining variable(s) such that the FoM is maximised. At worst, this should
        return the same figure of merit as shown in the literature (FoM = 1.04). Note that the FoM has units of 1/GHz.
        """

        # First we must define the bounds over which the algorithm will search.
        inputBounds = {"Bfield": (0, 1e4), "Btheta": (0, 90), "temp": (20, 400)}

        # Start timing.
        print("Beginning optimisation for 3D...")
        startTime = time.time()

        # Start the optimisation algorithm.
        optimiser = BayesianOptimization(f = BayesianObjective3D, pbounds = inputBounds, random_state = np.random.randint(1, 1e4), verbose = 2)
        optimiser.maximize(init_points = 100, n_iter = 400)

        # Print results.
        elapsedTime = time.time() - startTime
        print("Optimisation complete!")
        print("Elapsed time: " + str(elapsedTime))
        print("Optimal result: " + str(optimiser.max["target"]))
        print("Obtained with values: " + str(optimiser.max["params"]))

        indexList = []
        paramList = []
        for index, result in enumerate(optimiser.res):
                indexList.append(index + 1)
                paramList.append(result["target"])

        plt.plot(indexList, paramList)

        plt.xlabel("Iteration number")
        plt.ylabel(r"FoM (GHz$^{-1}$)")

        plt.show()

        return

def BayesianObjective4D(Bfield, Btheta, Etheta, temp):
        """
        This function is the function which will be optimised using the Bayesian optimisation algorithm.
        TODO: See if it is faster to put the entire function in here rather than calling other functions?
        """
        inputParams = {'Bfield':Bfield, 'rb85frac':72.17, 'Btheta':np.deg2rad(Btheta), 'Etheta':np.deg2rad(Etheta), 'lcell':5e-3, 'T':temp, 'Dline':'D2', 'Elem':'Rb'}

        return CalculateFoM(globalDetuning, inputParams)

def TestOpt4D():
        """
        This function uses Bayesian optimisation to reproduce the D2 rubidium entry in Table 1 of the paper defined in TestFoM.
        To do this, all parameters (bar 1 for 1D, 2 for 2D etc..) are fixed at their optimum as given in the literature, and the Bayesian
        optimisation algorithm is tasked with finding the remaining variable(s) such that the FoM is maximised. At worst, this should
        return the same figure of merit as shown in the literature (FoM = 1.04). Note that the FoM has units of 1/GHz.
        """

        # First we must define the bounds over which the algorithm will search.
        inputBounds = {"Bfield": (0, 1e4), "Btheta": (0,90), "Etheta": (0,90), "temp": (20, 400)}

        # Start timing.
        print("Beginning optimisation for 4D...")
        startTime = time.time()

        # Start the optimisation algorithm.
        optimiser = BayesianOptimization(f = BayesianObjective4D, pbounds = inputBounds, random_state = np.random.randint(1, 1e4), verbose = 2)
        optimiser.maximize(init_points = 200, n_iter = 800)

        # Print results.
        elapsedTime = time.time() - startTime
        print("Optimisation complete!")
        print("Elapsed time: " + str(elapsedTime))
        print("Optimal result: " + str(optimiser.max["target"]))
        print("Obtained with values: " + str(optimiser.max["params"]))

        indexList = []
        paramList = []
        for index, result in enumerate(optimiser.res):
                indexList.append(index + 1)
                paramList.append(result["target"])

        plt.plot(indexList, paramList)

        plt.xlabel("Iteration number")
        plt.ylabel(r"FoM (GHz$^{-1}$)")

        plt.show()

        return

def CompareAcqFuncs():
        """
        This function compares the three acquisition functions that are packaged with the Bayesian optimisation library, used to guide the algorithm in its point selection.
        """
        # Create a list containing the names for each acquisition function.
        functionList = {"ucb": "Upper Confidence Bound", "ei": "Expected Improvement", "poi": "Probability of Improvement"}

        # Set up the problem for a 3D optimisation, as that includes angles which seem to be tough to optimise in general.
        inputBounds = {"Bfield": (0, 1e4), "Btheta": (0, 90), "temp": (20, 400)}
        # Note that the same seed is used for each optimisation, such that the starting position is always the same.
        seed = np.random.randint(1, 1e4)

        for function, name in functionList.items():

                print("Using acquisition function: " + name)
                # We need to start a new optimiser from scratch each time.
                optimiser = BayesianOptimization(f = BayesianObjective3D, pbounds = inputBounds, random_state = seed, verbose = 2)
                optimiser.maximize(init_points = 30, n_iter = 70, acq = function)
                indexList = []
                paramList = []
                for index, result in enumerate(optimiser.res):
                        indexList.append(index + 1)
                        paramList.append(result["target"])

                plt.plot(indexList, paramList, label = name)
        
        plt.legend()
        plt.xlabel("Iteration number")
        plt.ylabel(r"FoM (GHz$^{-1}$)")

        plt.show()

        return

def CompareKappaXi():
        """
        Each acquisition function depends on two parameters: kappa and xi. This function explores how the success of the algorithm depends on these values. 
        """

        xiRange = np.linspace(1, 10, num = 2)
        kappaRange = np.linspace(1, 10, num = 2)
        seed = np.random.randint(1, 1e4)

        bestFom = np.empty((xiRange.size, kappaRange.size))

        for i in tqdm(range(xiRange.size)):
                for j in range(kappaRange.size):

                        inputBounds = {"Bfield": (0, 1e4)}
                        optimiser = BayesianOptimization(BayesianObjective1D, inputBounds, random_state = seed, verbose = 0)

                        optimiser.maximize(init_points = 2, n_iter = 3, acq = "poi", kappa = kappaRange[j], xi = xiRange[i])
                        
                        bestFom[i, j] = optimiser.max["target"]

        # Start plotting the figure.
        fig = plt.figure()
        plot = fig.subplots()
        im = plot.imshow(bestFom, extent = (xiRange[0], xiRange[-1], kappaRange[0], kappaRange[-1]), origin = "lower")

        fig.colorbar(im)

        plt.show()

        return


def EasomFunction(x1, x2):
        """
        The Easom function is used to test the Bayesian optimisation algorithm, as it has similar features to the FoM objective function - namely a sharp peak
        surrounded by what seems like a simple linear function. This will show how the algorithm deals with potentially skipping over the peak during exploration.
        Note that this function is only defined on the 2D plane. The maximum occurs at x1 = x2 = pi with a value of 1. The function is usually evaluated from
        -100 to 100 for both dimensions.
        """

        return np.cos(x1) * np.cos(x2) * np.exp(-1 * (x1 - np.pi)**2 - (x2 - np.pi)**2)

def XinSheYang4Function(x1, x2):
        """
        This is another test function used to test the Bayesian optimisation algortithm, featuring many local maxima and one sharp peak.
        This function is defined for any number of dimensions, with a maximum at x = 0 for all x, with a value of 1. This function is usually
        evaluated from -10 to 10 for all dimensions.
        """

        # The input is required to be of this form to be compatible with the optimser.
        x = [x1, x2]
        return -1 * (np.sum(np.square(np.sin(x))) - np.exp(-1 * np.sum(np.square(x)))) * np.exp(-1 * np.sum(np.square(np.sin(np.sqrt(np.abs(x))))))

def StuckmanFunction(x1, x2):
        """
        This function features discontinuities and relies on random numbers so is quite strange. A great test function.
        The only variables are x1, x2. The function will peak at (2.0, 5.0) by construction.
        DEPRECATED: We will assume no discontinuities.
        """

        # We will fix the values of the 'random' variables here to make the results more predictable.
        b = 4.234
        m1 = 75.0
        m2 = 50.0
        xr11 = 2.0
        xr12 = 7.5
        xr21 = 5.0
        xr22 = 3.14


        if x1 <= b:
                a = np.floor(x1 - xr11) + np.floor(x2 - xr21)

                return -1 * np.floor((np.floor(m1) + 0.5) * np.sin(a)/a)

        elif x2 <= 10 and x2 >= b:
                a = np.floor(x1 - xr12) + np.floor(x2 - xr22)

                return -1 * np.floor((np.floor(m2) + 0.5) * np.sin(a)/a)

        else:
                return 0

def Schaffer2Function(x1, x2):
        """
        Another test function, which is very jagged around the maximum. Maximum of 0 at (0, 0), and we search in the square
        -100 < x1, x2 < 100
        """

        return -1 * (0.5 + (np.square(np.sin(np.square(x1) - np.square(x2))) - 0.5)/np.square(1 + 0.001 * np.square(x1) + np.square(x2)))

def ExploreVsExploit(seedNumber):
        """
        This function will show how important the random exploration of the function is before starting the fitting. To do this, 
        the test functions (defined above) as optimised for 10 different seeds, with a different ratio of random evaluations to 
        algorithm iterations. For each test, 100 function evaluations are made. The Probability of Improvement acquisition function
        is used due to its success in ElecSus (see CompareAcqFuncs()).
        """

        seedList = [np.random.randint(1, 1e4) for i in range(seedNumber)]
        exploreList = [1, 2, 5, 10, 20, 25, 50, 75, 90, 99, 100]

        # Define the search space for each function.
        easomBounds = {"x1": (-100, 100), "x2": (-100, 100)}
        xsyBounds = {"x1": (-10, 10), "x2": (-10, 10)}
        schafferBounds = {"x1": (-100, 100), "x2": (-100, 100)}

        exploreScore = [0] * len(exploreList)

        for index, exploreNo in enumerate(tqdm(exploreList)):
                #print("Testing for " + str(exploreNo) + " random search point(s)...")
                for seed in tqdm(seedList):

                        # Optimise the Easom function.
                        optimiserEasom = BayesianOptimization(f = EasomFunction, pbounds = easomBounds, random_state = seed, verbose = 0)
                        optimiserEasom.maximize(init_points = exploreNo, n_iter = (100 - exploreNo), acq = "poi")

                        # Optimise the Xin She Yang 4th function.
                        optimiserXSY = BayesianOptimization(f = XinSheYang4Function, pbounds = xsyBounds, random_state = seed, verbose = 0)
                        optimiserXSY.maximize(init_points = exploreNo, n_iter = (100 - exploreNo), acq = "poi")

                        # Optimise the Schaffer 2nd function.
                        optimiserSchaffer = BayesianOptimization(f = Schaffer2Function, pbounds = schafferBounds, random_state = seed, verbose = 0)
                        optimiserSchaffer.maximize(init_points = exploreNo, n_iter = (100 - exploreNo), acq = "poi")
                        
                        # Check if the optimum was reached for each function.
                        if np.isclose(optimiserEasom.max["target"], 1, rtol = 1.e-3):
                                # The algorithm was successful, add a point to it.
                                exploreScore[index] += 1
                        
                        if np.isclose(optimiserXSY.max["target"], 1, rtol = 1.e-3):
                                # The algorithm was successful, add a point to it.
                                exploreScore[index] += 1

                        if np.isclose(optimiserSchaffer.max["target"], 0, rtol = 1.e-3):
                                # The algorithm was successful, add a point to it.
                                exploreScore[index] += 1

        print("Final score:")
        print(exploreScore)

        return

def AnimateEThetaSensitivity():
        """
        This function will animate the change in transmission and FoM as the incident electric field angle changes.
        This will be used to show how sensitive the system is to this change. We include at the optimum angle as determined by the literature.
        """

        thetaVals = []
        thetaVals.extend(np.geomspace(2, 6, 50))
        thetaVals.extend(np.geomspace(6, 10, 50))
        thetaVals = list(set(thetaVals))
        thetaVals.sort()

        transmissionList = []
        FoMList = []

        for Etheta in tqdm(thetaVals):
                elecsusParams = {'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(Etheta), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}
                transmissionList.append(ProduceSpectrum(globalDetuning, elecsusParams, False))
                FoMList.append([CalculateFoM(globalDetuning, elecsusParams)])

        fig = plt.figure("Sensitivity Animation")
        ax = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        ax.set(xlim = (-1e4, 1e4), ylim = (0, 0.75))
        ax2.set(xlim = (2, 10), ylim = (np.min(FoMList), np.max(FoMList)))
        ax.set_xlabel("Detuning")
        ax.set_ylabel("Transmission")
        ax2.set_xlabel("Etheta")
        ax2.set_ylabel("FoM")

        # Plot static objects.
        ax2.axvline(x = 6, color = "m")
        ax2.axhline(y = FoMList[thetaVals.index(6)], color = "m")
        # The following assumes theta = 6 is always in the array, as it should be.
        ax.plot(globalDetuning, transmissionList[thetaVals.index(6)], color = "m", linestyle = "dashed")

        # Create objects which will be updated.
        transLine = ax.plot(globalDetuning, transmissionList[0])[0]
        infoText = ax.text(0.5e4, 0.5, str(thetaVals[0]))
        FoMLine = ax2.plot(0, 0)[0]

        FoMx, FoMy = [], []

        def animate(i):
                # Define what the graph does on each frame.
                if i == 0:
                        FoMx.clear()
                        FoMy.clear()
                FoMx.append(thetaVals[i])
                FoMy.append(FoMList[i])
                transLine.set_ydata(transmissionList[i])
                infoText.set_text("E theta = " + str(thetaVals[i]) + "\n FoM = " + str(FoMList[i][0]))
                FoMLine.set_data(FoMx, FoMy)
        
        
        animator = FuncAnimation(fig, animate, frames = len(transmissionList))

        plt.draw()
        plt.show()

        return

def AnimateBayesianOpt1D(numIters):
        """
        This method will plot the 'thinking' behind the Bayesian optimisation algorithm, in one dimension.
        We pick the magnetic field as our variable as it is the least computationally cumbersome.
        """

        # Define parameters used for the optimisation algorithm.
        acqKappa = 3
        acqXi = 0.5

        # Start by plotting the objective function being optimised. This could take a while.
        bVals = np.linspace(0, 1e4, num = 50)

        print("Plotting objective function...")
        FoMList = []

        for bField in tqdm(bVals):
                FoMList.append(BayesianObjective1D(bField))

        # Begin the figure plot.
        fig = plt.figure("Optimisation Animation")

        fomAxis = plt.subplot(2, 1, 1)
        acqAxis = plt.subplot(2, 1, 2)

        # Create optimisers.
        inputBounds = {"Bfield": (0, 1e4)}
        optimiser = BayesianOptimization(BayesianObjective1D, inputBounds, random_state = np.random.randint(1, 1e4), verbose = 0)

        # Initialise optimisation with 1 random point.
        optimiser.maximize(init_points = 1, n_iter = 0, acq = "poi", kappa = acqKappa, xi = acqXi)

        # Use points to generate initial prediction graph data, which will obviously be terrible.
        mean, stdDev = optimiser._gp.predict(bVals.reshape(-1, 1), return_std = True)
        # The utility function is used to choose the next point to query. It is an important part of the algorithm.
        utilFunction = UtilityFunction(kind = "poi", kappa = acqKappa, xi = acqXi)
        util = utilFunction.utility(bVals.reshape(-1, 1), optimiser._gp, 0)
        predictB, predictFoM = bVals[np.argmax(util)], np.max(util)
        
        # Create lists to store objects at each iteration.
        # Observed points. To be marked.
        obsList = [(optimiser.space.params.flatten(), optimiser.space.target)]
        # How the optimiser sees the function, and its uncertainty.
        gpList = [mean]
        # How the optimiser will pick its next point, and the point it picked.
        utilList = [(util, predictB, predictFoM)]
        # Store the polygons representing the confidence levels at each iteration.
        confidenceList = [fomAxis.fill_between(bVals, mean + stdDev, mean - stdDev, alpha = 0.3, visible = False)]

        print("Preparing animation...")

        # Iterate the optimiser to see how it evolves.
        for iteration in tqdm(range(numIters)):
                # Run the same code as above.
                optimiser.maximize(init_points = 0, n_iter = 1, acq = "poi", kappa = acqKappa, xi = acqXi)

                mean, stdDev = optimiser._gp.predict(bVals.reshape(-1, 1), return_std = True)
                util = utilFunction.utility(bVals.reshape(-1, 1), optimiser._gp, 0)
                predictB, predictFoM = bVals[np.argmax(util)], np.max(util)

                # Add the obtained values to the various lists.
                obsList.append((optimiser.space.params.flatten(), optimiser.space.target))
                gpList.append(mean)
                utilList.append((util, predictB, predictFoM))
                confidenceList.append(fomAxis.fill_between(bVals, mean + stdDev, mean - stdDev, alpha = 0.3, visible = False))

        """
        Once that for loop is over, we will have all the data required for the animation. The data will not be produced
        on the fly as it would be too computationally expensive to complete between frames.
        """

        fomAxis.set(xlim = (0, 1e4), ylim = (-0.5, 1.1))
        acqAxis.set(xlim = (0, 1e4), ylim = (0, 3))
        fomAxis.set_xlabel("Magnetic field strength (Gauss)")        
        acqAxis.set_xlabel("Magnetic field strength (Gauss)")
        fomAxis.set_ylabel("FoM")
        acqAxis.set_ylabel("Acquisition function value")

        # Plot the functions that won't change with each frame.
        fomAxis.plot(bVals, FoMList)

        # Plot the changing graphs.
        meanLine = fomAxis.plot(bVals, gpList[0])[0]
        acqLine = acqAxis.plot(bVals, utilList[0][0])[0]
        probedScat = fomAxis.scatter(obsList[0][0], obsList[0][1])
        predScat = acqAxis.scatter(utilList[0][1], utilList[0][2])
        confidenceList[0].set_visible(True)

        def animate(i):
                # Define what the graph does on each frame.
                meanLine.set_ydata(gpList[i])
                acqLine.set_ydata(utilList[i][0])
                probedScat.set_offsets(np.c_[obsList[i][0], obsList[i][1]])
                predScat.set_offsets(np.c_[utilList[i][1], utilList[i][2]])
                confidenceList[i - 1].set_visible(False)
                confidenceList[i].set_visible(True)
        
        animator = FuncAnimation(fig, animate, frames = len(obsList), interval = 1e3)

        plt.draw()
        plt.show()

        return


if __name__ == "__main__":
        print("Running test cases...")
        # Test to make sure program is running okay.
        #TestFoM()

        # Confirm results of paper by varying B field strength. Note the OptGraph1D function is *very* long as it evaluates hundreds/thousands of FoM's.
        #TestOpt1D()
        #OptGraph1D()

        # Extend to 2D, now including temperature.
        #TestOpt2D()

        # Extend to 3D, now including the B field angle.
        #TestOpt3D()

        # Extend to 4D, now including the E field angle.
        #TestOpt4D()

        # Animate the sensitivity change of the electric field angle, Etheta.
        #AnimateEThetaSensitivity()

        # Below we look into the algorithm deeper.

        # See how the algorithm thinks.
        #AnimateBayesianOpt1D(30)

        # Compare acquisition functions.
        #CompareAcqFuncs()

        # Compare the effects of kappa and xi.
        CompareKappaXi()

        # See how random exploration affects the algorithms effectiveness.
        #ExploreVsExploit(3)
