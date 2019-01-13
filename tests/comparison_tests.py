"""
This module will aim to answer the question "Is Bayesian Optimisation worth it?".
To do this, we test:
- Average running time for X iterations.
- Average best FoM after X iterations.
- Average FoM 'gain' per unit time.
- Average FoM 'gain' per unit iteration.
- Absolute best FoM after X iterations.
Bayesian Optimisation will be tested against the following algorthms:
- Random Restart Cobyla Optimisation (used in the previous paper).
- Random Search (to show whether any optimisation algorithm is actually useful).
- Covariance Matrix Adaptation - Evolutionary Strategy (commonly called CMA-ES, widely popular for black-box problems).
- Monte Carlo Markov Chain (also commonly used for optimisation problems).
"""

import time
from tqdm import tqdm
from elecsus import elecsus_methods as elecsus
import chocolate as choco
import numpy as np
from scipy.integrate import simps as integrate
import matplotlib.pyplot as plt
import lmfit
from mpi4py import MPI
import pandas as pd

# Here we define some global variables so that it is easier to change it for all functions.
# Detuning used for all tests.
globalDetuning = np.arange(-20000, 20000, 10) # MHz
# Input parameters used for all tests.
globalParams = {'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(0), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}
# Global target FoM. This is used to test for time how long it takes to reproduce a paper value.
globalFoM = 1.04

def ProduceSpectrum(detuning, params, toPlot = False):
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
	        return np.zeros(len(detuning))

        transmittedE =  np.array(J_out * E_out[:2])
        transmission =  (transmittedE * transmittedE.conjugate()).sum(axis = 0)

        if toPlot:
                # Plot the result.
                plt.plot(detuning, transmission)
                plt.show()

        return transmission

def CalculateFoM(detuning, params):

        # Get the overall transmission.
        transmission = ProduceSpectrum(detuning, params, False)
        maxTransmission = np.max(transmission)

        try:
            ENBW = integrate(transmission, detuning)/maxTransmission
        except IndexError:
            print("Index error in determining ENBW! Here is what we tried to integrate:")
            print("Transmission: " + str(transmission) + ". Size: " + str(len(transmission)) + ". Should be " + str(len(detuning)))
            print("Detuning: " + str(detuning))
            print("maxTransmission: " + str(maxTransmission) + ". Should be a single value")
            print("Input parameters for ElecSus: " + str(params))

            ENBW = 1e24
        
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

def Objective1D(bField):
    """
    First variable to vary is the magnetic field strength.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField

    return CalculateFoM(globalDetuning, elecsusParams)

def Objective2D(bField, temperature):
    """
    Second variable to vary is the temperature.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField
    elecsusParams["T"] = temperature

    return CalculateFoM(globalDetuning, elecsusParams)

def Objective3D(bField, temperature, bTheta):
    """
    Third variable to vary is the angle the magnetic field makes with the wavevector.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField
    elecsusParams["T"] = temperature
    elecsusParams["Btheta"] = np.deg2rad(bTheta)

    return CalculateFoM(globalDetuning, elecsusParams)

def Objective4D(bField, temperature, bTheta, eTheta):
    """
    Fourth variable to vary is the angle the electric field makes with the wavevector.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField
    elecsusParams["T"] = temperature
    elecsusParams["Btheta"] = np.deg2rad(bTheta)
    elecsusParams["Etheta"] = np.deg2rad(eTheta)

    return CalculateFoM(globalDetuning, elecsusParams)

def Objective5D(bField, temperature, bTheta, eTheta, bPhi):
    """
    The final variable for a single filter is the angle the magnetic field makes with the electric field.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField
    elecsusParams["T"] = temperature
    elecsusParams["Btheta"] = np.deg2rad(bTheta)
    elecsusParams["Etheta"] = np.deg2rad(eTheta)
    elecsusParams["Bphi"] = np.deg2rad(bPhi)

    return CalculateFoM(globalDetuning, elecsusParams)

def StatsBayes(dimension = 1, numIters = 100, numRuns = 1, paperCompare = False):
    """
    This function will produce all values required for comparison with other algorithms as defined at the top of the module.
    """

    # Define a dictionary to store each possible case for dimensions. Each entry is the objective function and the respective bounds.
    problemList = {
        1: [Objective1D, {"bField": choco.uniform(0, 2000)}],
        2: [Objective2D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400)}],
        3: [Objective3D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 180)}],
        4: [Objective4D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 180), "eTheta": choco.uniform(0, 180)}],
        5: [Objective5D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 180), "eTheta": choco.uniform(0, 180), "bPhi": choco.uniform(0, 180)}]
    }

    problem = problemList.get(dimension)

    # Set up the database for the chocolate optimiser.
    connection = choco.SQLiteConnection("sqlite:///bayes_db" + str(dimension) + ".db")

    if paperCompare:

        timeList = []
        iterationList = []

        for run in range(numRuns):

            # Define which solver will be used.
            solver = choco.Bayes(connection, problem[1], utility_function = "ei", n_bootstrap = int(np.ceil(numIters/10)), clear_db = True)

            # Clear the database. TODO: To do this?
            connection.clear()

            # Start timing.
            startTime = time.time()

            # Start optimisation.
            for iteration in range(numIters):

                # Make one suggestion.
                try:
                    token, nextParams = solver.next()
                except:
                    print("Error suggesting a new point. Here are the last set of parameters sampled, and it's returned value:")
                    print(str(nextParams))
                    print("Iteration number: " + str(iteration))
                    continue

                # Check what FoM this gives. Go negative as this is a minimisation routine.
                fEval =  abs(problem[0].__call__(**nextParams))

                # Update best FoM.
                if fEval >= globalFoM:
                    # The algorithm has managed to surpass or equal the paper value.
                    iterationList.append(iteration + 1)

                    # One run complete.
                    timeElapsed = time.time() - startTime
                    timeList.append(timeElapsed)

                    break
                
                # Tell the optimiser about the result.
                solver.update(token, fEval)

        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(timeList)
        avgIters = np.average(iterationList)
        try:

            fastestTime = np.min(timeList)

        except ValueError:
            
            # List is empty.
            fastestTime = float('NaN')

        numSuccess = len(iterationList)
        successRate = numSuccess/numRuns

        return [numSuccess, successRate, avgIters, avgRuntime, fastestTime]

    else:

        timeList = []
        bestFoMList = []

        for run in range(numRuns):

            # Define which solver will be used.
            solver = choco.Bayes(connection, problem[1], utility_function = "ei", n_bootstrap = int(np.ceil(numIters/10)), clear_db = True)

            # Clear the database. TODO: To do this?
            connection.clear()

            # Start timing.
            startTime = time.time()
            bestFoM = 0

            # Start optimisation.
            for iteration in range(numIters):

                # Make one suggestion.
                try:
                    token, nextParams = solver.next()
                except:
                    print("Error suggesting a new point. Here are the last set of parameters sampled, and it's returned value:")
                    print(str(nextParams))
                    print("Iteration number: " + str(iteration))
                    continue

                # Check what FoM this gives. Go negative as this is a minimisation routine.
                fEval = abs(problem[0].__call__(**nextParams))

                # Update best FoM.
                if fEval > bestFoM:
                    bestFoM = fEval
                
                # Tell the optimiser about the result.
                solver.update(token, fEval)

            # One run complete.
            timeElapsed = time.time() - startTime
            timeList.append(timeElapsed)
            bestFoMList.append(bestFoM)
        
        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(timeList)
        avgFoM = np.average(bestFoMList)
        avgFoMPerTime = np.average(np.divide(bestFoMList, timeList))
        avgFoMPerIter = np.average(np.divide(bestFoMList, numIters))
        absBestFoM = np.max(bestFoMList)

        return [avgRuntime, avgFoM, avgFoMPerTime, avgFoMPerIter, absBestFoM]

def StatsCMAES(dimension = 1, numIters = 100, numRuns = 1, paperCompare = False):
    """
    This function will produce all values required for comparison with other algorithms as defined at the top of the module.
    Thankfully most algorithms are within the same library so we need only change small amounts.
    """

    # Define a dictionary to store each possible case for dimensions. Each entry is the objective function and the respective bounds.
    problemList = {
        1: [Objective1D, {"bField": choco.uniform(0, 2000)}],
        2: [Objective2D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400)}],
        3: [Objective3D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 180)}],
        4: [Objective4D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 180), "eTheta": choco.uniform(0, 180)}],
        5: [Objective5D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 180), "eTheta": choco.uniform(0, 180), "bPhi": choco.uniform(0, 180)}]
    }

    problem = problemList.get(dimension)

    # Set up the database for the chocolate optimiser.
    connection = choco.SQLiteConnection("sqlite:///cmaes_db" + str(dimension) + ".db")

    if paperCompare:

        timeList = []
        iterationList = []

        for run in range(numRuns):

            # Define which solver will be used.
            solver = choco.CMAES(connection, problem[1], clear_db = True)

            # Clear the database. TODO: To do this?
            connection.clear()

            # Start timing.
            startTime = time.time()

            # Start optimisation.
            for iteration in range(numIters):

                # Make one suggestion.
                try:
                    token, nextParams = solver.next()
                except:
                    print("Error suggesting a new point. Here are the last set of parameters sampled, and it's returned value:")
                    print(str(nextParams))
                    print("Iteration number: " + str(iteration))
                    continue

                # Check what FoM this gives. Go negative as this is a minimisation routine.
                fEval = problem[0].__call__(**nextParams)

                # Update best FoM.
                if fEval >= globalFoM:
                    # The algorithm has managed to surpass or equal the paper value.
                    iterationList.append(iteration + 1)

                    # One run complete.
                    timeElapsed = time.time() - startTime
                    timeList.append(timeElapsed)

                    break
                
                # Tell the optimiser about the result.
                solver.update(token, fEval)

        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(timeList)
        avgIters = np.average(iterationList)
        try:

            fastestTime = np.min(timeList)

        except ValueError:
            
            # List is empty.
            fastestTime = float('NaN')

        numSuccess = len(iterationList)
        successRate = numSuccess/numRuns

        return [numSuccess, successRate, avgIters, avgRuntime, fastestTime]

    else:

        timeList = []
        bestFoMList = []

        for run in range(numRuns):

            # Define which solver will be used.
            solver = choco.CMAES(connection, problem[1], clear_db = True)

            # Clear the database. TODO: To do this?
            connection.clear()

            # Start timing.
            startTime = time.time()
            bestFoM = 0

            # Start optimisation.
            for iteration in range(numIters):

                # Make one suggestion.
                try:
                    token, nextParams = solver.next()
                except:
                    print("Error suggesting a new point. Here are the last set of parameters sampled, and it's returned value:")
                    print(str(nextParams))
                    print("Iteration number: " + str(iteration))
                    continue

                # Check what FoM this gives.
                fEval = problem[0].__call__(**nextParams)

                # Update best FoM.
                if fEval > bestFoM:
                    bestFoM = fEval
                
                # Tell the optimiser about the result.
                solver.update(token, fEval)

            # One run complete.
            timeElapsed = time.time() - startTime
            timeList.append(timeElapsed)
            bestFoMList.append(bestFoM)
        
        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(timeList)
        avgFoM = np.average(bestFoMList)
        avgFoMPerTime = np.average(np.divide(bestFoMList, timeList))
        avgFoMPerIter = np.average(np.divide(bestFoMList, numIters))
        absBestFoM = np.max(bestFoMList)

        return [avgRuntime, avgFoM, avgFoMPerTime, avgFoMPerIter, absBestFoM]

def StatsRandom(dimension = 1, numIters = 100, numRuns = 1, paperCompare = False):
    """
    This function will produce all values required for comparison with other algorithms as defined at the top of the module.
    Thankfully most algorithms are within the same library so we need only change small amounts.
    """

    # Define a dictionary to store each possible case for dimensions. Each entry is the objective function and the respective bounds.
    problemList = {
        1: [Objective1D, {"bField": choco.uniform(0, 2000)}],
        2: [Objective2D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400)}],
        3: [Objective3D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 180)}],
        4: [Objective4D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 180), "eTheta": choco.uniform(0, 180)}],
        5: [Objective5D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 180), "eTheta": choco.uniform(0, 180), "bPhi": choco.uniform(0, 180)}]
    }

    problem = problemList.get(dimension)

    # Set up the database for the chocolate optimiser.
    connection = choco.SQLiteConnection("sqlite:///random_db" + str(dimension) + ".db")

    if paperCompare:

        timeList = []
        iterationList = []

        for run in range(numRuns):

            # Define which solver will be used.
            solver = choco.QuasiRandom(connection, problem[1], clear_db = True, skip = int(np.ceil(numIters/10)), seed = np.random.randint(1e4))

            # Clear the database. TODO: To do this?
            connection.clear()

            # Start timing.
            startTime = time.time()

            # Start optimisation.
            for iteration in range(numIters):

                # Make one suggestion.
                try:
                    token, nextParams = solver.next()
                except:
                    print("Error suggesting a new point. Here are the last set of parameters sampled, and it's returned value:")
                    print(str(nextParams))
                    print("Iteration number: " + str(iteration))
                    continue

                # Check what FoM this gives. Go negative as this is a minimisation routine.
                fEval = abs(problem[0].__call__(**nextParams))

                # Update best FoM.
                if fEval >= globalFoM:
                    # The algorithm has managed to surpass or equal the paper value.
                    iterationList.append(iteration + 1)

                    # One run complete.
                    timeElapsed = time.time() - startTime
                    timeList.append(timeElapsed)

                    break
                    
                # Tell the optimiser about the result.
                solver.update(token, fEval)
            
        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(timeList)
        avgIters = np.average(iterationList)
        try:

            fastestTime = np.min(timeList)

        except ValueError:

            # List is empty.
            fastestTime = float('NaN')

        numSuccess = len(iterationList)
        successRate = numSuccess/numRuns

        return [numSuccess, successRate, avgIters, avgRuntime, fastestTime]

    else:

        timeList = []
        bestFoMList = []

        for run in range(numRuns):

            # Define which solver will be used.
            solver = choco.QuasiRandom(connection, problem[1], clear_db = True, skip = int(np.ceil(numIters/10)), seed = np.random.randint(1e4))

            # Clear the database. TODO: To do this?
            connection.clear()

            # Start timing.
            startTime = time.time()
            bestFoM = 0

            # Start optimisation.
            for iteration in range(numIters):

                # Make one suggestion.
                try:
                    token, nextParams = solver.next()
                except:
                    print("Error suggesting a new point. Here are the last set of parameters sampled, and it's returned value:")
                    print(str(nextParams))
                    print("Iteration number: " + str(iteration))
                    continue

                # Check what FoM this gives. Go negative as this is a minimisation routine.
                fEval = abs(problem[0].__call__(**nextParams))

                # Update best FoM.
                if fEval > bestFoM:
                    bestFoM = fEval
                
                # Tell the optimiser about the result.
                solver.update(token, fEval)

            # One run complete.
            timeElapsed = time.time() - startTime
            timeList.append(timeElapsed)
            bestFoMList.append(bestFoM)
        
        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(timeList)
        avgFoM = np.average(bestFoMList)
        avgFoMPerTime = np.average(np.divide(bestFoMList, timeList))
        avgFoMPerIter = np.average(np.divide(bestFoMList, numIters))
        absBestFoM = np.max(bestFoMList)

        return [avgRuntime, avgFoM, avgFoMPerTime, avgFoMPerIter, absBestFoM]

def ObjectiveLmFit(inputDict):
    """
    This is the objective function definition required for use with lmfit. It is easily extensible to multiple dimensions which is nice.
    NOTE: All angles used in this fitting will be in radians as opposed to degrees. Use np.rad2deg() to turn them to degrees, comparable to the other functions.
    NOTE: The return value is negative. 
    """

    elecsusParams = globalParams
    elecsusParams.update(inputDict)

    return -1 * CalculateFoM(globalDetuning, elecsusParams)

def StatsCobyla(dimension = 1, numIters = 100, numRuns = 1, paperCompare = False):
    """
    Cobyla is the (local) minimisation method used in the Optics Letters paper regarding arbitrary magnetic field angles. This requires use of the lmfit module (l as in love).
    """
    compareIterList = []
    compareTimeList = []
    startTime = None

    def iter_cb(params, iteration, resid):
        """
        What the optimiser does at each iteration.
        """

        if paperCompare:

            if abs(resid) >= globalFoM:

                # End the fitting and report the time, iterations etc.
                compareIterList.append(iteration)
                timeElapsedCompare = time.time() - startTime
                compareTimeList.append(timeElapsedCompare)
                
                return True

        else:

            # Nothing to do here.

            return None

    # Define all possible problems, so that the FoM can be evaluated.
    problemList = [Objective1D, Objective2D, Objective3D, Objective4D, Objective5D]

    # Define all parameters for use in lmfit. Each parameter is given a name (exactly the same as in the ElecSus params dict), its minimum value, and its maximum value.
    allParams = [("Bfield", 0.0, 2000), ("T", 20, 400), ("Btheta", 0, np.pi), ("Etheta", 0, np.pi), ("Bphi", 0, np.pi)]

    # Initialise the parameters to be varied.
    inputParams = lmfit.Parameters()

    # Lists containing important values.
    bestFoMList = []
    timeList = []
    
    for run in range(numRuns):
        # Start timing.
        startTime = time.time()

        # Initialise the parameters with random values (hence the name random restart!).
        for i in range(dimension):
            inputParams.add(allParams[i][0], min = allParams[i][1], max = allParams[i][2], vary = True, value = np.random.uniform(allParams[i][1], allParams[i][2]))
            
        # Perform optimisation.
        optimiser = lmfit.minimize(ObjectiveLmFit, inputParams, method = "cobyla", options = {"maxiter": numIters}, iter_cb = iter_cb)

        # One run complete.
        timeElapsed = time.time() - startTime
        timeList.append(timeElapsed)
        bestParams = optimiser.params.valuesdict()

        try:
            # 'Translate' the dictionary to a form understood by the objective functions.
            # NOTE: This may not work if you start shuffling the order of the dimensions being varied.
            bestParams["bField"] = bestParams.pop("Bfield")
            bestParams["temperature"] = bestParams.pop("T")
            bestParams["bTheta"] = bestParams.pop("Btheta")
            bestParams["eTheta"] = bestParams.pop("Etheta")
            bestParams["bPhi"] = bestParams.pop("Bphi")
        except KeyError:
            # Key is not present in dictionary (dimension lower than 5).
            pass

        bestFoM = problemList[dimension - 1].__call__(**bestParams)
        bestFoMList.append(bestFoM)

    if paperCompare:

        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(compareTimeList)
        avgIters = np.average(compareIterList)
        try:

            fastestTime = np.min(compareTimeList)
            
        except ValueError:

            # List is empty.
            fastestTime = float('NaN')

        numSuccess = len(compareIterList)
        successRate = numSuccess/numRuns

        return [numSuccess, successRate, avgIters, avgRuntime, fastestTime]

    else:
    
        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(timeList)
        avgFoM = np.average(bestFoMList)
        avgFoMPerTime = np.average(np.divide(bestFoMList, timeList))
        avgFoMPerIter = np.average(np.divide(bestFoMList, numIters))
        absBestFoM = np.max(bestFoMList)

        return [avgRuntime, avgFoM, avgFoMPerTime, avgFoMPerIter, absBestFoM]

def StatsMCMC(dimension = 1, numIters = 100, numRuns = 1, paperCompare = False):
    """
    Monte-Carlo Markov Chain method, commonly used for optimisation. This requires use of the lmfit module (l as in love).
    """

    startTime = None
    compareIterList = []
    compareTimeList = []

    def iter_cb(params, iteration, resid):
        """
        What the optimiser does at each iteration.
        """

        if paperCompare:

            if abs(resid) >= globalFoM:

                # End the fitting and report the time, iterations etc.
                compareIterList.append(iteration)
                timeElapsedCompare = time.time() - startTime
                compareTimeList.append(timeElapsedCompare)
                
                return True

        else:

            # Nothing to do here.

            return None

    # Define all possible problems, so that the FoM can be evaluated.
    problemList = [Objective1D, Objective2D, Objective3D, Objective4D, Objective5D]

    # Define all parameters for use in lmfit. Each parameter is given a name (exactly the same as in the ElecSus params dict), its minimum value, and its maximum value.
    allParams = [("Bfield", 0.0, 2000), ("T", 20, 400), ("Btheta", 0, np.pi), ("Etheta", 0, np.pi), ("Bphi", 0, np.pi)]

    # Initialise the parameters to be varied.
    inputParams = lmfit.Parameters()
    # Lists containing important values.
    timeList = []
    bestFoMList = []

    # Change the MCMC input parameters to ensure the number of iterations is comparable.
    # Should be much greater than the dimension of the problem. Defaults to 10 walkers if the other value is too small. The number of walkers must be even.
    numWalkers = 2 * np.ceil(dimension * np.ceil(numIters/10))
    # Number of steps for each walker to take. Note that one step requires numWalkers * dimension function evaluations. 
    numSteps = np.ceil(numIters/(numWalkers * dimension))
    
    for run in range(numRuns):
        # Start timing.
        startTime = time.time()

        # Initialise the parameters with random values.
        for i in range(dimension):
            inputParams.add(allParams[i][0], min = allParams[i][1], max = allParams[i][2], vary = True, value = np.random.uniform(allParams[i][1], allParams[i][2]))
            
        # Perform optimisation.
        #optimiser = lmfit.minimize(ObjectiveLmFit, inputParams, method = "emcee", steps = int(numSteps), nwalkers = int(numWalkers), iter_cb = iter_cb)
        minimiser = lmfit.Minimizer(ObjectiveLmFit, inputParams, iter_cb = iter_cb)
        optimiser = minimiser.emcee(steps = int(numSteps), nwalkers = int(numWalkers))

        # One run complete.
        timeElapsed = time.time() - startTime
        timeList.append(timeElapsed)
        bestParams = optimiser.params.valuesdict()
        try:
            # 'Translate' the dictionary to a form understood by the objective functions.
            # NOTE: This may not work if you start shuffling the order of the dimensions being varied.
            bestParams["bField"] = bestParams.pop("Bfield")
            bestParams["temperature"] = bestParams.pop("T")
            bestParams["bTheta"] = bestParams.pop("Btheta")
            bestParams["eTheta"] = bestParams.pop("Etheta")
            bestParams["bPhi"] = bestParams.pop("Bphi")
        except KeyError:
            # Key is not present in dictionary (dimension lower than 5).
            pass

        bestFoM = problemList[dimension - 1].__call__(**bestParams)
        bestFoMList.append(bestFoM)

    if paperCompare:

        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(compareTimeList)
        avgIters = np.average(compareIterList)
        try:
            fastestTime = np.min(compareTimeList)
        except ValueError:
            # List is empty.
            fastestTime = float('NaN')

        numSuccess = len(compareIterList)
        successRate = numSuccess/numRuns

        return [numSuccess, successRate, avgIters, avgRuntime, fastestTime]

    else:
    
        # Compute required values. This could probably be sped up by using created variables but I chose not to do so.
        avgRuntime = np.average(timeList)
        avgFoM = np.average(bestFoMList)
        avgFoMPerTime = np.average(np.divide(bestFoMList, timeList))
        avgFoMPerIter = np.average(np.divide(bestFoMList, numIters))
        absBestFoM = np.max(bestFoMList)

        return [avgRuntime, avgFoM, avgFoMPerTime, avgFoMPerIter, absBestFoM]

def CompareAlgos(numIters, numRuns):
    """
    The main function of this module. Runs all the above code to print a comparison table to show the strengths and weaknesses of each module.
    """

    maxDim = 5

    for dimension in [x + 1 for x in range(maxDim)]:

        print("----------------------------------------------------------------------------------")
        print("DIMENSION: " + str(dimension))
        # Determine the important values for each algorithm.
        print("Bayesian Optimisation:")
        print(StatsBayes(dimension, numIters, numRuns))
        print("CMAES:")
        print(StatsCMAES(dimension, numIters, numRuns))
        print("Random:")
        print(StatsRandom(dimension, numIters, numRuns))
        print("Cobyla:")
        print(StatsCobyla(dimension, numIters, numRuns))
        print("MCMC:")
        print(StatsMCMC(dimension, numIters, numRuns))
        print("----------------------------------------------------------------------------------")

    return

def ComparePaperAlgos(numIters, numRuns):
    """
    Compares the algorithms against the literature value of the specific experiment defined in the global parameters.
    This method will return values is a neat-ish table, but it will be very slow. Refer to ComparePaperMPI for a parallelised version 
    that's a little sore on the eyes but faster.
    """

    maxDim = 5

    for dimension in [x + 1 for x in range(maxDim)]:

        print("----------------------------------------------------------------------------------")
        print("DIMENSION: " + str(dimension))
        # Determine the important values for each algorithm.
        print("Bayesian Optimisation:")
        print(StatsBayes(dimension, numIters, numRuns, True))
        print("CMAES:")
        print(StatsCMAES(dimension, numIters, numRuns, True))
        print("Random:")
        print(StatsRandom(dimension, numIters, numRuns, True))
        print("Cobyla:")
        print(StatsCobyla(dimension, numIters, numRuns, True))
        print("MCMC:")
        print(StatsMCMC(dimension, numIters, numRuns, True))
        print("----------------------------------------------------------------------------------")

    return

def CompareMPI(numIters, numRuns):
    """
    Compares the algorithms against the literature value of the specific experiment defined in the global parameters.
    Uses multiple cores thanks to the MPI framework.
    """

    # Find all the cores being used.
    comm = MPI.COMM_WORLD
    
    # Get the 'name' of each core to refer to them.
    rank = comm.Get_rank()

    # Create a list of all problems. This is explicitly defined.
    problemList = [("Bayesian Optimisation, 1D", 1, StatsBayes), ("Bayesian Optimisation, 2D", 2, StatsBayes), ("Bayesian Optimisation, 3D", 3, StatsBayes), ("Bayesian Optimisation, 4D", 4, StatsBayes),
    ("Bayesian Optimisation, 5D", 5, StatsBayes), ("CMAES, 1D", 1, StatsCMAES), ("CMAES, 2D", 2, StatsCMAES), ("CMAES, 3D", 3, StatsCMAES), ("CMAES, 4D", 4, StatsCMAES),
    ("CMAES, 5D", 5, StatsCMAES), ("Random Searching, 1D", 1, StatsRandom), ("Random Searching, 2D", 2, StatsRandom), ("Random Searching, 3D", 3, StatsRandom), ("Random Searching, 4D", 4, StatsRandom),
    ("Random Searching, 5D", 5, StatsRandom), ("Cobyla, 1D", 1, StatsCobyla), ("Cobyla, 2D", 2, StatsCobyla), ("Cobyla, 3D", 3, StatsCobyla), ("Cobyla, 4D", 4, StatsCobyla),
    ("Cobyla, 5D", 5, StatsCobyla)]#, ("MCMC, 1D", 1, StatsMCMC), ("MCMC, 2D", 2, StatsMCMC), ("MCMC, 3D", 3, StatsMCMC), ("MCMC, 4D", 4, StatsMCMC), ("MCMC, 5D", 5, StatsMCMC)]

    # Ensure there is a core per calculation.
    assert (len(problemList)%comm.Get_size() == 0), "You need to run this on a number of cores such that " + str(len(problemList)) + " problems can be separated evenly amongst each one."

    # Pick the problems the thread will solve.
    problemIndices = np.arange(rank, len(problemList), step = comm.Get_size())

    for index in problemIndices:
        problem = problemList[index]
        results = problem[2].__call__(problem[1], numIters, numRuns)

        #Append results to a file.
        with open("compare_timings.txt", "a") as textFile:
            textFile.write(str(problem[0]))
            textFile.write("\n")
            textFile.write(str(results))
            textFile.write("\n")
            textFile.close()

        print(problem[0])
        print(str(results))

    comm.Barrier()

    if rank == 0:
        print("All processes ended.")

    return

def ComparePaperMPI(numIters, numRuns):
    """
    Compares the algorithms against the literature value of the specific experiment defined in the global parameters.
    Uses multiple cores thanks to the MPI framework.
    """

    # Find all the cores being used.
    comm = MPI.COMM_WORLD
    
    # Get the 'name' of each core to refer to them.
    rank = comm.Get_rank()

    # Create a list of all problems. This is explicitly defined.
    problemList = [("Bayesian Optimisation, 1D", 1, StatsBayes), ("Bayesian Optimisation, 2D", 2, StatsBayes), ("Bayesian Optimisation, 3D", 3, StatsBayes), ("Bayesian Optimisation, 4D", 4, StatsBayes),
    ("Bayesian Optimisation, 5D", 5, StatsBayes), ("CMAES, 1D", 1, StatsCMAES), ("CMAES, 2D", 2, StatsCMAES), ("CMAES, 3D", 3, StatsCMAES), ("CMAES, 4D", 4, StatsCMAES),
    ("CMAES, 5D", 5, StatsCMAES), ("Random Searching, 1D", 1, StatsRandom), ("Random Searching, 2D", 2, StatsRandom), ("Random Searching, 3D", 3, StatsRandom), ("Random Searching, 4D", 4, StatsRandom),
    ("Random Searching, 5D", 5, StatsRandom), ("Cobyla, 1D", 1, StatsCobyla), ("Cobyla, 2D", 2, StatsCobyla), ("Cobyla, 3D", 3, StatsCobyla), ("Cobyla, 4D", 4, StatsCobyla),
    ("Cobyla, 5D", 5, StatsCobyla)]#, ("MCMC, 1D", 1, StatsMCMC), ("MCMC, 2D", 2, StatsMCMC), ("MCMC, 3D", 3, StatsMCMC), ("MCMC, 4D", 4, StatsMCMC), ("MCMC, 5D", 5, StatsMCMC)]

    # Ensure there is a core per calculation.
    assert (len(problemList)%comm.Get_size() == 0), "You need to run this on a number of cores such that " + str(len(problemList)) + " problems can be separated evenly amongst each one."

    # Pick the problems the thread will solve.
    problemIndices = np.arange(rank, len(problemList), step = comm.Get_size())

    for index in problemIndices:
        problem = problemList[index]
        results = problem[2].__call__(problem[1], numIters, numRuns, True)

        # Append results to a file.
        with open("compare_paper.txt", "a") as textFile:
            textFile.write(str(problem[0]))
            textFile.write("\n")
            textFile.write(str(results))
            textFile.write("\n")
            textFile.close()

        print(problem[0])
        print(str(results))

    comm.Barrier()

    if rank == 0:
        print("All processes ended.")

    return

if __name__ == "__main__":
    # Ensure calculations are being carried out correctly.
    #TestFoM()

    # Determine the stats for an optimisation algorithm.
    #print(StatsCobyla(2, 2000))

    # Test spectrum plotting.
    # Input parameters used.
    #inputParams = {'Bfield':144, 'rb85frac':72.17, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(0), 'lcell':5e-3, 'T':245, 'Dline':'D2', 'Elem':'Na'}
    #ProduceSpectrum(globalDetuning, inputParams, toPlot = True)

    # Compare algorithms using different tests.
    #CompareAlgos(200, 10)
    #ComparePaperAlgos(200, 10)
    ComparePaperMPI(300, 7)
    #CompareMPI(200, 1)