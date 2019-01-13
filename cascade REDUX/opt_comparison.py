"""
Compares the strategies of the skopt module to the Cobyla algorithm through two tests:
- Reaching the literature value within a certain number of iterations (or less).
- Seeing the highest value obtained after a fixed number of iterations
"""

from elecsus import elecsus_methods as elecsus
import numpy as np
import skopt
import lmfit
from scipy.integrate import simps as integrate
from matplotlib import pyplot as plt
from skopt.plots import plot_convergence
import time
from mpi4py import MPI

# Global parameters.
baseParams = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
globalDetuning = np.linspace(-25000, 25000, 1000)

def FitnessSkopt(inputParams):
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
    
    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print(str(elecsusParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def FitnessSkopt5D(inputParams):
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

def FitnessLMFit(inputParams):
    """
    The figure of merit generation function as required by lmfit. The input is a list of each parameter. This is for a single filter.
    """

    elecsusParams = baseParams.copy()

    # This is the full dictionary to use on ElecSus.
    elecsusParams.update(inputParams)

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

def ComparePaper(maxIters, numRuns):
    """
    This test tasks each algorithm with obtaining the rubidium single cell best FoM, which is 1.04.
    """

    toMPI = False
    # Allow for MPI usage.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if comm.Get_size() == 6:
        toMPI = True

    assert (comm.Get_size() == 6 or comm.Get_size() == 1), "There are not enough/too many processes running for this problem. Please ensure the number of processes is either 1 or 6."

    # The target for each algorithm. This was determined by using the values in the literature, so there is clearly come deviation either due to the detuning or computation.
    globalFoM = 1.033

    if toMPI:

        if rank == 0:

            # The first process works with the Cobyla test.
            cobylaIterList = []
            cobylaTimeList = []
            cobylaStartTime = None

            # Define the checks made at every iteration.
            def iter_cb(params, iteration, resid):

                if abs(resid) >= globalFoM:

                    # End the fitting and report the time, iterations etc.
                    cobylaIterList.append(iteration)
                    timeElapsedCompare = time.time() - cobylaStartTime
                    cobylaTimeList.append(timeElapsedCompare)
                        
                    return True

                else:

                    # Nothing to do here.

                    return None

            # Define all parameters for use in lmfit. Each parameter is given a name (exactly the same as in the ElecSus params dict), its minimum value, and its maximum value.
            cobylaParams = [("Bfield", 10, 1300), ("T", 50, 230), ("Btheta", 0, np.pi/2.0), ("Etheta", 0, np.pi/2.0)]

            # Initialise the parameters to be varied.
            inputParams = lmfit.Parameters()

            # NOTE: Multiplied by 10 for comparable times to other methods.
            for run in range(numRuns * 10):
                # Start timing.
                cobylaStartTime = time.time()

                # Initialise the parameters with random values (hence the name random restart!).
                for i in range(len(cobylaParams)):
                    inputParams.add(cobylaParams[i][0], min = cobylaParams[i][1], max = cobylaParams[i][2], vary = True, value = np.random.uniform(cobylaParams[i][1], cobylaParams[i][2]))
                    
                # Perform optimisation.
                optimiser = lmfit.minimize(FitnessLMFit, inputParams, method = "cobyla", options = {"maxiter": maxIters}, iter_cb = iter_cb)

            # Compute required values
            avgRuntime = np.average(cobylaTimeList)
            avgIters = np.average(cobylaIterList)
            try:
                fastestTime = np.min(cobylaTimeList) 
            except ValueError:
                # List is empty.
                fastestTime = float('NaN')

            numSuccess = len(cobylaIterList)
            successRate = numSuccess/numRuns

            print("Cobyla algorithm testing complete! Here are the stats:")
            print("Number of successful runs: " + str(numSuccess) + " (Success rate of " + str(successRate) + ")")
            print("Average iterations required for success: " + str(avgIters))
            print("Average time required for success: " + str(avgRuntime))
            print("Fastest convergence time: " + str(fastestTime))
            print("------------------------------------------------------------------------------------------------------------------")

        else:

            # Define the problem bounds.
            skoptBounds = [(10, 1300), (40, 230), (0, 90), (0, 90)]

            # Give each algorithm the same seeds so they start from the same place.
            seedList = [np.random.randint(1, 1e6) for i in range(numRuns)]

            strategyList = [("GP", "Bayesian optimisation"), ("RF", "Random forest"), ("ET", "Extra trees"), ("GBRT", "Gradient boosted random trees"), ("DUMMY", "Random sampling")] 

            iterList = []
            timeList = []

            for run in range(numRuns):

                startTime = time.time()
                optimiser = skopt.Optimizer(skoptBounds, base_estimator = strategyList[rank - 1][0], n_initial_points = int(np.ceil(maxIters/10)), random_state = seedList[run])

                for iteration in range(maxIters):

                    # Find out which point to sample next.
                    nextParams = optimiser.ask()

                    # Evaluate the objective function.
                    nextFoM = FitnessSkopt(nextParams)

                    # Check if it has reached the literature.
                    if abs(nextFoM) >= globalFoM:
                        # End the fitting and report the time, iterations etc.
                        iterList.append(iteration + 1)
                        timeElapsedCompare = time.time() - startTime
                        timeList.append(timeElapsedCompare)
                        break
                    else:
                        # Update the model.
                        optimiser.tell(nextParams, nextFoM)
                            
            # Compute required values.
            avgRuntime = np.average(timeList)
            avgIters = np.average(iterList)
            try:
                fastestTime = np.min(timeList) 
            except ValueError:
                # List is empty.
                fastestTime = float('NaN')

            numSuccess = len(iterList)
            successRate = numSuccess/numRuns

            print(strategyList[rank - 1][1] + " testing complete! Here are the stats:")
            print("Number of successful runs: " + str(numSuccess) + " (Success rate of " + str(successRate) + ")")
            print("Average iterations required for success: " + str(avgIters))
            print("Average time required for success: " + str(avgRuntime))
            print("Fastest convergence time: " + str(fastestTime))
            print("------------------------------------------------------------------------------------------------------------------")

    else:

        # Start with the Cobyla test.
        cobylaIterList = []
        cobylaTimeList = []
        cobylaStartTime = None

        # Define the checks made at every iteration.
        def iter_cb(params, iteration, resid):

            if abs(resid) >= globalFoM:

                # End the fitting and report the time, iterations etc.
                cobylaIterList.append(iteration)
                timeElapsedCompare = time.time() - cobylaStartTime
                cobylaTimeList.append(timeElapsedCompare)
                    
                return True

            else:

                # Nothing to do here.

                return None

        # Define all parameters for use in lmfit. Each parameter is given a name (exactly the same as in the ElecSus params dict), its minimum value, and its maximum value.
        cobylaParams = [("Bfield", 10, 1300), ("T", 50, 230), ("Btheta", 0, np.pi/2.0), ("Etheta", 0, np.pi/2.0)]

        # Initialise the parameters to be varied.
        inputParams = lmfit.Parameters()

        # NOTE: Multiplied for comparable times.
        for run in range(numRuns * 10):
            # Start timing.
            cobylaStartTime = time.time()

            # Initialise the parameters with random values (hence the name random restart!).
            for i in range(len(cobylaParams)):
                inputParams.add(cobylaParams[i][0], min = cobylaParams[i][1], max = cobylaParams[i][2], vary = True, value = np.random.uniform(cobylaParams[i][1], cobylaParams[i][2]))
                
            # Perform optimisation.
            optimiser = lmfit.minimize(FitnessLMFit, inputParams, method = "cobyla", options = {"maxiter": maxIters}, iter_cb = iter_cb)

        # Compute required values
        avgRuntime = np.average(cobylaTimeList)
        avgIters = np.average(cobylaIterList)
        try:
            fastestTime = np.min(cobylaTimeList) 
        except ValueError:
            # List is empty.
            fastestTime = float('NaN')

        numSuccess = len(cobylaIterList)
        successRate = numSuccess/numRuns

        print("Cobyla algorithm testing complete! Here are the stats:")
        print("Number of successful runs: " + str(numSuccess) + " (Success rate of " + str(successRate) + ")")
        print("Average iterations required for success: " + str(avgIters))
        print("Average time required for success: " + str(avgRuntime))
        print("Fastest convergence time: " + str(fastestTime))
        print("------------------------------------------------------------------------------------------------------------------")

        # Define the problem bounds.
        skoptBounds = [(10, 1300), (40, 230), (0, 90), (0, 90)]

        # Give each algorithm the same seeds so they start from the same place.
        seedList = [np.random.randint(1, 1e6) for i in range(numRuns)]

        strategyList = [("GP", "Bayesian optimisation"), ("RF", "Random forest"), ("ET", "Extra trees"), ("GBRT", "Gradient boosted random trees"), ("DUMMY", "Random sampling")] 

        for strategy in strategyList:

            iterList = []
            timeList = []

            for run in range(numRuns):

                startTime = time.time()
                optimiser = skopt.Optimizer(skoptBounds, base_estimator = strategy[0], n_initial_points = int(np.ceil(maxIters/10)), random_state = seedList[run])

                for iteration in range(maxIters):

                    # Find out which point to sample next.
                    nextParams = optimiser.ask()

                    # Evaluate the objective function.
                    nextFoM = FitnessSkopt(nextParams)

                    # Check if it has reached the literature.
                    if abs(nextFoM) >= globalFoM:
                        # End the fitting and report the time, iterations etc.
                        iterList.append(iteration + 1)
                        timeElapsedCompare = time.time() - startTime
                        timeList.append(timeElapsedCompare)
                        break
                    else:
                        # Update the model.
                        optimiser.tell(nextParams, nextFoM)
                        
            # Compute required values.
            avgRuntime = np.average(timeList)
            avgIters = np.average(iterList)
            try:
                fastestTime = np.min(timeList) 
            except ValueError:
                # List is empty.
                fastestTime = float('NaN')

            numSuccess = len(iterList)
            successRate = numSuccess/numRuns

            print(strategy[1] + " testing complete! Here are the stats:")
            print("Number of successful runs: " + str(numSuccess) + " (Success rate of " + str(successRate) + ")")
            print("Average iterations required for success: " + str(avgIters))
            print("Average time required for success: " + str(avgRuntime))
            print("Fastest convergence time: " + str(fastestTime))
            print("------------------------------------------------------------------------------------------------------------------")

    return

def Compare5D(maxIters, numRuns):
    """
    This test tasks each algorithm with maximising the rubidium single cell 5D problem.
    """

    toMPI = False
    # Allow for MPI usage.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if comm.Get_size() == 6:
        toMPI = True

    assert (comm.Get_size() == 6 or comm.Get_size() == 1), "There are not enough/too many processes running for this problem. Please ensure the number of processes is either 1 or 6."

    if toMPI:

        if rank == 0:

            # The first process works with the Cobyla test.
            cobylaTimeList = []
            bestFoMList = []
            cobylaStartTime = None

            # Define all parameters for use in lmfit. Each parameter is given a name (exactly the same as in the ElecSus params dict), its minimum value, and its maximum value.
            cobylaParams = [("Bfield", 10, 1300), ("T", 50, 230), ("Btheta", 0, np.pi/2.0), ("Etheta", 0, np.pi/2.0), ("Bphi", 0, np.pi/2.0)]

            # Initialise the parameters to be varied.
            inputParams = lmfit.Parameters()

            # NOTE: Multiplied by 10 for comparable times.
            for run in range(numRuns * 10):
                # Start timing.
                cobylaStartTime = time.time()

                # Initialise the parameters with random values (hence the name random restart!).
                for i in range(len(cobylaParams)):
                    inputParams.add(cobylaParams[i][0], min = cobylaParams[i][1], max = cobylaParams[i][2], vary = True, value = np.random.uniform(cobylaParams[i][1], cobylaParams[i][2]))
                    
                # Perform optimisation.
                optimiser = lmfit.minimize(FitnessLMFit, inputParams, method = "cobyla", options = {"maxiter": maxIters})

                # Optimisation complete, get the best result.
                bestParams = optimiser.params.valuesdict()
                bestFoM = FitnessLMFit(bestParams)
                bestFoMList.append(abs(bestFoM))

                cobylaTimeList.append(time.time() - cobylaStartTime)

            # Compute required values
            avgRuntime = np.average(cobylaTimeList)
            avgFoM = np.average(bestFoMList)
            avgFoMPerTime = np.average(np.divide(bestFoMList, cobylaTimeList))
            avgFoMPerIter = np.average(np.divide(bestFoMList, maxIters))
            absBestFoM = np.max(bestFoMList)

            print("Cobyla algorithm testing complete! Here are the stats:")
            print("Average runtime per run (s): " + str(avgRuntime))
            print("Average FoM: " + str(avgFoM))
            print("Average FoM per unit time: " + str(avgFoMPerTime))
            print("Average FoM per unit iteration: " + str(avgFoMPerIter))
            print("Absolute best FoM determined: " + str(absBestFoM))
            print("------------------------------------------------------------------------------------------------------------------")

        else:

            # Define the problem bounds.
            skoptBounds = [(10, 1300), (40, 230), (0, 90), (0, 90), (0, 90)]

            # Give each algorithm the same seeds so they start from the same place.
            seedList = [np.random.randint(1, 1e6) for i in range(numRuns)]

            strategyList = [("GP", "Bayesian optimisation"), ("RF", "Random forest"), ("ET", "Extra trees"), ("GBRT", "Gradient boosted random trees"), ("DUMMY", "Random sampling")] 

            bestFoMList = []
            timeList = []

            for run in range(numRuns):

                startTime = time.time()
                optimiser = skopt.Optimizer(skoptBounds, base_estimator = strategyList[rank - 1][0], n_initial_points = int(np.ceil(maxIters/10)), random_state = seedList[run])

                bestFoM = 0

                for iteration in range(maxIters):

                    # Find out which point to sample next.
                    nextParams = optimiser.ask()

                    # Evaluate the objective function.
                    nextFoM = FitnessSkopt5D(nextParams)

                    if abs(nextFoM) > bestFoM:
                        bestFoM = abs(nextFoM)
                    
                    # Update the model.
                    optimiser.tell(nextParams, nextFoM)

                bestFoMList.append(bestFoM)
                timeList.append(time.time() - startTime)
                            
            # Compute required values
            avgRuntime = np.average(timeList)
            avgFoM = np.average(bestFoMList)
            avgFoMPerTime = np.average(np.divide(bestFoMList, timeList))
            avgFoMPerIter = np.average(np.divide(bestFoMList, maxIters))
            absBestFoM = np.max(bestFoMList)

            print(strategyList[rank - 1][1] + " testing complete! Here are the stats:")
            print("Average runtime per run (s): " + str(avgRuntime))
            print("Average FoM: " + str(avgFoM))
            print("Average FoM per unit time: " + str(avgFoMPerTime))
            print("Average FoM per unit iteration: " + str(avgFoMPerIter))
            print("Absolute best FoM determined: " + str(absBestFoM))
            print("------------------------------------------------------------------------------------------------------------------")

    else:

        # The first process works with the Cobyla test.
        cobylaTimeList = []
        bestFoMList = []
        cobylaStartTime = None

        # Define all parameters for use in lmfit. Each parameter is given a name (exactly the same as in the ElecSus params dict), its minimum value, and its maximum value.
        cobylaParams = [("Bfield", 10, 1300), ("T", 50, 230), ("Btheta", 0, np.pi/2.0), ("Etheta", 0, np.pi/2.0), ("Bphi", 0, np.pi/2.0)]

        # Initialise the parameters to be varied.
        inputParams = lmfit.Parameters()

        # NOTE: Multiplied by 10 for comparable times.
        for run in range(numRuns * 10):
            # Start timing.
            cobylaStartTime = time.time()

            # Initialise the parameters with random values (hence the name random restart!).
            for i in range(len(cobylaParams)):
                inputParams.add(cobylaParams[i][0], min = cobylaParams[i][1], max = cobylaParams[i][2], vary = True, value = np.random.uniform(cobylaParams[i][1], cobylaParams[i][2]))
                    
            # Perform optimisation.
            optimiser = lmfit.minimize(FitnessLMFit, inputParams, method = "cobyla", options = {"maxiter": maxIters})

            # Optimisation complete, get the best result.
            bestParams = optimiser.params.valuesdict()
            bestFoM = FitnessLMFit(bestParams)
            bestFoMList.append(abs(bestFoM))

            cobylaTimeList.append(time.time() - cobylaStartTime)

        # Compute required values
        avgRuntime = np.average(cobylaTimeList)
        avgFoM = np.average(bestFoMList)
        avgFoMPerTime = np.average(np.divide(bestFoMList, cobylaTimeList))
        avgFoMPerIter = np.average(np.divide(bestFoMList, maxIters))
        absBestFoM = np.max(bestFoMList)

        print("Cobyla algorithm testing complete! Here are the stats:")
        print("Average runtime per run (s): " + str(avgRuntime))
        print("Average FoM: " + str(avgFoM))
        print("Average FoM per unit time: " + str(avgFoMPerTime))
        print("Average FoM per unit iteration: " + str(avgFoMPerIter))
        print("Absolute best FoM determined: " + str(absBestFoM))
        print("------------------------------------------------------------------------------------------------------------------")

        # Define the problem bounds.
        skoptBounds = [(10, 1300), (40, 230), (0, 90), (0, 90,), (0, 90)]

        # Give each algorithm the same seeds so they start from the same place.
        seedList = [np.random.randint(1, 1e6) for i in range(numRuns)]

        strategyList = [("GP", "Bayesian optimisation"), ("RF", "Random forest"), ("ET", "Extra trees"), ("GBRT", "Gradient boosted random trees"), ("DUMMY", "Random sampling")] 

        for strategy in strategyList:

            bestFoMList = []
            timeList = []

            for run in range(numRuns):

                startTime = time.time()
                optimiser = skopt.Optimizer(skoptBounds, base_estimator = strategy[0], n_initial_points = int(np.ceil(maxIters/10)), random_state = seedList[run])
                bestFoM = 0

                for iteration in range(maxIters):

                    # Find out which point to sample next.
                    nextParams = optimiser.ask()

                    # Evaluate the objective function.
                    nextFoM = FitnessSkopt5D(nextParams)

                    if abs(nextFoM) > bestFoM:
                        bestFoM = abs(nextFoM)
                        
                    # Update the model.
                    optimiser.tell(nextParams, nextFoM)

            bestFoMList.append(bestFoM)
            timeList.append(time.time() - startTime)
                            
            # Compute required values
            avgRuntime = np.average(timeList)
            avgFoM = np.average(bestFoMList)
            avgFoMPerTime = np.average(np.divide(bestFoMList, timeList))
            avgFoMPerIter = np.average(np.divide(bestFoMList, maxIters))
            absBestFoM = np.max(bestFoMList)

            print(strategy[1] + " testing complete! Here are the stats:")
            print("Average runtime per run (s): " + str(avgRuntime))
            print("Average FoM: " + str(avgFoM))
            print("Average FoM per unit time: " + str(avgFoMPerTime))
            print("Average FoM per unit iteration: " + str(avgFoMPerIter))
            print("Absolute best FoM determined: " + str(absBestFoM))
            print("------------------------------------------------------------------------------------------------------------------")

    return

if __name__ == "__main__":
    # Run the first comparison test.
    ComparePaper(1000, 5)

    # Run the second comparison test.
    Compare5D(1000, 5)
