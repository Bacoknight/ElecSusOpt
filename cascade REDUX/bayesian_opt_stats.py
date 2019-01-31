"""
Due to the length of time required to test Bayesian Optimisation in concurrence with other skopt strategies, we use the Chocolate library instead.
"""

import numpy as np
import chocolate as choco
from elecsus import elecsus_methods as elecsus
import time
from scipy.integrate import simps as integrate
from mpi4py import MPI
import skopt

# Global parameters.
baseParams = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
globalDetuning = np.linspace(-25000, 25000, 1000)

def FitnessPaper(Bfield, Etheta, Btheta, T):
    """
    The fitness function for comparing to the paper i.e 4D problem.
    """

    elecsusParams = baseParams.copy()

    # NOTE: No bPhi as of yet.
    paramDict = {'Bfield': Bfield, "T": T, 'Btheta': np.deg2rad(Btheta), 'Etheta': np.deg2rad(Etheta)}

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

def Fitness5D(Bfield, Etheta, Btheta, T, Bphi):
    """
    The fitness function for the full single filter.
    """

    elecsusParams = baseParams.copy()

    # NOTE: No bPhi as of yet.
    paramDict = {'Bfield': Bfield, "T": T, 'Btheta': np.deg2rad(Btheta), 'Etheta': np.deg2rad(Etheta), 'Bphi': np.deg2rad(Bphi)}

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

def BayesPaperStats(maxIters, numRuns):
    """
    Obtain the paper comparison stats.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    assert comm.Get_size() == numRuns, "Please ensure there is one process running per run i.e " + str(numRuns) + " processes."
    
    problemBounds = {"Bfield": choco.uniform(10, 1300), "T": choco.uniform(50, 230), "Btheta": choco.uniform(0, 90), "Etheta": choco.uniform(0, 90)}

    # The target for each algorithm. This was determined by using the values in the literature, so there is clearly some deviation either due to the detuning or computation.
    globalFoM = 1.033

    if rank == 0:
        timeList = []
        iterationList = []

    # Set up the database for the chocolate optimiser.
    connection = choco.SQLiteConnection("sqlite:///bayes_paper_" + str(rank) + "_db.db")

    # Define which solver will be used.
    solver = choco.Bayes(connection, problemBounds, utility_function = "ei", n_bootstrap = int(np.ceil(maxIters/10)), clear_db = True)

    # Clear the database. TODO: To do this?
    connection.clear()

    # Start timing.
    startTime = time.time()
    timeElapsed = None
    iterationSuccess = None

    # Start optimisation.
    for iteration in range(maxIters):

        # Make one suggestion.
        try:
            token, nextParams = solver.next()
        except:
            print("Error suggesting a new point. Here are the last set of parameters sampled:")
            print(str(nextParams))
            print("Iteration number: " + str(iteration))
            continue

        # Check what FoM this gives. Go negative as this is a minimisation routine.
        fEval =  abs(FitnessPaper(**nextParams))

        # Update best FoM.
        if fEval >= globalFoM:
            # The algorithm has managed to surpass or equal the paper value.
            iterationSuccess = iteration
            timeElapsed = time.time() - startTime
            
            if rank == 0:
                iterationList.append(iterationSuccess)
                timeList.append(timeElapsed)

            break
        
        # Tell the optimiser about the result.
        solver.update(token, fEval)

    # Run complete. Send results to main process. Tags are unique identifiers.
    if rank != 0:
        comm.send(timeElapsed, dest = 0, tag = 1)
        comm.send(iterationSuccess, dest = 0, tag = 2)

    # Wait for all the processes to end.
    comm.Barrier()

    if rank == 0:
        # Aggregate the data.
        for process in range(comm.Get_size() - 1):
            # Get the data.
            individualTime = None
            individualTime = comm.recv(individualTime, source = process + 1, tag = 1)

            individualIter = None
            individualIter = comm.recv(individualIter, source = process + 1, tag = 2)

            if individualIter is not None:
                # Both values must therefore be non-null.
                iterationList.append(individualIter)
                timeList.append(individualTime)

        avgRuntime = np.average(timeList)
        avgIters = np.average(iterationList)
        try:

            fastestTime = np.min(timeList)

        except ValueError:
            
            # List is empty.
            fastestTime = float('NaN')

        numSuccess = len(iterationList)
        successRate = numSuccess/numRuns

        print("Bayesian optimisation paper testing complete! Here are the stats:")
        print("Number of successful runs: " + str(numSuccess) + " (Success rate of " + str(successRate) + ")")
        print("Average iterations required for success: " + str(avgIters))
        print("Average time required for success: " + str(avgRuntime))
        print("Fastest convergence time: " + str(fastestTime))
        print("------------------------------------------------------------------------------------------------------------------")
    
    return

def Bayes5DStats(numIters, numRuns):
    """
    Obtain the 5D comparison stats.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    assert comm.Get_size() == numRuns, "Please ensure there is one process running per run i.e " + str(numRuns) + " processes."
    
    problemBounds = {"Bfield": choco.uniform(10, 1300), "T": choco.uniform(50, 230), "Btheta": choco.uniform(0, 90), "Etheta": choco.uniform(0, 90), "Bphi": choco.uniform(0, 90)}

    # Set up the database for the chocolate optimiser.
    connection = choco.SQLiteConnection("sqlite:///bayes_5D_" + str(rank) + "_db.db")

    if rank == 0:
        timeList = []
        bestFoMList = []

    # Define which solver will be used.
    solver = choco.Bayes(connection, problemBounds, utility_function = "ei", n_bootstrap = int(np.ceil(numIters/10)), clear_db = True)

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
        fEval = abs(Fitness5D(**nextParams))

        # Update best FoM.
        if fEval > bestFoM:
            bestFoM = fEval
        
        # Tell the optimiser about the result.
        solver.update(token, fEval)

    # One run complete.
    timeElapsed = time.time() - startTime
    # Run complete. Send results to main process. Tags are unique identifiers.
    if rank != 0:
        comm.send(timeElapsed, dest = 0, tag = 1)
        comm.send(bestFoM, dest = 0, tag = 2)
    
    # Wait for all the processes to end.
    comm.Barrier()
    
    if rank == 0:
        # Add own data first.
        bestFoMList.append(bestFoM)
        timeList.append(timeElapsed)

        for process in range(comm.Get_size() - 1):
            # Get the data.
            individualTime = None
            individualTime = comm.recv(individualTime, source = process + 1, tag = 1)

            individualFoM = None
            individualFoM = comm.recv(individualFoM, source = process + 1, tag = 2)

            bestFoMList.append(individualFoM)
            timeList.append(individualTime)

        avgRuntime = np.average(timeList)
        avgFoM = np.average(bestFoMList)
        avgFoMPerTime = np.average(np.divide(bestFoMList, timeList))
        avgFoMPerIter = np.average(np.divide(bestFoMList, numIters))
        absBestFoM = np.max(bestFoMList)

        print("Bayesian optimisation 5D testing complete! Here are the stats:")
        print("Average runtime per run (s): " + str(avgRuntime))
        print("Average FoM: " + str(avgFoM))
        print("Average FoM per unit time: " + str(avgFoMPerTime))
        print("Average FoM per unit iteration: " + str(avgFoMPerIter))
        print("Absolute best FoM determined: " + str(absBestFoM))
        print("------------------------------------------------------------------------------------------------------------------")
        
    return

def SkoptPaperStats(maxIters, numRuns):
    """
    Obtain the paper comparison stats.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    assert comm.Get_size() == numRuns, "Please ensure there is one process running per run i.e " + str(numRuns) + " processes."
    
    # Define the problem bounds.
    skoptBounds = [(10, 1300), (40, 230), (0, 90), (0, 90)]

    # Use the same seed list as previously.
    seedList = [572505, 357073, 584216, 604873, 854690, 573165, 298975, 650770, 243921, 191168]

    # The target for each algorithm. This was determined by using the values in the literature, so there is clearly some deviation either due to the detuning or computation.
    globalFoM = 1.033

    if rank == 0:
        timeList = []
        iterationList = []

    # Define which solver will be used.
    optimiser = skopt.Optimizer(skoptBounds, base_estimator = "GP", n_initial_points = int(np.ceil(maxIters/10)), random_state = seedList[rank])

    # Start timing.
    startTime = time.time()
    timeElapsed = None
    iterationSuccess = None

    # Start optimisation.
    for iteration in range(maxIters):

        # Make one suggestion.
        nextParams = optimiser.ask()

        # Check what FoM this gives. Go negative as this is a minimisation routine.
        fEval =  FitnessSkopt(nextParams)

        # Update best FoM.
        if abs(fEval) >= globalFoM:
            # The algorithm has managed to surpass or equal the paper value.
            iterationSuccess = iteration
            timeElapsed = time.time() - startTime
            
            if rank == 0:
                iterationList.append(iterationSuccess)
                timeList.append(timeElapsed)

            break
        
        # Tell the optimiser about the result.
        optimiser.tell(nextParams, fEval)

    # Run complete. Send results to main process. Tags are unique identifiers.
    if rank != 0:
        comm.send(timeElapsed, dest = 0, tag = 1)
        comm.send(iterationSuccess, dest = 0, tag = 2)

    # Wait for all the processes to end.
    comm.Barrier()

    if rank == 0:
        # Aggregate the data.
        for process in range(comm.Get_size() - 1):
            # Get the data.
            individualTime = None
            individualTime = comm.recv(individualTime, source = process + 1, tag = 1)

            individualIter = None
            individualIter = comm.recv(individualIter, source = process + 1, tag = 2)

            if individualIter is not None:
                # Both values must therefore be non-null.
                iterationList.append(individualIter)
                timeList.append(individualTime)

        avgRuntime = np.average(timeList)
        avgIters = np.average(iterationList)
        try:

            fastestTime = np.min(timeList)

        except ValueError:
            
            # List is empty.
            fastestTime = float('NaN')

        numSuccess = len(iterationList)
        successRate = numSuccess/numRuns

        print("Bayesian optimisation paper testing complete! Here are the stats:")
        print("Number of successful runs: " + str(numSuccess) + " (Success rate of " + str(successRate) + ")")
        print("Average iterations required for success: " + str(avgIters))
        print("Average time required for success: " + str(avgRuntime))
        print("Fastest convergence time: " + str(fastestTime))
        print("------------------------------------------------------------------------------------------------------------------")
    
    return

def Skopt5DStats(numIters, numRuns):
    """
    Obtain the 5D comparison stats.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    assert comm.Get_size() == numRuns, "Please ensure there is one process running per run i.e " + str(numRuns) + " processes."
    
    # Define the problem bounds.
    skoptBounds = [(10, 1300), (40, 230), (0, 90), (0, 90), (0, 90)]

    # Use the seedlist from the other runs.
    seedList = [843484, 61806, 570442, 867402, 192390, 60563, 899483, 732848, 243267, 439621] 

    if rank == 0:
        timeList = []
        bestFoMList = []

    # Define which solver will be used.
    optimiser = skopt.Optimizer(skoptBounds, base_estimator = "RF", n_initial_points = int(np.ceil(numIters/10)), random_state = seedList[rank])

    # Start timing.
    startTime = time.time()
    bestFoM = 0

    # Start optimisation.
    for iteration in range(numIters):

        # Find out which point to sample next.
        nextParams = optimiser.ask()

        # Evaluate the objective function.
        nextFoM = FitnessSkopt5D(nextParams)

        if abs(nextFoM) > bestFoM:
            bestFoM = abs(nextFoM)
        
        # Update the model.
        optimiser.tell(nextParams, nextFoM)

    # One run complete.
    timeElapsed = time.time() - startTime
    # Run complete. Send results to main process. Tags are unique identifiers.
    if rank != 0:
        comm.send(timeElapsed, dest = 0, tag = 1)
        comm.send(bestFoM, dest = 0, tag = 2)
    
    # Wait for all the processes to end.
    comm.Barrier()
    
    if rank == 0:
        # Add own data first.
        bestFoMList.append(bestFoM)
        timeList.append(timeElapsed)

        for process in range(comm.Get_size() - 1):
            # Get the data.
            individualTime = None
            individualTime = comm.recv(individualTime, source = process + 1, tag = 1)

            individualFoM = None
            individualFoM = comm.recv(individualFoM, source = process + 1, tag = 2)

            bestFoMList.append(individualFoM)
            timeList.append(individualTime)

        avgRuntime = np.average(timeList)
        avgFoM = np.average(bestFoMList)
        avgFoMPerTime = np.average(np.divide(bestFoMList, timeList))
        avgFoMPerIter = np.average(np.divide(bestFoMList, numIters))
        absBestFoM = np.max(bestFoMList)

        print("Bayesian optimisation 5D testing complete! Here are the stats:")
        print("Average runtime per run (s): " + str(avgRuntime))
        print("Average FoM: " + str(avgFoM))
        print("Average FoM per unit time: " + str(avgFoMPerTime))
        print("Average FoM per unit iteration: " + str(avgFoMPerIter))
        print("Absolute best FoM determined: " + str(absBestFoM))
        print("------------------------------------------------------------------------------------------------------------------")
        
    return

if __name__ == "__main__":

    # Run the paper test.
    #BayesPaperStats(1000, 10)
    #SkoptPaperStats(1000, 10)

    # Run the 5D test.
    #Bayes5DStats(1000, 10)
    Skopt5DStats(1000, 10)