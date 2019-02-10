"""
This module splits the beam into two so that they can be recombined after undergoing different processes.
"""

import numpy as np
from elecsus import elecsus_methods as elecsus
from mpi4py import MPI
from scipy.integrate import simps as integrate
from os.path import isfile
from skopt import Optimizer, forest_minimize, load
from skopt.callbacks import CheckpointSaver
import gc
from tqdm import tqdm

# Define some global parameters. NOTE: The two base filters have the same length in this version.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def CircPolFitness(inputParams):
    """
    This fitness function has two independent beams which go through a single filter each. Before this,
    one of the beams passes through a right circular polariser, and the other through the left. It is
    assumed that these beams are parallel and have the same linear polarisation angle before starting.
    The input variables for this function are:
    - E theta
    - B field 1
    - Temp 1
    - B theta 1
    - B phi 1
    - B field 2
    - Temp 2
    - B theta 2
    - B phi 2
    After the filter, the electric fields are combined additively, and pass through a final polariser
    which is perpendicular to the input polarisation to ensure a convergent integral.
    """

    filterRParams = baseParamsFilter1.copy()
    filterLParams = baseParamsFilter2.copy()

    filterRParams["Etheta"] = np.deg2rad(inputParams[0])
    filterRParams["Bfield"] = inputParams[1]
    filterRParams["T"] = inputParams[2]
    filterRParams["Btheta"] = np.deg2rad(inputParams[3])
    filterRParams["Bphi"] = np.deg2rad(inputParams[4])
    filterLParams["Bfield"] = inputParams[5]
    filterLParams["T"] = inputParams[6]
    filterLParams["Btheta"] = np.deg2rad(inputParams[7])
    filterLParams["Bphi"] = np.deg2rad(inputParams[8])

    # Both filters have the same input field.
    inputE = np.array([np.cos(filterRParams["Etheta"]), np.sin(filterRParams["Etheta"]), 0])

    # Define the Jones matrices for right and left circular polarisers.
    rightPolariser = 0.5 * np.matrix([[1, 1j, 0],
								    [-1j, 1, 0],
                                    [0, 0, 1]])

    leftPolariser = 0.5 * np.matrix([[1, -1j, 0],
								    [1j, 1, 0],
                                    [0, 0, 1]])

    # Create the polarised fields.
    rightField = np.array(inputE * rightPolariser)[0]
    leftField = np.array(inputE * leftPolariser)[0]

    # Put each field through their own filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputER] = elecsus.calculate(globalDetuning, rightField, filterRParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the right filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterRParams))
        print("Input field: " + str(rightField))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEL] = elecsus.calculate(globalDetuning, leftField, filterLParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the left filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterLParams))
        print("Input field: " + str(leftField))
        return 0.0

    # Recombine the two fields to form the total output field.
    combinedField = np.array(outputER) + np.array(outputEL)

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filterRParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * combinedField)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    ENBW = ((integrate(filterTransmission, globalDetuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print("Filter R parameters:")
        print(str(filterRParams))
        print("Filter L parameters:")
        print(str(filterLParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def TestCircPolFitness():
    """
    This test ensures the fitness method is correct by having the filter be the same for both beams.
    This should result in similar outputs to a single filter case without split beams.
    """

    print("Testing fitness function, we're looking for around 1.03388391:")

    inputParams = [6, 230, 126, 83, 0, 230, 126, 83, 0]

    print(abs(CircPolFitness(inputParams)))

    return

def OptimiseCircPol(numIters):
    """
    Run the optimisation algorithm.
    """

    # Define the problem bounds.
    problemBounds = [(0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]

    result = forest_minimize(CircPolFitness, problemBounds, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), verbose = True, base_estimator = "ET", acq_func = "LCB")

    print("Optimisation of circular polarisations complete! Here are the stats:")
    print(result.x)
    print(result.fun)

    return

def OptimiseCircPolMPI(numItersPerCore):
    """
    Optimise the fitness function over multiple processes.
    How's your project going anyways, I heard you managed to get an amazing filter, even better than the one before! What kinda FoM did you manage to get?
    NOTE: Old models are removed to prevent the program becoming too bloated.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]
        optimizer = Optimizer(dimensions = problemBounds, base_estimator = "ET", n_initial_points = int(np.ceil(numItersPerCore * numCores/10)), acq_func = "LCB")
        pbar = tqdm(total = numItersPerCore)

    for _ in range(numItersPerCore):
        if rank == 0:
            nextPoint = optimizer.ask(n_points = numCores)
            if len(optimizer.models) != 0:
                # Clear old models, to prevent memory overload.
                optimizer.models = [optimizer.models[-1]]
            
        else:
            nextPoint = None
        
        nextPoint = comm.scatter(nextPoint, root = 0)

        nextVal = CircPolFitness(nextPoint)

        nextVals = comm.gather(nextVal, root = 0)
        nextPoints = comm.gather(nextPoint, root = 0)

        if rank == 0:
            optimizer.tell(nextPoints, nextVals)

            # Update the progress bar.
            pbar.update(1)
        
        # Collect unused memory. NOTE: May be overkill/useless.
        gc.collect()

    # Optimisation complete.
    if rank == 0:
        pbar.close()
        print("\nOptimisation for circular polarisers complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(min(optimizer.yi)), len(optimizer.yi), optimizer.Xi[np.argmin(optimizer.yi)]))

    return

def PhaseDiffFitness(inputParams):
    """
    This fitness function has two independent beams which go through a single filter each. Before this,
    one of the beams passes through a waveplate which introduces a (given) phase difference between the two waves without attenuation. 
    It is assumed that these beams are parallel and have the same linear polarisation angle before starting.
    The input variables for this function are:
    - E theta
    - B field 1
    - Temp 1
    - B theta 1
    - B phi 1
    - B field 2
    - Temp 2
    - B theta 2
    - B phi 2
    - Phase difference between two waves.
    After the filters, the electric fields are combined additively, and pass through a final polariser
    which is perpendicular to the input polarisation to ensure a convergent integral.
    """

    # The slow side is the side being affected by the waveplate.
    filterSlowParams = baseParamsFilter1.copy()
    filterNormParams = baseParamsFilter2.copy()

    filterSlowParams["Etheta"] = np.deg2rad(inputParams[0])
    filterSlowParams["Bfield"] = inputParams[1]
    filterSlowParams["T"] = inputParams[2]
    filterSlowParams["Btheta"] = np.deg2rad(inputParams[3])
    filterSlowParams["Bphi"] = np.deg2rad(inputParams[4])
    filterNormParams["Bfield"] = inputParams[5]
    filterNormParams["T"] = inputParams[6]
    filterNormParams["Btheta"] = np.deg2rad(inputParams[7])
    filterNormParams["Bphi"] = np.deg2rad(inputParams[8])

    # Both filters have the same input field. NOTE: Division by 2 as intensity is quartered due to splitting (why isn't this needed for circular polarisers?).
    inputE = np.array([np.cos(filterSlowParams["Etheta"]), np.sin(filterSlowParams["Etheta"]), 0])/2

    # Apply the waveplate action.
    waveplateMatrix = np.exp(-1j * np.deg2rad(inputParams[9])) * np.matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    # Create the polarised fields.
    slowField = np.array(inputE * waveplateMatrix)[0]
    normField = np.array(inputE)

    # Put each field through their own filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputESlow] = elecsus.calculate(globalDetuning, slowField, filterSlowParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the slow filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterSlowParams))
        print("Input field: " + str(slowField))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputENorm] = elecsus.calculate(globalDetuning, normField, filterNormParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the normal filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterNormParams))
        print("Input field: " + str(normField))
        return 0.0

    # Recombine the two fields to form the total output field.
    combinedField = np.array(outputESlow) + np.array(outputENorm)

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filterSlowParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * combinedField)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    ENBW = ((integrate(filterTransmission, globalDetuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print("Filter slow parameters:")
        print(str(filterSlowParams))
        print("Filter normal parameters:")
        print(str(filterNormParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def TestPhaseDiffFitness():
    """
    This test ensures the fitness method is correct by having the filter be the same for both beams, with no phase difference.
    This should result in similar outputs to a single filter case without split beams.
    """

    print("Testing fitness function, we're looking for around 1.03388391:")

    inputParams = [6, 230, 126, 83, 0, 230, 126, 83, 0, 0]
    print(abs(PhaseDiffFitness(inputParams)))

    print("Looking for around zero due to destructive interference:")

    inputParams = [6, 230, 126, 83, 0, 230, 126, 83, 0, 180]
    print(abs(PhaseDiffFitness(inputParams)))
    
    return

def OptimisePhaseDiff(numIters):
    """
    Run the optimisation algorithm.
    """

    # Define the problem bounds.
    problemBounds = [(0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.)]

    result = forest_minimize(PhaseDiffFitness, problemBounds, verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)), base_estimator = "ET", acq_func = "LCB")

    print("Optimisation of waveplate setup complete! Here are the stats:")
    print(result.x)
    print(result.fun)

    return

def OptimisePhaseDiffMPI(numItersPerCore):
    """
    Optimise the fitness function over multiple processes.
    NOTE: Old models are removed to prevent the program becoming too bloated.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.)]
        optimizer = Optimizer(dimensions = problemBounds, base_estimator = "ET", n_initial_points = int(np.ceil(numItersPerCore * numCores/10)), acq_func = "LCB")
        pbar = tqdm(total = numItersPerCore)

    for _ in range(numItersPerCore):
        
        if rank == 0:
            nextPoint = optimizer.ask(n_points = numCores)
            if len(optimizer.models) != 0:
                # Clear old models, to prevent memory overload.
                optimizer.models = [optimizer.models[-1]]
            
        else:
            nextPoint = None
        
        nextPoint = comm.scatter(nextPoint, root = 0)

        nextVal = PhaseDiffFitness(nextPoint)

        nextVals = comm.gather(nextVal, root = 0)
        nextPoints = comm.gather(nextPoint, root = 0)

        if rank == 0:
            optimizer.tell(nextPoints, nextVals)

            # Update the progress bar.
            pbar.update(1)
        
        # Collect unused memory. NOTE: May be overkill/useless.
        gc.collect()

    # Optimisation complete.
    if rank == 0:
        pbar.close()
        print("\nOptimisation for waveplate complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(min(optimizer.yi)), len(optimizer.yi), optimizer.Xi[np.argmin(optimizer.yi)]))

    return

if __name__ == "__main__":
    # Test the accuracy of the circular polarisers fitness function.
    #TestCircPolFitness()

    # Run the optimisation.
    #OptimiseCircPol(6000)

    # Run the optimisation (MPI).
    OptimiseCircPolMPI(1000)

    # Test the accuracy of the circular polarisers fitness function.
    #TestPhaseDiffFitness()

    # Run the optimisation.
    #OptimisePhaseDiff(6000)
    
    # Run the optimisation (MPI).
    #OptimisePhaseDiffMPI(1000)
