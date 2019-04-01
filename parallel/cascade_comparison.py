"""
Generates a graph of number of filters against maximal figure of merit and ENBW.
This will be used to justify the claim that cascading filters seems to have a limit.
An ultra narrow detuning spacing is used to ensure really good results are captured too.
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
import scipy.optimize as opt
from sklearn.externals import joblib
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("text", usetex = True)

import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

# Define some global parameters.
globalDetuning = np.sort(np.append(np.linspace(-20000, 20000, 1000), np.linspace(-500, 500, 10000)))
baseParams = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def SingleFilter(inputParams):
    """
    The fitness function for a single filter. No phi for the magnetic field.
    """

    filterParams = baseParams.copy()

    filterParams["Etheta"] = np.deg2rad(inputParams[0])
    filterParams["Bfield"] = inputParams[1]
    filterParams["T"] = inputParams[2]
    filterParams["Btheta"] = np.deg2rad(inputParams[3])

    inputE = np.array([np.cos(filterParams["Etheta"]), np.sin(filterParams["Etheta"]), 0])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, inputE, filterParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterParams))
        print("Input field: " + str(inputE))
        return 0.0

    polariserAngle = filterParams["Etheta"] + np.pi/2

    polariserMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    outputE = np.array(polariserMatrix * outputE)

    transmission = (outputE * outputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(transmission, globalDetuning)/transmission.max().real)/1e3).real

    figureOfMerit = (transmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN!")
        return 0.0
    else:
        return figureOfMerit, ENBW

def OptimiseSingle(numItersPerCore):
    """
    Optimises the single filter fitness function.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 180.), (10., 1300.), (40., 230.), (0., 180.)]
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

        nextVal = SingleFilter(nextPoint)

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
        print("\nGlobal optimum region found, starting local optimisation at best point.")
        finResult = opt.minimize(SingleFilter, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for single filter complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
    return

def DoubleFilter(inputParams):
    """
    The fitness function for a double filter. No phi for the magnetic fields.
    """

    filter1Params = baseParams.copy()
    filter2Params = baseParams.copy()

    filter1Params["Etheta"] = np.deg2rad(inputParams[0])
    filter1Params["Bfield"] = inputParams[1]
    filter1Params["T"] = inputParams[2]
    filter1Params["Btheta"] = np.deg2rad(inputParams[3])
    filter2Params["Bfield"] = inputParams[4]
    filter2Params["T"] = inputParams[5]
    filter2Params["Btheta"] = np.deg2rad(inputParams[6])

    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE))
        return 0.0

    polariserAngle = filter1Params["Etheta"] + np.pi/2

    polariserMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    outputE = np.array(polariserMatrix * outputE)

    transmission = (outputE * outputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(transmission, globalDetuning)/transmission.max().real)/1e3).real

    figureOfMerit = (transmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN!")
        return 0.0
    else:
        return figureOfMerit, ENBW

def OptimiseDouble(numItersPerCore):
    """
    Optimises the double filter fitness function. No phi for the magnetic fields.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 180.), (10., 1300.), (40., 230.), (0., 180.), (10., 1300.), (40., 230.), (0., 180.)]
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

        nextVal = DoubleFilter(nextPoint)

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
        print("\nGlobal optimum region found, starting local optimisation at best point.")
        finResult = opt.minimize(DoubleFilter, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for double filter complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
    return

def TripleFilter(inputParams):
    """
    Fitness function for three cascaded filters.
    """

    filter1Params = baseParams.copy()
    filter2Params = baseParams.copy()
    filter3Params = baseParams.copy()

    filter1Params["Etheta"] = np.deg2rad(inputParams[0])
    filter1Params["Bfield"] = inputParams[1]
    filter1Params["T"] = inputParams[2]
    filter1Params["Btheta"] = np.deg2rad(inputParams[3])
    filter2Params["Bfield"] = inputParams[4]
    filter2Params["T"] = inputParams[5]
    filter2Params["Btheta"] = np.deg2rad(inputParams[6])
    filter3Params["Bfield"] = inputParams[7]
    filter3Params["T"] = inputParams[8]
    filter3Params["Btheta"] = np.deg2rad(inputParams[9])

    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter3Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter3Params))
        print("Input field: " + str(outputE))
        return 0.0

    polariserAngle = filter1Params["Etheta"] + np.pi/2

    polariserMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    outputE = np.array(polariserMatrix * outputE)

    transmission = (outputE * outputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(transmission, globalDetuning)/transmission.max().real)/1e3).real

    figureOfMerit = (transmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN!")
        return 0.0
    else:
        return figureOfMerit, ENBW

def OptimiseTriple(numItersPerCore):
    """
    Optimises the triple filter fitness function. No phi for the magnetic fields.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 180.), (0., 1300.), (0., 230.), (0., 180.), (0., 1300.), (0., 230.), (0., 180.), (0., 1300.), (0., 230.), (0., 180.)]
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

        nextVal = TripleFilter(nextPoint)

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
        print("\nGlobal optimum region found, starting local optimisation at best point.")
        finResult = opt.minimize(TripleFilter, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for triple filter complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
    return

def OptimiseTripleExtra(numItersPerCore):
    """
    Optimises the triple filter setup but with the first two filters fixed. This is to see whether there is a limit on filter functionality.
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 1300.), (0., 230.), (0., 180.)]
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

        nextVal = TripleFilter(np.append([87.88, 291.38, 127.43, 92.36, 265.93, 89.38, 13.64], nextPoint))

        nextVals = comm.gather(nextVal, root = 0)
        nextPoints = comm.gather(nextPoint, root = 0)

        if rank == 0:
            result = optimizer.tell(nextPoints, nextVals)

            # Update the progress bar.
            pbar.update(1)
        
        # Collect unused memory. NOTE: May be overkill/useless.
        gc.collect()

    # Optimisation complete.
    if rank == 0:
        pbar.close()
        print("\nOptimisation for triple filter (first two fixed) complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(result.fun), len(result.func_vals), result.x))
        
    return

def QuadFilter(inputParams):
    """
    The fitness function for the quadruple filter. The first three filters are fixed at the optimal filter parameters as determined by the
    three filter (fixed) optimisation.
    """

    filter1Params = baseParams.copy()
    filter2Params = baseParams.copy()
    filter3Params = baseParams.copy()
    filter4Params = baseParams.copy()

    filter1Params["Etheta"] = np.deg2rad(87.88)
    filter1Params["Bfield"] = 291.38
    filter1Params["T"] = 127.43
    filter1Params["Btheta"] = np.deg2rad(92.36)
    filter2Params["Bfield"] = 265.93
    filter2Params["T"] = 89.38
    filter2Params["Btheta"] = np.deg2rad(13.64)
    filter3Params["Bfield"] = 351.9008560937018
    filter3Params["T"] = 91.3563061802824
    filter3Params["Btheta"] = np.deg2rad(96.06300701777786)
    filter4Params["Bfield"] = inputParams[0]
    filter4Params["T"] = inputParams[1]
    filter4Params["Btheta"] = np.deg2rad(inputParams[2])

    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter3Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter3Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter4Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter4Params))
        print("Input field: " + str(outputE))
        return 0.0

    polariserAngle = filter1Params["Etheta"] + np.pi/2

    polariserMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    outputE = np.array(polariserMatrix * outputE)

    transmission = (outputE * outputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(transmission, globalDetuning)/transmission.max().real)/1e3).real

    figureOfMerit = (transmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN!")
        return 0.0
    else:
        return figureOfMerit, ENBW

def OptimiseQuadExtra(numItersPerCore):
    """
    Optimises the quad filter fitness function.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 1300.), (0., 230.), (0., 180.)]
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

        nextVal = QuadFilter(nextPoint)

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
        print("\nGlobal optimum region found, starting local optimisation at best point.")
        finResult = opt.minimize(QuadFilter, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for quad (first three fixed) filter complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
    return

def PentaFilter(inputParams):
    """
    The fitness function for 5 filters. The first four filters are fixed at the optimal filter parameters as determined by the
    four filter (fixed) optimisation.
    """

    filter1Params = baseParams.copy()
    filter2Params = baseParams.copy()
    filter3Params = baseParams.copy()
    filter4Params = baseParams.copy()
    filter5Params = baseParams.copy()

    filter1Params["Etheta"] = np.deg2rad(87.88)
    filter1Params["Bfield"] = 291.38
    filter1Params["T"] = 127.43
    filter1Params["Btheta"] = np.deg2rad(92.36)
    filter2Params["Bfield"] = 265.93
    filter2Params["T"] = 89.38
    filter2Params["Btheta"] = np.deg2rad(13.64)
    filter3Params["Bfield"] = 351.9008560937018
    filter3Params["T"] = 91.3563061802824
    filter3Params["Btheta"] = np.deg2rad(96.06300701777786)
    filter4Params["Bfield"] = 76.08560524
    filter4Params["T"] = 81.63672624
    filter4Params["Btheta"] = np.deg2rad(169.88632875)
    filter5Params["Bfield"] = inputParams[0]
    filter5Params["T"] = inputParams[1]
    filter5Params["Btheta"] = np.deg2rad(inputParams[2])

    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter3Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter3Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter4Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter4Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter5Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter5Params))
        print("Input field: " + str(outputE))
        return 0.0

    polariserAngle = filter1Params["Etheta"] + np.pi/2

    polariserMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    outputE = np.array(polariserMatrix * outputE)

    transmission = (outputE * outputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(transmission, globalDetuning)/transmission.max().real)/1e3).real

    figureOfMerit = (transmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN!")
        return 0.0
    else:
        return figureOfMerit, ENBW

def OptimisePentaExtra(numItersPerCore):
    """
    Optimises the penta filter fitness function.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 1300.), (0., 230.), (0., 180.)]
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

        nextVal = PentaFilter(nextPoint)

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
        print("\nGlobal optimum region found, starting local optimisation at best point.")
        finResult = opt.minimize(PentaFilter, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for penta (first four fixed) filter complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
    return

def HexaFilter(inputParams):
    """
    The fitness function for 6 filters. The first four filters are fixed at the optimal filter parameters as determined by the
    5 filter (fixed) optimisation.
    """

    filter1Params = baseParams.copy()
    filter2Params = baseParams.copy()
    filter3Params = baseParams.copy()
    filter4Params = baseParams.copy()
    filter5Params = baseParams.copy()
    filter6Params = baseParams.copy()

    filter1Params["Etheta"] = np.deg2rad(87.88)
    filter1Params["Bfield"] = 291.38
    filter1Params["T"] = 127.43
    filter1Params["Btheta"] = np.deg2rad(92.36)
    filter2Params["Bfield"] = 265.93
    filter2Params["T"] = 89.38
    filter2Params["Btheta"] = np.deg2rad(13.64)
    filter3Params["Bfield"] = 351.9008560937018
    filter3Params["T"] = 91.3563061802824
    filter3Params["Btheta"] = np.deg2rad(96.06300701777786)
    filter4Params["Bfield"] = 76.08560524
    filter4Params["T"] = 81.63672624
    filter4Params["Btheta"] = np.deg2rad(169.88632875)
    filter5Params["Bfield"] = 355.82023869
    filter5Params["T"] = 85.24975747
    filter5Params["Btheta"] = np.deg2rad(96.36557205)
    filter6Params["Bfield"] = inputParams[0]
    filter6Params["T"] = inputParams[1]
    filter6Params["Btheta"] = np.deg2rad(inputParams[2])


    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter3Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter3Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter4Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter4Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter5Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter5Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter6Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter6Params))
        print("Input field: " + str(outputE))
        return 0.0

    polariserAngle = filter1Params["Etheta"] + np.pi/2

    polariserMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    outputE = np.array(polariserMatrix * outputE)

    transmission = (outputE * outputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(transmission, globalDetuning)/transmission.max().real)/1e3).real

    figureOfMerit = (transmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN!")
        return 0.0
    else:
        return figureOfMerit, ENBW

def OptimiseHexaExtra(numItersPerCore):
    """
    Optimises the 6 filter fitness function.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 1300.), (0., 230.), (0., 180.)]
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

        nextVal = HexaFilter(nextPoint)

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
        print("\nGlobal optimum region found, starting local optimisation at best point.")
        finResult = opt.minimize(HexaFilter, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for 6 (first 5 fixed) filter complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
    return

def HeptaFilter(inputParams):
    """
    The fitness function for 7 filters. The first four filters are fixed at the optimal filter parameters as determined by the
    6 filter (fixed) optimisation.
    """

    filter1Params = baseParams.copy()
    filter2Params = baseParams.copy()
    filter3Params = baseParams.copy()
    filter4Params = baseParams.copy()
    filter5Params = baseParams.copy()
    filter6Params = baseParams.copy()
    filter7Params = baseParams.copy()

    filter1Params["Etheta"] = np.deg2rad(87.88)
    filter1Params["Bfield"] = 291.38
    filter1Params["T"] = 127.43
    filter1Params["Btheta"] = np.deg2rad(92.36)
    filter2Params["Bfield"] = 265.93
    filter2Params["T"] = 89.38
    filter2Params["Btheta"] = np.deg2rad(13.64)
    filter3Params["Bfield"] = 351.9008560937018
    filter3Params["T"] = 91.3563061802824
    filter3Params["Btheta"] = np.deg2rad(96.06300701777786)
    filter4Params["Bfield"] = 76.08560524
    filter4Params["T"] = 81.63672624
    filter4Params["Btheta"] = np.deg2rad(169.88632875)
    filter5Params["Bfield"] = 355.82023869
    filter5Params["T"] = 85.24975747
    filter5Params["Btheta"] = np.deg2rad(96.36557205)
    filter6Params["Bfield"] = 359.71567833
    filter6Params["T"] = 84.65648202
    filter6Params["Btheta"] = np.deg2rad(83.30269821)
    filter7Params["Bfield"] = inputParams[0]
    filter7Params["T"] = inputParams[1]
    filter7Params["Btheta"] = np.deg2rad(inputParams[2])

    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter3Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter3Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter4Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter4Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter5Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter5Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter6Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter6Params))
        print("Input field: " + str(outputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, outputE, filter7Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for a filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter7Params))
        print("Input field: " + str(outputE))
        return 0.0

    polariserAngle = filter1Params["Etheta"] + np.pi/2

    polariserMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    outputE = np.array(polariserMatrix * outputE)

    transmission = (outputE * outputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(transmission, globalDetuning)/transmission.max().real)/1e3).real

    figureOfMerit = (transmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN!")
        return 0.0
    else:
        return figureOfMerit, ENBW

def OptimiseHeptaExtra(numItersPerCore):
    """
    Optimises the 7 filter fitness function.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 1300.), (0., 230.), (0., 180.)]
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

        nextVal = HeptaFilter(nextPoint)

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
        print("\nGlobal optimum region found, starting local optimisation at best point.")
        finResult = opt.minimize(HeptaFilter, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for 7 (first 6 fixed) filter complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
    return

def FilterVsFigure():
    """
    Plots a graph of number of cascaded filters vs figure of merit and ENBW.
    """

    # Create a list of figures of merit, ENBW pairs.
    resultList = [(SingleFilter([6, 230, 126, 83])), (DoubleFilter([87.88, 291.38, 127.43, 92.36, 265.93, 89.38, 13.64])), TripleFilter([87.88, 291.38, 127.43, 92.36, 265.93, 89.38, 13.64, 351.9008560937018, 91.3563061802824, 96.06300701777786]),
    QuadFilter([76.08560524, 81.63672624, 169.88632875]), PentaFilter([355.82023869, 85.24975747, 96.36557205]), HexaFilter([359.71567833, 84.65648202, 83.30269821]),
    HeptaFilter([358.90401126, 82.32289617, 96.61110236])]

    # Define bar width.
    width = 0.5
    fomax = plt.subplot()
    enbwax = plt.twinx(fomax)

    for position, result in enumerate(resultList):
        fomax.bar(position - width/2, result[0], width = width, color = "b")
        enbwax.bar(position + width/2, result[1], width = width, color = "g")

    fomax.plot([position - width/2 for position, _ in enumerate(resultList)], [result[0] for result in resultList], color = "c")
    enbwax.plot([position + width/2 for position, _ in enumerate(resultList)], [result[1] for result in resultList], color = "m")

    plt.xticks(np.arange(len(resultList)), [1, 2, 3, 4, 5, 6, 7])
    fomax.set_ylabel(r"Figure of Merit (GHz$^{-1}$)")
    enbwax.set_ylabel(r"Equivalent Noise Bandwidth (GHz)")
    plt.xlabel("Number of Cascaded Filters")

    plt.show()

    return

if __name__ == "__main__":
    # Run the single filter optimisation.
    #OptimiseSingle(500)

    # Run the double filter optimisation.
    #OptimiseDouble(1000)

    # Test the double filter best result.
    #print(DoubleFilter([87.88, 291.38, 127.43, 92.36, 265.93, 89.38, 13.64]))

    # Run the triple filter optimisation.
    #OptimiseTriple(2000)
    #print(TripleFilter([87.88, 291.38, 127.43, 92.36, 265.93, 89.38, 13.64, 0, 0, 0]))

    # Run the triple filter optimisation with the first two filters fixed.
    #OptimiseTripleExtra(500)

    # Optimise the quad filter (with first three filters fixed).
    #OptimiseQuadExtra(1000)

    # Optimise the penta filter (with first four filters fixed).
    #OptimisePentaExtra(1000)

    # Optimise the 6 filter (with first 5 filters fixed).
    #OptimiseHexaExtra(1000)

    #OptimiseHeptaExtra(1000)

    FilterVsFigure()