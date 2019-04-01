"""
Cascades four filters to see if it can compete with the parallel filter which in total has 4 beam-filter interactions
"""

"""
This module looks into cascading three filters and seeing how that improves the figure of merit.
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

# Define some global parameters. NOTE: The two base filters have the same length in this version.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def FourFilterFitness(inputParams):
    """
    This fitness function has one beam which passes through three filters.
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
    - B field 3
    - Temp 3
    - B theta 3
    - B phi 3
    - B field 4
    - Temp 4
    - B theta 4
    - B phi 4
    After the filter the fieldpasses through a polariser
    which is perpendicular to the input polarisation to ensure a convergent integral.
    """

    filterFirstParams = baseParamsFilter1.copy()
    filterSecondParams = baseParamsFilter2.copy()
    filterThreeParams = baseParamsFilter1.copy()
    filterFourParams = baseParamsFilter1.copy()

    filterFirstParams["Etheta"] = np.deg2rad(inputParams[0])
    filterFirstParams["Bfield"] = inputParams[1]
    filterFirstParams["T"] = inputParams[2]
    filterFirstParams["Btheta"] = np.deg2rad(inputParams[3])
    filterFirstParams["Bphi"] = np.deg2rad(inputParams[4])
    filterSecondParams["Bfield"] = inputParams[5]
    filterSecondParams["T"] = inputParams[6]
    filterSecondParams["Btheta"] = np.deg2rad(inputParams[7])
    filterSecondParams["Bphi"] = np.deg2rad(inputParams[8])    
    filterThreeParams["Bfield"] = inputParams[9]
    filterThreeParams["T"] = inputParams[10]
    filterThreeParams["Btheta"] = np.deg2rad(inputParams[11])
    filterThreeParams["Bphi"] = np.deg2rad(inputParams[12])
    filterFourParams["Bfield"] = inputParams[13]
    filterFourParams["T"] = inputParams[14]
    filterFourParams["Btheta"] = np.deg2rad(inputParams[15])
    filterFourParams["Bphi"] = np.deg2rad(inputParams[16])

    inputE = np.array([np.cos(filterFirstParams["Etheta"]), np.sin(filterFirstParams["Etheta"]), 0])

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEFirst] = elecsus.calculate(globalDetuning, inputE, filterFirstParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the right filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterFirstParams))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputESecond] = elecsus.calculate(globalDetuning, outputEFirst, filterSecondParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the left filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterSecondParams))
        print("Input field: " + str(outputEFirst))
        return 0.0

    # Pass through a third filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEThird] = elecsus.calculate(globalDetuning, outputESecond, filterThreeParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the third filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterThreeParams))
        print("Input field: " + str(outputESecond))
        return 0.0

    # Pass through a final filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEFin] = elecsus.calculate(globalDetuning, outputEThird, filterFourParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the final filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterFourParams))
        print("Input field: " + str(outputESecond))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filterFirstParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * outputEFin)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    assert filterTransmission.max() <= 1., "Maximal transmission is greater than 1, ensure your electric fields are correct in magnitude."

    ENBW = ((integrate(filterTransmission, globalDetuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print("First filter parameters:")
        print(str(filterFirstParams))
        print("Second filter parameters:")
        print(str(filterSecondParams))
        print("Final filter parameters:")
        print(str(filterThreeParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def Optimise(numItersPerCore, toSave = False):
    """
    Optimise the fitness function over multiple processes.
    NOTE: Old models are removed to prevent the program becoming too bloated.
    If toSave is true, saves the model to a pkl file for future analysis.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]
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

        nextVal = FourFilterFitness(nextPoint)

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
        finResult = opt.minimize(FourFilterFitness, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for four filters (four cascaded filters) complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
        if toSave:
            print("Updating the model and saving model for SHAP analysis.")
            modelResult = optimizer.tell(finResult.x.tolist(), finResult.fun)
            bestIndex = np.argwhere(modelResult.func_vals == finResult.fun)[0][0]

            # Create a Pandas dataframe to hold the optimiser results.
            resultDataframe = pd.DataFrame(optimizer.Xi, columns = [r"$|\textbf{B}_{\textrm{1}}|$", r"$T_{\textrm{1}}$", r"$\theta_{\textrm{E}}$", 
            r"$\theta_{\textrm{B}_1}$", r"$\phi_{\textrm{B}_1}$", r"$|\textbf{B}_{\textrm{2}}|$", r"$T_{\textrm{2}}$", r"$\theta_{\textrm{B}_2}$", r"$\phi_{\textrm{B}_2}$",
            r"$|\textbf{B}_{\textrm{3}}|$", r"$T_{\textrm{3}}$", r"$\theta_{\textrm{B}_3}$", r"$\phi_{\textrm{B}_3}$",
            r"$|\textbf{B}_{\textrm{4}}|$", r"$T_{\textrm{4}}$", r"$\theta_{\textrm{B}_4}$", r"$\phi_{\textrm{B}_4}$"])

            # Save the model in a pickle file.
            joblib.dump((optimizer.models[-1], resultDataframe, bestIndex), "shap_data_four_cascade.pkl")
            print("Data saved to shap_data_four_cascade.pkl.")

    return

if __name__ == "__main__":
    # Run the optimisation.
    Optimise(1000, toSave = True)