"""
A single filter followed by another for the light rejected by the final polariser.
"""

import numpy as np
from elecsus import elecsus_methods as elecsus
from mpi4py import MPI
from scipy.integrate import simps as integrate
import pandas as pd
from os.path import isfile
from skopt import Optimizer, forest_minimize, load
from skopt.callbacks import CheckpointSaver
import gc
from tqdm import tqdm
import scipy.optimize as opt
from sklearn.externals import joblib

# Define some global parameters.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 50e-3, "Dline": "D2", "rb85frac": 72.17}

def RejectFilterFitness(inputParams):
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
    filterRejectParams = baseParamsFilter2.copy()

    filter1Params["Bfield"] = inputParams[0]
    filter1Params["T"] = inputParams[1]
    filter1Params["Etheta"] = np.deg2rad(inputParams[2])
    filter1Params["Btheta"] = np.deg2rad(inputParams[3])
    filter1Params["Bphi"] = np.deg2rad(inputParams[4])
    filterRejectParams["Bfield"] = inputParams[5]
    filterRejectParams["T"] = inputParams[6]
    filterRejectParams["Btheta"] = np.deg2rad(inputParams[7])
    filterRejectParams["Bphi"] = np.deg2rad(inputParams[8])

    # First generate the output transmission for the first filter.
    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])/2

    # Call ElecSus to obtain the output electric field from the first filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE1] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the first filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filter1Params["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputEPol = np.array(jonesMatrix * outputE1)

    outputERej = outputE1 - outputEPol

    # Call ElecSus to obtain the output field from the second filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE2] = elecsus.calculate(globalDetuning, outputERej, filterRejectParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the second filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterRejectParams))
        print("Input field: " + str(outputE1))
        return 0.0

    # Pass the rejected light through a polariser.
    # Use a Jones matrix to determine the electric field after the action of the polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngleRej = filter1Params["Etheta"]

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrixRej = np.matrix([[np.cos(polariserAngleRej)**2, np.sin(polariserAngleRej)*np.cos(polariserAngleRej), 0],
								[np.sin(polariserAngleRej)*np.cos(polariserAngleRej), np.sin(polariserAngleRej)**2, 0],
                                [0, 0, 1]])

    outputEPol2 = np.array(jonesMatrixRej * outputE2)

    outputE = outputEPol + outputEPol2

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
        print(str(filterRejectParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def OptimiseRejectMPI(numItersPerCore, toSave = False):
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
        problemBounds = [(10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.)]
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

        nextVal = RejectFilterFitness(nextPoint)

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
        finResult = opt.minimize(RejectFilterFitness, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for rejected filter complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
        if toSave:
            print("Updating the model and saving model for SHAP analysis.")
            modelResult = optimizer.tell(finResult.x.tolist(), finResult.fun)
            bestIndex = np.argwhere(modelResult.func_vals == finResult.fun)[0][0]

            # Create a Pandas dataframe to hold the optimiser results.
            resultDataframe = pd.DataFrame(optimizer.Xi, columns = [r"$|\textbf{B}_{\textrm{1}}|$", r"$T_{\textrm{1}}$", r"$\theta_{\textrm{E}}$", 
            r"$\theta_{\textrm{B}_1}$", r"$\phi_{\textrm{B}_1}$", r"$|\textbf{B}_{\textrm{2}}|$", r"$T_{\textrm{2}}$", r"$\theta_{\textrm{B}_2}$", r"$\phi_{\textrm{B}_2}$"])

            # Save the model in a pickle file.
            joblib.dump((optimizer.models[-1], resultDataframe, bestIndex), "shap_data_three_filter_split.pkl")
            print("Data saved to shap_data_rejected_light.pkl.")

    return

if __name__ == "__main__":
    # Run the optimisation.
    OptimiseRejectMPI(1000, toSave = False)