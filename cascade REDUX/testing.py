"""
Similar to the three_split script, however the parameters for the final filter
and the angles of the PBS and waveplate are free. The first two filters
have their parameters fixed because they look good.
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
from qutip.bloch import Bloch
import pandas as pd
import matplotlib.pyplot as plt
#plt.rc("text", usetex = True)

# Define some global parameters. NOTE: The two base filters have the same length in this version.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def ThreeFilter(inputParams):
    """
    This fitness function has two independent beams which go through a single filter each. It is
    assumed that these beams are parallel and have the same linear polarisation angle before starting.
    The input variables for this function are:
    - B field 3
    - Temp 3
    - B theta 3
    - B phi 3
    - Waveplate angle
    - PBS angle
    After the filter, the electric fields are combined using a PBS, and pass through a final filter and a polariser
    which is perpendicular to the input polarisation to ensure a convergent integral.
    """

    filterRParams = baseParamsFilter1.copy()
    filterLParams = baseParamsFilter2.copy()
    filterFinParams = baseParamsFilter1.copy()

    filterRParams["Etheta"] = np.deg2rad(87.61085044)
    filterRParams["Bfield"] = 343.66864345
    filterRParams["T"] = 76.11772531
    filterRParams["Btheta"] = np.deg2rad(5.09080708)
    filterRParams["Bphi"] = np.deg2rad(42.19671567)
    filterLParams["Bfield"] = 143.9819049
    filterLParams["T"] = 129.29791277
    filterLParams["Btheta"] = np.deg2rad(82.58289292)
    filterLParams["Bphi"] = np.deg2rad(1.73454687)    
    filterFinParams["Bfield"] = 280.37227475
    filterFinParams["T"] = 114.23252712
    filterFinParams["Btheta"] = np.deg2rad(87.23663536)
    filterFinParams["Bphi"] = np.deg2rad(90.)

    # Both filters have the same input field. Normalised for intensity to be 1.
    # NOTE: The scaling is 0.5, from comparing with known results.
    inputE = np.array([np.cos(filterRParams["Etheta"]), np.sin(filterRParams["Etheta"]), 0])/2

    # Put each field through their own filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputER] = elecsus.calculate(globalDetuning, inputE, filterRParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the right filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterRParams))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEL] = elecsus.calculate(globalDetuning, inputE, filterLParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the left filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterLParams))
        print("Input field: " + str(inputE))
        return 0.0

    # qWaveplateAng1L = np.deg2rad(inputParams[0])
    # hWaveplateAngL = np.deg2rad(inputParams[1])
    # qWaveplateAng2L = np.deg2rad(inputParams[2])

    # # qWaveplateAng1R = np.deg2rad(inputParams[3])
    # # hWaveplateAngR = np.deg2rad(inputParams[4])
    # # qWaveplateAng2R = np.deg2rad(inputParams[5])

    # qWaveplate1L = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng1L)**2 + (1j * np.sin(qWaveplateAng1L)**2), (1- 1j) * np.sin(qWaveplateAng1L) * np.cos(qWaveplateAng1L), 0],
    # [(1- 1j) * np.sin(qWaveplateAng1L) * np.cos(qWaveplateAng1L), np.sin(qWaveplateAng1L)**2 + (1j * np.cos(qWaveplateAng1L)**2), 0],
    # [0, 0, 1]])

    # hWaveplateL = -1j * np.matrix([[np.cos(hWaveplateAngL), np.sin(hWaveplateAngL), 0],
	# 							[np.sin(hWaveplateAngL), -1 * np.cos(hWaveplateAngL), 0],
    #                             [0, 0, 1]])

    # qWaveplate2L = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng2L)**2 + (1j * np.sin(qWaveplateAng2L)**2), (1- 1j) * np.sin(qWaveplateAng2L) * np.cos(qWaveplateAng2L), 0],
    # [(1- 1j) * np.sin(qWaveplateAng2L) * np.cos(qWaveplateAng2L), np.sin(qWaveplateAng2L)**2 + (1j * np.cos(qWaveplateAng2L)**2), 0],
    # [0, 0, 1]])

    # rotatedLeft = np.array(qWaveplate2L * hWaveplateL * qWaveplate1L * np.array(outputEL)[:])

    # # qWaveplate1R = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng1R)**2 + (1j * np.sin(qWaveplateAng1R)**2), (1- 1j) * np.sin(qWaveplateAng1R) * np.cos(qWaveplateAng1R), 0],
    # # [(1- 1j) * np.sin(qWaveplateAng1R) * np.cos(qWaveplateAng1R), np.sin(qWaveplateAng1R)**2 + (1j * np.cos(qWaveplateAng1R)**2), 0],
    # # [0, 0, 1]])

    # # hWaveplateR = -1j * np.matrix([[np.cos(hWaveplateAngR), np.sin(hWaveplateAngR), 0],
	# # 							[np.sin(hWaveplateAngR), -1 * np.cos(hWaveplateAngR), 0],
    # #                             [0, 0, 1]])

    # # qWaveplate2R = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng2R)**2 + (1j * np.sin(qWaveplateAng2R)**2), (1- 1j) * np.sin(qWaveplateAng2R) * np.cos(qWaveplateAng2R), 0],
    # # [(1- 1j) * np.sin(qWaveplateAng2R) * np.cos(qWaveplateAng2R), np.sin(qWaveplateAng2R)**2 + (1j * np.cos(qWaveplateAng2R)**2), 0],
    # # [0, 0, 1]])

    # # rotatedRight = np.array(qWaveplate2R * hWaveplateR * qWaveplate1R * np.array(outputER)[:])

    # # Now pass both beams through a PBS. The PBS is aligned such that the semimajor axis of light in the first input passes.
    # # This will be merged with light along the perpendicular axis of the second input.
    # pbsAng = np.deg2rad(inputParams[3])
    # pbsMatrixAccept = np.matrix([[np.cos(pbsAng)**2, np.sin(pbsAng)*np.cos(pbsAng), 0],
	# 							[np.sin(pbsAng)*np.cos(pbsAng), np.sin(pbsAng)**2, 0],
    #                             [0, 0, 1]])

    # acceptedFromInput1 = np.array(pbsMatrixAccept * np.array(outputER)[:])
    # rejectedFromInput2 = np.array(pbsMatrixAccept * rotatedLeft)
    # acceptedFromInput2 = rotatedLeft - rejectedFromInput2

    # combinedField = acceptedFromInput1 + acceptedFromInput2
    combinedField = outputER + outputEL

    # Pass the combined field through a final filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEFin] = elecsus.calculate(globalDetuning, combinedField, filterFinParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the final filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterFinParams))
        print("Input field: " + str(combinedField))
        return 0.0

    # qWaveplateAng1F = np.deg2rad(inputParams[4])
    # hWaveplateAngF = np.deg2rad(inputParams[5])
    # qWaveplateAng2F = np.deg2rad(inputParams[6])

    # qWaveplate1F = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng1F)**2 + (1j * np.sin(qWaveplateAng1F)**2), (1- 1j) * np.sin(qWaveplateAng1F) * np.cos(qWaveplateAng1F), 0],
    # [(1- 1j) * np.sin(qWaveplateAng1F) * np.cos(qWaveplateAng1F), np.sin(qWaveplateAng1F)**2 + (1j * np.cos(qWaveplateAng1F)**2), 0],
    # [0, 0, 1]])

    # hWaveplateF = -1j * np.matrix([[np.cos(hWaveplateAngF), np.sin(hWaveplateAngF), 0],
	# 							[np.sin(hWaveplateAngF), -1 * np.cos(hWaveplateAngF), 0],
    #                             [0, 0, 1]])

    # qWaveplate2F = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng2F)**2 + (1j * np.sin(qWaveplateAng2F)**2), (1- 1j) * np.sin(qWaveplateAng2F) * np.cos(qWaveplateAng2F), 0],
    # [(1- 1j) * np.sin(qWaveplateAng2F) * np.cos(qWaveplateAng2F), np.sin(qWaveplateAng2F)**2 + (1j * np.cos(qWaveplateAng2F)**2), 0],
    # [0, 0, 1]])

    # rotatedFinal = np.array(qWaveplate2F * hWaveplateF * qWaveplate1F * np.array(outputEFin)[:])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = np.deg2rad(inputParams[4])

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * outputEFin)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    # plt.plot(globalDetuning, filterTransmission)
    # plt.show()

    #assert filterTransmission.max() <= 1., "Maximal transmission is greater than 1, ensure your electric fields are correct in magnitude."

    ENBW = ((integrate(filterTransmission, globalDetuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real

    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print("Filter R parameters:")
        print(str(filterRParams))
        print("Filter L parameters:")
        print(str(filterLParams))
        print("Final filter parameters:")
        print(str(filterFinParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def OptimiseThreeFilter(numItersPerCore, toSave = False):
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
        problemBounds = [(0., 180.), (0., 180.), (0., 180.), (0., 180.), (0., 180.)]
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

        nextVal = ThreeFilter(nextPoint)

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
        finResult = opt.minimize(ThreeFilter, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
        print("\nOptimisation for three filters (split, PBS, left polarised to match right peak) complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
        if toSave:
            print("Updating the model and saving model for SHAP analysis.")
            modelResult = optimizer.tell(finResult.x.tolist(), finResult.fun)
            bestIndex = np.argwhere(modelResult.func_vals == finResult.fun)[0][0]

            # Create a Pandas dataframe to hold the optimiser results.
            resultDataframe = pd.DataFrame(optimizer.Xi, columns = [r"$|\textbf{B}_{\textrm{1}}|$", r"$T_{\textrm{1}}$", r"$\theta_{\textrm{E}}$", 
            r"$\theta_{\textrm{B}_1}$", r"$\phi_{\textrm{B}_1}$", r"$|\textbf{B}_{\textrm{2}}|$", r"$T_{\textrm{2}}$", r"$\theta_{\textrm{B}_2}$", r"$\phi_{\textrm{B}_2}$",
            r"$|\textbf{B}_{\textrm{3}}|$", r"$T_{\textrm{3}}$", r"$\theta_{\textrm{B}_3}$", r"$\phi_{\textrm{B}_3}$"])

            # Save the model in a pickle file.
            joblib.dump((optimizer.models[-1], resultDataframe, bestIndex), "shap_data_final_filter.pkl")
            print("Data saved to shap_data_final_filter.pkl.")

    return

if __name__ == "__main__":
    # Run the function.
    print(ThreeFilter([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90.]))
    # print(ThreeFilter([np.rad2deg(1.2729563270779147), 653.2026710391845, 59.07061955395512, np.rad2deg(0.022582921277005945), np.rad2deg(1.0222099201704498),
    # 103.95733628450398, 194.50243952818656, np.rad2deg(1.2197690515735653), np.rad2deg(1.2151585412088093),
    # 889.5920681456691, 46.09141348824217, np.rad2deg(1.2489169286173634), np.rad2deg(0.239471815289869)]))

    # Run the function.
    #ThreeFilter([114.54936821, 0., 356.94295062, 312.07698169])

    # Run the optimisation.
    #OptimiseThreeFilter(1000, toSave = False)