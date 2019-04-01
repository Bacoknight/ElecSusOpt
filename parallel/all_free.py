"""
This fitness function has the most free variables, with each field having
polarisations controllers available.
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

    filterRParams["Etheta"] = np.deg2rad(inputParams[0])
    filterRParams["Bfield"] = inputParams[1]
    filterRParams["T"] = inputParams[2]
    filterRParams["Btheta"] = np.deg2rad(inputParams[3])
    filterRParams["Bphi"] = np.deg2rad(inputParams[4])
    filterLParams["Bfield"] = inputParams[5]
    filterLParams["T"] = inputParams[6]
    filterLParams["Btheta"] = np.deg2rad(inputParams[7])
    filterLParams["Bphi"] = np.deg2rad(inputParams[8])    
    filterFinParams["Bfield"] = inputParams[9]
    filterFinParams["T"] = inputParams[10]
    filterFinParams["Btheta"] = np.deg2rad(inputParams[11])
    filterFinParams["Bphi"] = np.deg2rad(inputParams[12])

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

    qWaveplateAng1L = np.deg2rad(inputParams[13])
    hWaveplateAngL = np.deg2rad(inputParams[14])
    qWaveplateAng2L = np.deg2rad(inputParams[15])

    qWaveplateAng1R = np.deg2rad(inputParams[16])
    hWaveplateAngR = np.deg2rad(inputParams[17])
    qWaveplateAng2R = np.deg2rad(inputParams[18])

    qWaveplate1L = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng1L)**2 + (1j * np.sin(qWaveplateAng1L)**2), (1- 1j) * np.sin(qWaveplateAng1L) * np.cos(qWaveplateAng1L), 0],
    [(1- 1j) * np.sin(qWaveplateAng1L) * np.cos(qWaveplateAng1L), np.sin(qWaveplateAng1L)**2 + (1j * np.cos(qWaveplateAng1L)**2), 0],
    [0, 0, 1]])

    hWaveplateL = -1j * np.matrix([[np.cos(hWaveplateAngL), np.sin(hWaveplateAngL), 0],
								[np.sin(hWaveplateAngL), -1 * np.cos(hWaveplateAngL), 0],
                                [0, 0, 1]])

    qWaveplate2L = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng2L)**2 + (1j * np.sin(qWaveplateAng2L)**2), (1- 1j) * np.sin(qWaveplateAng2L) * np.cos(qWaveplateAng2L), 0],
    [(1- 1j) * np.sin(qWaveplateAng2L) * np.cos(qWaveplateAng2L), np.sin(qWaveplateAng2L)**2 + (1j * np.cos(qWaveplateAng2L)**2), 0],
    [0, 0, 1]])

    rotatedLeft = np.array(qWaveplate2L * hWaveplateL * qWaveplate1L * np.array(outputEL)[:])

    qWaveplate1R = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng1R)**2 + (1j * np.sin(qWaveplateAng1R)**2), (1- 1j) * np.sin(qWaveplateAng1R) * np.cos(qWaveplateAng1R), 0],
    [(1- 1j) * np.sin(qWaveplateAng1R) * np.cos(qWaveplateAng1R), np.sin(qWaveplateAng1R)**2 + (1j * np.cos(qWaveplateAng1R)**2), 0],
    [0, 0, 1]])

    hWaveplateR = -1j * np.matrix([[np.cos(hWaveplateAngR), np.sin(hWaveplateAngR), 0],
								[np.sin(hWaveplateAngR), -1 * np.cos(hWaveplateAngR), 0],
                                [0, 0, 1]])

    qWaveplate2R = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng2R)**2 + (1j * np.sin(qWaveplateAng2R)**2), (1- 1j) * np.sin(qWaveplateAng2R) * np.cos(qWaveplateAng2R), 0],
    [(1- 1j) * np.sin(qWaveplateAng2R) * np.cos(qWaveplateAng2R), np.sin(qWaveplateAng2R)**2 + (1j * np.cos(qWaveplateAng2R)**2), 0],
    [0, 0, 1]])

    rotatedRight = np.array(qWaveplate2R * hWaveplateR * qWaveplate1R * np.array(outputER)[:])

    # Now pass both beams through a PBS. The PBS is aligned such that the semimajor axis of light in the first input passes.
    # This will be merged with light along the perpendicular axis of the second input.
    pbsAng = np.deg2rad(inputParams[19])
    pbsMatrixAccept = np.matrix([[np.cos(pbsAng)**2, np.sin(pbsAng)*np.cos(pbsAng), 0],
								[np.sin(pbsAng)*np.cos(pbsAng), np.sin(pbsAng)**2, 0],
                                [0, 0, 1]])

    acceptedFromInput1 = np.array(pbsMatrixAccept * rotatedRight)
    rejectedFromInput2 = np.array(pbsMatrixAccept * rotatedLeft)
    acceptedFromInput2 = rotatedLeft - rejectedFromInput2

    combinedField = acceptedFromInput1 + acceptedFromInput2

    # Pass the combined field through a final filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEFin] = elecsus.calculate(globalDetuning, combinedField, filterFinParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the final filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterFinParams))
        print("Input field: " + str(combinedField))
        return 0.0

    qWaveplateAng1F = np.deg2rad(inputParams[20])
    hWaveplateAngF = np.deg2rad(inputParams[21])
    qWaveplateAng2F = np.deg2rad(inputParams[22])

    qWaveplate1F = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng1F)**2 + (1j * np.sin(qWaveplateAng1F)**2), (1- 1j) * np.sin(qWaveplateAng1F) * np.cos(qWaveplateAng1F), 0],
    [(1- 1j) * np.sin(qWaveplateAng1F) * np.cos(qWaveplateAng1F), np.sin(qWaveplateAng1F)**2 + (1j * np.cos(qWaveplateAng1F)**2), 0],
    [0, 0, 1]])

    hWaveplateF = -1j * np.matrix([[np.cos(hWaveplateAngF), np.sin(hWaveplateAngF), 0],
								[np.sin(hWaveplateAngF), -1 * np.cos(hWaveplateAngF), 0],
                                [0, 0, 1]])

    qWaveplate2F = np.exp(-1j * np.pi/4.) * np.matrix([[np.cos(qWaveplateAng2F)**2 + (1j * np.sin(qWaveplateAng2F)**2), (1- 1j) * np.sin(qWaveplateAng2F) * np.cos(qWaveplateAng2F), 0],
    [(1- 1j) * np.sin(qWaveplateAng2F) * np.cos(qWaveplateAng2F), np.sin(qWaveplateAng2F)**2 + (1j * np.cos(qWaveplateAng2F)**2), 0],
    [0, 0, 1]])

    rotatedFinal = np.array(qWaveplate2F * hWaveplateF * qWaveplate1F * np.array(outputEFin)[:])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = np.arctan(np.divide(rotatedFinal[0][1].real, rotatedFinal[0][0].real)) + np.pi/2.

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * rotatedFinal)

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
        problemBounds = [(0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 180.), (0., 180.), (0., 180.),
        (0., 180.), (0., 180.), (0., 180.), (0., 180.), (0., 180.), (0., 180.), (0., 180.)]
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
        print("\nOptimisation for three filters (maximum freedom) complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
        if toSave:
            print("Updating the model and saving model for SHAP analysis.")
            modelResult = optimizer.tell(finResult.x.tolist(), finResult.fun)
            bestIndex = np.argwhere(modelResult.func_vals == finResult.fun)[0][0]

            # Create a Pandas dataframe to hold the optimiser results.
            resultDataframe = pd.DataFrame(optimizer.Xi, columns = [r"$|\textbf{B}_{\textrm{1}}|$", r"$T_{\textrm{1}}$", r"$\theta_{\textrm{E}}$", 
            r"$\theta_{\textrm{B}_1}$", r"$\phi_{\textrm{B}_1}$", r"$|\textbf{B}_{\textrm{2}}|$", r"$T_{\textrm{2}}$", r"$\theta_{\textrm{B}_2}$", r"$\phi_{\textrm{B}_2}$",
            r"$|\textbf{B}_{\textrm{3}}|$", r"$T_{\textrm{3}}$", r"$\theta_{\textrm{B}_3}$", r"$\phi_{\textrm{B}_3}$", r"$\theta_{\textrm{QWAVL1}}$", r"$\theta_{\textrm{HWAVL}}$", r"$\theta_{\textrm{QWAVL2}}$", 
            r"$\theta_{\textrm{QWAVR1}}$", r"$\theta_{\textrm{HWAVR}}$", r"$\theta_{\textrm{QWAVR2}}$", r"$\theta_{\textrm{PBS}}$", r"$\theta_{\textrm{QWAVF1}}$", r"$\theta_{\textrm{HWAVF}}$", r"$\theta_{\textrm{QWAVF2}}$", 
            r"$\theta_{\textrm{POL}}$"])

            # Save the model in a pickle file.
            joblib.dump((optimizer.models[-1], resultDataframe, bestIndex), "shap_data_all_free.pkl")
            print("Data saved to shap_data_all_free.pkl.")

    return

if __name__ == "__main__":
    # Run the optimisation.
    Optimise(1000, toSave = False)