"""
This module investigates the three split filter setup in more depth.
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
plt.rc("text", usetex = True)

# Define some global parameters. NOTE: The two base filters have the same length in this version.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def ThreeFilterNaive(inputParams):
    """
    This fitness function has two independent beams which go through a single filter each. It is
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
    - B field 3
    - Temp 3
    - B theta 3
    - B phi 3
    After the filter, the electric fields are combined additively (hence being naive), and pass through a final filter and a polariser
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

    # Recombine the two fields to form the total output field. This is where the fitness function is naive.
    combinedField = np.array(outputER) + np.array(outputEL)
    # testTransmission = (combinedField * combinedField.conjugate()).sum(axis=0).real

    # plt.plot(globalDetuning, testTransmission)
    # plt.show()

    # Pass the combined field through a final filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEFin] = elecsus.calculate(globalDetuning, combinedField, filterFinParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the final filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterFinParams))
        print("Input field: " + str(combinedField))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filterRParams["Etheta"] + np.pi/2
    #polariserAngle = inputParams[0]

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngleFin = inputParams[1]

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrixFin = np.matrix([[np.cos(polariserAngleFin)**2, np.sin(polariserAngleFin)*np.cos(polariserAngleFin), 0],
								[np.sin(polariserAngleFin)*np.cos(polariserAngleFin), np.sin(polariserAngleFin)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * jonesMatrixFin * outputEFin)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    # plt.plot(globalDetuning, filterTransmission)
    # plt.show()

    #assert filterTransmission.max() <= 1., "Maximal transmission is greater than 1, ensure your electric fields are correct in magnitude."

    ENBW = ((integrate(filterTransmission, globalDetuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real

    # print(outputER[:, 504])
    # print(outputEL[:, 504])

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

def OptimiseThreeFilterNaive(numItersPerCore, toSave = False):
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
        problemBounds = [(0., 180.), (0., 180.)]
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

        nextVal = ThreeFilterNaive(nextPoint)

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
        finResult = opt.minimize(ThreeFilterNaive, optimizer.Xi[np.argmin(optimizer.yi)], method = 'L-BFGS-B', bounds = problemBounds, options = {"disp": True})
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
            joblib.dump((optimizer.models[-1], resultDataframe, bestIndex), "shap_data_three_filter_split.pkl")
            print("Data saved to shap_data_three_filter_split.pkl.")

    return

def ThreeFilterPoincare(inputParams):
    """
    For a given input, plots the polarisations of the output on the Poincare sphere.
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

    # Both filters have the same input field.
    inputE = np.array([np.cos(filterRParams["Etheta"]), np.sin(filterRParams["Etheta"]), 0])/np.sqrt(2)

    # Put each field through their own filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [S0R, S1R, S2R, S3R, outputER] = elecsus.calculate(globalDetuning, inputE, filterRParams, outputs = ["S0", "S1", "S2", "S3", "E_out"])
    except:
        print("There was an issue in ElecSus for the right filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterRParams))
        print("Input field: " + str(inputE))
        return 0.0

    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [S0L, S1L, S2L, S3L, outputEL] = elecsus.calculate(globalDetuning, inputE, filterLParams, outputs = ["S0", "S1", "S2", "S3", "E_out"])
    except:
        print("There was an issue in ElecSus for the left filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterLParams))
        print("Input field: " + str(inputE))
        return 0.0

    # Recombine the two fields to form the total output field.
    combinedField = np.array(outputER) + np.array(outputEL)

    # Pass the combined field through a final filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputEFin] = elecsus.calculate(globalDetuning, combinedField, filterFinParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the final filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filterFinParams))
        print("Input field: " + str(combinedField))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filterRParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * outputEFin)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real
    plt.plot(globalDetuning, filterTransmission)
    plt.show()

    # Generate the required angles for plotting on the Poincare sphere. Psi is the angle in the S1-S2 plane, whilst zeta is the azimuthal angle.
    # NOTE: This generates lists for the angles of each detuning.
    psiRight = np.arctan(np.divide(S2R, S1R))
    zetaRight = np.arctan(np.divide(S3R, np.sqrt(np.square(S1R) + np.square(S2R))))
    psiLeft = np.arctan(np.divide(S2L, S1L))
    zetaLeft = np.arctan(np.divide(S3L, np.sqrt(np.square(S1L) + np.square(S2L))))

    # Create an empty Poincare sphere.
    poincareSphere = Bloch()

    # Add the right filter Poincare trajectory. Currently assume uniform polarisation magnitude.
    rightPoints = [np.multiply(np.cos(psiRight), np.cos(zetaRight)), np.multiply(np.sin(psiRight), np.cos(zetaRight)), np.sin(zetaRight)]
    poincareSphere.add_points(rightPoints, meth = "l")

    # Now add the left filter.
    leftPoints = [np.multiply(np.cos(psiLeft), np.cos(zetaLeft)), np.multiply(np.sin(psiLeft), np.cos(zetaLeft)), np.sin(zetaLeft)]
    poincareSphere.add_points(leftPoints, meth = "l")

    # Also plot two individual points which show the most important points - the polarisation at peak detuning for both filters.
    # The peak is taken from the right filter, which has the output most similar to the one we want.
    peakDetuning = np.argmax(filterTransmission)
    print(S1R[peakDetuning], S1L[peakDetuning])
    print(S2R[peakDetuning], S2L[peakDetuning])
    print(S3R[peakDetuning], S3L[peakDetuning])
    print(psiRight[peakDetuning], psiLeft[peakDetuning])
    print(zetaRight[peakDetuning], zetaLeft[peakDetuning])

    poincareSphere.add_points([np.multiply(np.cos(psiLeft[peakDetuning]), np.cos(zetaLeft[peakDetuning])), np.multiply(np.sin(psiLeft[peakDetuning]), np.cos(zetaLeft[peakDetuning])), np.sin(zetaLeft[peakDetuning])])
    poincareSphere.add_points([np.multiply(np.cos(psiRight[peakDetuning]), np.cos(zetaRight[peakDetuning])), np.multiply(np.sin(psiRight[peakDetuning]), np.cos(zetaRight[peakDetuning])), np.sin(zetaRight[peakDetuning])])

    poincareSphere.show()

    plt.plot(globalDetuning, S0R)
    plt.plot(globalDetuning, S0L)
    plt.plot(globalDetuning, filterTransmission)

    plt.show()

    return

if __name__ == "__main__":
    # Test the realistic setup on the best result found so far.
    # print(ThreeFilterNaive([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    # 143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    # 87.23663536, 90.]))

    # Run the optimisation algorithm.
    OptimiseThreeFilterNaive(100)

    #print(ThreeFilterNaive([47.05211408, 166.43263494]))

    # Plot the Poincare trajectories.
    # print(ThreeFilterPoincare([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    # 143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    # 87.23663536, 90.]))