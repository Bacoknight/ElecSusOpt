"""
Similar to three_split_rotate but the waveplate angle is free.
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
    - Half waveplate angle
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

    # Determine which point has the most viable candidate for the peak location.
    peakPolAng = filterRParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    peakPolMatrix = np.matrix([[np.cos(peakPolAng)**2, np.sin(peakPolAng)*np.cos(peakPolAng), 0],
								[np.sin(peakPolAng)*np.cos(peakPolAng), np.sin(peakPolAng)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    polarisedPeak = np.array(peakPolMatrix * outputER)

    # Get the transmission.
    peakArg = np.argmax((polarisedPeak * polarisedPeak.conjugate()).sum(axis=0).real)
    #print("Index of potential peak: {}".format(peakArg))

    # Recombine the two fields using a PBS at a specific angle so as to combine as much light as possible.
    # First determine the linear wave representing the semimajor axis.
    peakFieldL = outputEL[:, peakArg]
    sqrtConjLenL = np.sqrt((np.array(peakFieldL).conjugate() * np.array(peakFieldL).conjugate()).sum())
    semimajorLenL = np.multiply(np.divide(1, np.linalg.norm(sqrtConjLenL)), np.multiply(sqrtConjLenL, peakFieldL).real)
    #print("Semimajor Length EL: {}".format(np.linalg.norm(semimajorLenL)))
    semiminorLenL = np.multiply(np.divide(1, np.abs(sqrtConjLenL)), np.multiply(sqrtConjLenL, peakFieldL).imag)
    #print("Semiminor Length EL: {}".format(np.linalg.norm(semiminorLenL)))
    eccentricityL = (1 - (np.linalg.norm(semiminorLenL)/np.linalg.norm(semimajorLenL))**2)**0.5
    #print("Eccentricity EL: {}".format(eccentricityL))
    # Calculate the angle of the semimajor axis to the x axis.
    semimajorAngL = np.arctan(np.divide(semimajorLenL[1], semimajorLenL[0]))
    #print("Angle of semimajor axis to x axis EL: {}".format(np.rad2deg(semimajorAngL)))

    peakField = outputER[:, peakArg]
    sqrtConjLen = np.sqrt((np.array(peakField).conjugate() * np.array(peakField).conjugate()).sum())
    semimajorLen = np.multiply(np.divide(1, np.linalg.norm(sqrtConjLen)), np.multiply(sqrtConjLen, peakField).real)
    #print("Semimajor Length ER: {}".format(np.linalg.norm(semimajorLen)))
    semiminorLen = np.multiply(np.divide(1, np.linalg.norm(sqrtConjLen)), np.multiply(sqrtConjLen, peakField).imag)
    #print("Semiminor Length ER: {}".format(np.linalg.norm(semiminorLen)))
    eccentricity = (1 - (np.linalg.norm(semiminorLen)/np.linalg.norm(semimajorLen))**2)**0.5
    #print("Eccentricity ER: {}".format(eccentricity))
    # Calculate the angle of the semimajor axis to the x axis.
    semimajorAng = np.arctan(np.divide(semimajorLen[1], semimajorLen[0]))
    #print("Angle of semimajor axis to x axis ER: {}".format(np.rad2deg(semimajorAng)))
    waveplateAng = inputParams[13]
    #print("Waveplate angle: {}".format(np.rad2deg(waveplateAng)))

    # Now that we know the angle we want in our waveplate, send the second beam through it.
    # NOTE: The factor of -1j causes the light passing through to change handedness.
    halfWaveplate = -1j * np.matrix([[np.cos(waveplateAng), np.sin(waveplateAng), 0],
								[np.sin(waveplateAng), -1 * np.cos(waveplateAng), 0],
                                [0, 0, 1]])

    orthoWave = np.array(halfWaveplate * np.array(outputEL)[:])
    #print("Wave sent through half waveplate: {}".format(outputEL[:, peakArg]))
    #plt.plot(globalDetuning, (outputEL * outputEL.conjugate()).sum(axis = 0).real)
    #plt.show()
    #print("Emerged wave: {}".format(orthoWave[:, peakArg]))

    peakFieldOrth = orthoWave[:, peakArg]
    sqrtConjLenOrth = np.sqrt((np.array(peakFieldOrth).conjugate() * np.array(peakFieldOrth).conjugate()).sum())
    semimajorLenOrth = np.multiply(np.divide(1, np.linalg.norm(sqrtConjLenOrth)), np.multiply(sqrtConjLenOrth, peakFieldOrth).real)
    #print("Semimajor Length Orth: {}".format(np.linalg.norm(semimajorLenOrth)))
    semiminorLenOrth = np.multiply(np.divide(1, np.abs(sqrtConjLenOrth)), np.multiply(sqrtConjLenOrth, peakFieldOrth).imag)
    #print("Semiminor Length Orth: {}".format(np.linalg.norm(semiminorLenOrth)))
    eccentricityOrth = (1 - (np.linalg.norm(semiminorLenOrth)/np.linalg.norm(semimajorLenOrth))**2)**0.5
    #print("Eccentricity Orth: {}".format(eccentricityOrth))
    # Calculate the angle of the semimajor axis to the x axis.
    semimajorAngOrth = np.arctan(np.divide(semimajorLenOrth[1], semimajorLenOrth[0]))
    #print("Angle of semimajor axis to x axis Orth: {}".format(np.rad2deg(semimajorAngOrth)))

    # Now pass both beams through a PBS. The PBS is aligned such that the semimajor axis of light in the first input passes.
    # This will be merged with light along the perpendicular axis of the second input.
    pbsMatrixAccept = np.matrix([[np.cos(semimajorAng)**2, np.sin(semimajorAng)*np.cos(semimajorAng), 0],
								[np.sin(semimajorAng)*np.cos(semimajorAng), np.sin(semimajorAng)**2, 0],
                                [0, 0, 1]])

    acceptedFromInput1 = np.array(pbsMatrixAccept * np.array(outputER)[:])
    rejectedFromInput1 = np.array(outputER) - acceptedFromInput1
    rejectedFromInput2 = np.array(pbsMatrixAccept * np.array(orthoWave))
    acceptedFromInput2 = orthoWave - rejectedFromInput2

    combinedField = acceptedFromInput1 + acceptedFromInput2
    rejectedWave = rejectedFromInput1 + rejectedFromInput2

    peakFieldComb = combinedField[:, peakArg]
    sqrtConjLenComb = np.sqrt((np.array(peakFieldComb).conjugate() * np.array(peakFieldComb).conjugate()).sum())
    semimajorLenComb = np.multiply(np.divide(1, np.linalg.norm(sqrtConjLenComb)), np.multiply(sqrtConjLenComb, peakFieldComb).real)
    #print("Semimajor Length Combined: {}".format(np.linalg.norm(semimajorLenComb)))
    semiminorLenComb = np.multiply(np.divide(1, np.abs(sqrtConjLenComb)), np.multiply(sqrtConjLenComb, peakFieldComb).imag)
    #print("Semiminor Length Combined: {}".format(np.linalg.norm(semiminorLenComb)))
    eccentricityComb = (1 - (np.linalg.norm(semiminorLenComb)/np.linalg.norm(semimajorLenComb))**2)**0.5
    #print("Eccentricity Combined: {}".format(eccentricityComb))
    # Calculate the angle of the semimajor axis to the x axis.
    semimajorAngComb = np.arctan(np.divide(semimajorLenComb[1], semimajorLenComb[0]))
    #print("Angle of semimajor axis to x axis Combined: {}".format(np.rad2deg(semimajorAngComb)))

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
        problemBounds = [(0., 180.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (10., 1300.), (40., 230.), (0., 90.), (0., 90.), (0., 180.)]
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
            joblib.dump((optimizer.models[-1], resultDataframe, bestIndex), "shap_data_free_rotator.pkl")
            print("Data saved to shap_data_free_rotator.pkl.")

    return

if __name__ == "__main__":
    # Run the function.
    # print(ThreeFilter([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    # 143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    # 87.23663536, 90.]))
    # print(ThreeFilter([np.rad2deg(1.2729563270779147), 653.2026710391845, 59.07061955395512, np.rad2deg(0.022582921277005945), np.rad2deg(1.0222099201704498),
    # 103.95733628450398, 194.50243952818656, np.rad2deg(1.2197690515735653), np.rad2deg(1.2151585412088093),
    # 889.5920681456691, 46.09141348824217, np.rad2deg(1.2489169286173634), np.rad2deg(0.239471815289869)]))

    # Run the optimisation.
    OptimiseThreeFilter(500, toSave = False)