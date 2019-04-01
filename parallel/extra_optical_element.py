"""
This module has the two resultant beams added with no further modifications. 
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
from qutip import Bloch

from matplotlib.patches import Ellipse

import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

# Define some global parameters. NOTE: The two base filters have the same length.
globalDetuning = np.sort(np.append(np.linspace(-20000, 20000, 10000), np.linspace(-500, 500, 10000)))
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}

def ThreeFilterPlain(inputParams):
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

    # Recombine the two fields to form the total output field. This is where the fitness function is naive.
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
    finalPolariserAngle = filterRParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    finalPolariser = np.matrix([[np.cos(finalPolariserAngle)**2, np.sin(finalPolariserAngle)*np.cos(finalPolariserAngle), 0],
								[np.sin(finalPolariserAngle)*np.cos(finalPolariserAngle), np.sin(finalPolariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(finalPolariser * outputEFin)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real
    plt.plot(globalDetuning/1e3, filterTransmission, label = "No extra element")

    assert filterTransmission.max() <= 1., "Maximal transmission is greater than 1, ensure your electric fields are correct in magnitude."

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
        return -1.0 * figureOfMerit, outputEFin

def ThreeFilterPol(inputParams):
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
    - Extra polariser angle
    After the filter, the electric fields are combined additively (hence being naive), and pass through a final filter and a polariser
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

    # Recombine the two fields to form the total output field. This is where the fitness function is naive.
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
    polariserAnglePreFin = np.deg2rad(inputParams[13])

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    preFinalPolariser = np.matrix([[np.cos(polariserAnglePreFin)**2, np.sin(polariserAnglePreFin)*np.cos(polariserAnglePreFin), 0],
								[np.sin(polariserAnglePreFin)*np.cos(polariserAnglePreFin), np.sin(polariserAnglePreFin)**2, 0],
                                [0, 0, 1]])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    finalPolariserAngle = filterRParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    finalPolariser = np.matrix([[np.cos(finalPolariserAngle)**2, np.sin(finalPolariserAngle)*np.cos(finalPolariserAngle), 0],
								[np.sin(finalPolariserAngle)*np.cos(finalPolariserAngle), np.sin(finalPolariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(finalPolariser * preFinalPolariser * outputEFin)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real
    plt.plot(globalDetuning/1e3, filterTransmission, label = "Polariser")

    assert filterTransmission.max() <= 1., "Maximal transmission is greater than 1, ensure your electric fields are correct in magnitude."

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
        return -1.0 * figureOfMerit, np.array(preFinalPolariser * outputEFin)

def ThreeFilterHWP(inputParams):
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
    After the filter, the electric fields are combined additively (hence being naive), and pass through a final filter and a polariser
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

    # Recombine the two fields to form the total output field. This is where the fitness function is naive.
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
    waveplateAnglePreFin = np.deg2rad(inputParams[13])

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    preFinalWaveplate = np.exp(-1j * np.pi/2) * np.matrix([[np.cos(waveplateAnglePreFin)**2 - np.sin(waveplateAnglePreFin)**2, 2 * np.cos(waveplateAnglePreFin) * np.sin(waveplateAnglePreFin), 0],
                    [2 * np.cos(waveplateAnglePreFin) * np.sin(waveplateAnglePreFin), (np.sin(waveplateAnglePreFin)**2 - np.cos(waveplateAnglePreFin)**2), 0],
                    [0, 0, 1]])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    finalPolariserAngle = filterRParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    finalPolariser = np.matrix([[np.cos(finalPolariserAngle)**2, np.sin(finalPolariserAngle)*np.cos(finalPolariserAngle), 0],
								[np.sin(finalPolariserAngle)*np.cos(finalPolariserAngle), np.sin(finalPolariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(finalPolariser * preFinalWaveplate * outputEFin)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real
    plt.plot(globalDetuning/1e3, filterTransmission, label = "Half waveplate")

    assert filterTransmission.max() <= 1., "Maximal transmission is greater than 1, ensure your electric fields are correct in magnitude."

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
        return -1.0 * figureOfMerit, np.array(preFinalWaveplate * outputEFin)

def PlotPoincare(targetDetuning):
    """
    For a given detuning, plot the trajectory of the polarisation state after the final filter 
    for light which goes through the various optical elements tested in this module.
    """

    # Obtain the fields for inspection.
    plainFOM, baseDetuning = ThreeFilterPlain([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90.])

    polFOM, polDetuning = ThreeFilterPol([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90., 175.90283232701586])

    hwpFOM, hwpDetuning = ThreeFilterHWP([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90., 86.5557511762424])

    # Extract only target fields.
    detuningIndex = (np.abs(globalDetuning - targetDetuning)).argmin()

    # These are 3-vectors.
    baseField = baseDetuning[:, detuningIndex]
    polField = polDetuning[:, detuningIndex]
    hwpField = hwpDetuning[:, detuningIndex]

    # Get the Stokes parameters for the output linear polarisation.
    polariserAngle = 87.61085044 + 90.
    polarisedField = [np.cos(np.deg2rad(polariserAngle)), np.sin(np.deg2rad(polariserAngle)), 0]
    polariserS0 = polarisedField[0]**2 + polarisedField[1]**2
    polariserS1 = polarisedField[0]**2 - polarisedField[1]**2
    polariserS2 = 2 * (polarisedField[0] * polarisedField[1].conjugate()).real
    polariserS3 = -2 * (polarisedField[0] * polarisedField[1].conjugate()).imag
    polariserRadius = np.sqrt(polariserS1**2 + polariserS2**2 + polariserS3**2)/polariserS0
    polariserPsi = np.arctan(polariserS2/polariserS1)
    polariserChi = np.arctan(polariserS3/np.sqrt(polariserS1**2 + polariserS2**2))

    # Get the Stokes parameters for the base field.
    baseS0 = baseField[0]**2 + baseField[1]**2
    baseS1 = baseField[0]**2 - baseField[1]**2
    baseS2 = 2 * (baseField[0] * baseField[1].conjugate()).real
    baseS3 = -2 * (baseField[0] * baseField[1].conjugate()).imag
    baseRadius = np.sqrt(baseS1**2 + baseS2**2 + baseS3**2)/baseS0
    basePsi = np.arctan(baseS2/baseS1)
    baseChi = np.arctan(baseS3/np.sqrt(baseS1**2 + baseS2**2))

    # Get the Stokes parameters for the pol field.
    polS0 = polField[0]**2 + polField[1]**2
    polS1 = polField[0]**2 - polField[1]**2
    polS2 = 2 * (polField[0] * polField[1].conjugate()).real
    polS3 = -2 * (polField[0] * polField[1].conjugate()).imag
    polRadius = np.sqrt(polS1**2 + polS2**2 + polS3**2)/polS0
    polPsi = np.arctan(polS2/polS1)
    polChi = np.arctan(polS3/np.sqrt(polS1**2 + polS2**2))

    # Get the Stokes parameters for the hwp field.
    hwpS0 = hwpField[0]**2 + hwpField[1]**2
    hwpS1 = hwpField[0]**2 - hwpField[1]**2
    hwpS2 = 2 * (hwpField[0] * hwpField[1].conjugate()).real
    hwpS3 = -2 * (hwpField[0] * hwpField[1].conjugate()).imag
    hwpRadius = np.sqrt(hwpS1**2 + hwpS2**2 + hwpS3**2)/hwpS0
    hwpPsi = np.arctan(hwpS2/hwpS1)
    hwpChi = np.arctan(hwpS3/np.sqrt(hwpS1**2 + hwpS2**2))

    pSphere = Bloch()
    pSphere.point_color = ["b", "#CC6600", "g"]
    # Plot the linear polarisation on the Poincare sphere.
    pSphere.add_vectors(np.multiply(polariserS0 * polariserRadius, [np.cos(polariserPsi) * np.cos(polariserChi), np.sin(polariserPsi) * np.cos(polariserChi), np.sin(polariserChi)]))

    # Plot the base polarisation on the Poincare sphere.
    pSphere.add_points(np.multiply(baseS0 * baseRadius, [np.cos(basePsi) * np.cos(baseChi), np.sin(basePsi) * np.cos(baseChi), np.sin(baseChi)]))

    # Plot the pol polarisation on the Poincare sphere.
    pSphere.add_points(np.multiply(polS0 * polRadius, [np.cos(polPsi) * np.cos(polChi), np.sin(polPsi) * np.cos(polChi), np.sin(polChi)]))

    # Plot the hwp polarisation on the Poincare sphere.
    pSphere.add_points(np.multiply(hwpS0 * hwpRadius, [np.cos(hwpPsi) * np.cos(hwpChi), np.sin(hwpPsi) * np.cos(hwpChi), np.sin(hwpChi)]))

    pSphere.xlabel = [r"$\textrm{S}_{\textrm{1}}$", ""]
    pSphere.ylabel = [r"$\textrm{S}_{\textrm{2}}$", ""]
    pSphere.zlabel = [r"$\textrm{S}_{\textrm{3}}$", ""]
    pSphere.frame_alpha = 0.
    pSphere.show()

    return

def OptimiseHWP(numItersPerCore):
    """
    Optimises the HWP.
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numCores = comm.Get_size()

    if rank == 0:
        # Set up the problem.
        problemBounds = [(0., 180.)]
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

        nextVal = ThreeFilterHWP(np.append([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90.], nextPoint))

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
        print("\nOptimisation for HWP complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(result.fun), len(result.func_vals), result.x))
        
    return

if __name__ == "__main__":
    # Produce the output with no extra element.
    ThreeFilterPlain([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90.])

    # Produce the output with an extra polariser.
    ThreeFilterPol([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90., 175.90283232701586])

    # Produce the output with a half waveplate.
    ThreeFilterHWP([87.61085044, 343.66864345, 76.11772531, 5.09080708, 42.19671567, 
    143.9819049, 129.29791277, 82.58289292, 1.73454687, 280.37227475, 114.23252712,
    87.23663536, 90., 175.90283232701586])

    # Show the plot.
    plt.legend()
    plt.tight_layout()
    plt.show()

    PlotPoincare(-4542.3)

    #OptimiseHWP(1000)