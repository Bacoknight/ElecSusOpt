"""
This modules optimises the half-Jamin interferometer used to merge beams
to ensure as little loss to the FoM as possible.
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

from matplotlib.patches import Ellipse

import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

# Define some global parameters. NOTE: The two base filters have the same length.
globalDetuning = np.linspace(-25000, 25000, 1000)
baseParamsFilter1 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
baseParamsFilter2 = {"Elem": "Rb", "lcell": 5e-3, "Dline": "D2", "rb85frac": 72.17}
glassIndex = 1.5
airIndex = 1.

def ThreeFilter(inputParams):

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

    # Calculate the losses from the interferometer.
    incidentAngleR = np.deg2rad(inputParams[0])
    sqrtTermR = np.sqrt(1 - np.square((airIndex/glassIndex) * np.sin(incidentAngleR)))
    reflectSR = np.square(np.abs(((airIndex * np.cos(incidentAngleR)) - (glassIndex * sqrtTermR))/((airIndex * np.cos(incidentAngleR)) + (glassIndex * sqrtTermR))))
    reflectPR = np.square(np.abs(((airIndex * sqrtTermR) - (glassIndex * np.cos(incidentAngleR)))/((airIndex * sqrtTermR) + (glassIndex * np.cos(incidentAngleR)))))

    incidentAngleL = np.deg2rad(inputParams[1])
    sqrtTermL = np.sqrt(1 - np.square((airIndex/glassIndex) * np.sin(incidentAngleL)))
    reflectSL = np.square(np.abs(((airIndex * np.cos(incidentAngleL)) - (glassIndex * sqrtTermL))/((airIndex * np.cos(incidentAngleL)) + (glassIndex * sqrtTermL))))
    reflectPL = np.square(np.abs(((airIndex * sqrtTermL) - (glassIndex * np.cos(incidentAngleL)))/((airIndex * sqrtTermL) + (glassIndex * np.cos(incidentAngleL)))))
    transSL = 1 - reflectSL
    transPL = 1 - reflectPL

    inGlassIncidenceL = np.arcsin((glassIndex/airIndex) * np.sin(incidentAngleR))
    if np.isnan(inGlassIncidenceL):
        return 0.0
    
    inGlassSqrtTermL = np.sqrt(1 - np.square((glassIndex/airIndex) * np.sin(inGlassIncidenceL)))
    if np.isnan(inGlassSqrtTermL):
        return 0.0

    inGlassReflectSL = np.square(np.abs(((glassIndex * np.cos(inGlassIncidenceL)) - (airIndex * inGlassSqrtTermL))/((glassIndex * np.cos(inGlassIncidenceL)) + (airIndex * inGlassSqrtTermL))))
    inGlassReflectPL = np.square(np.abs(((glassIndex * inGlassSqrtTermL) - (airIndex * np.cos(inGlassIncidenceL)))/((glassIndex * inGlassSqrtTermL) + (airIndex * np.cos(inGlassIncidenceL)))))
    inGlassTransSL = 1 - inGlassReflectSL
    inGlassTransPL = 1 - inGlassReflectPL

    # Recombine the two fields to form the total output field. This is where the fitness function is naive.
    sPolAng = np.deg2rad(inputParams[2])
    totXROut = reflectSR**0.5 * np.cos(sPolAng)**2 + reflectPR**0.5 * np.sin(sPolAng)*np.cos(sPolAng)
    totYROut = reflectSR**0.5 * np.sin(sPolAng)*np.cos(sPolAng) + reflectPR**0.5 * np.cos(sPolAng)**2
    totXLOut = (inGlassTransSL * transSL)**0.5 * np.cos(sPolAng)**2 + (inGlassTransPL * transPL)**0.5 * np.sin(sPolAng)*np.cos(sPolAng)
    totYLOut = (inGlassTransSL * transSL)**0.5 * np.sin(sPolAng)*np.cos(sPolAng) + (inGlassTransPL * transPL)**0.5 * np.cos(sPolAng)**2
    combinedField = (np.array(np.multiply(np.array([totXROut, totYROut, 0]), outputER.transpose())) + 
    np.array(np.multiply(np.array([totXLOut, totYLOut, 0]), outputEL.transpose()))).T

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
    polariserAnglePreFin = np.deg2rad(175.90283232701586)

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
        problemBounds = [(0., 90.), (0., 90.), (0., 180.)]
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
        print("\nOptimisation for three filters (Jamin combiner with variable s angle) complete. Best FoM: {}, number of evaluations: {}, best parameters: {}".format(abs(finResult.fun), finResult.nfev + len(optimizer.yi), finResult.x))
        
        if toSave:
            print("Updating the model and saving model for SHAP analysis.")
            modelResult = optimizer.tell(finResult.x.tolist(), finResult.fun)
            bestIndex = np.argwhere(modelResult.func_vals == finResult.fun)[0][0]

            # Create a Pandas dataframe to hold the optimiser results.
            resultDataframe = pd.DataFrame(optimizer.Xi, columns = [r"$\theta_{\textrm{i}_\textrm{R}}$", r"$\theta_{\textrm{i}_\textrm{L}}$", r"$\theta_{\textrm{s}}$"])

            # Save the model in a pickle file.
            joblib.dump((optimizer.models[-1], resultDataframe, bestIndex), "shap_data_jamin_rots.pkl")
            print("Data saved to shap_data_jamin_rots.pkl.")

    return

if __name__ == "__main__":
    # Test the function.
    #print(ThreeFilter([0., 0.]))

    # Run the optimisation.
    Optimise(3000, toSave = True)