"""
A module which will toy with the idea of two filters. There can be a polariser in between filters or not.
We will then attempt to optimise it for figure of merit.
"""

import time
from elecsus import elecsus_methods as elecsus
import chocolate as choco
import numpy as np
from scipy.integrate import simps as integrate
import matplotlib.pyplot as plt
import lmfit
from mpi4py import MPI
import pandas as pd
import itertools
import seaborn as sb

# Here we define some global variables so that it is easier to change it for all functions.
# Detuning used for all tests.
globalDetuning = np.arange(-100000, 100000, 10) # MHz
# Input parameters used for all tests.
globalParams = {'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}
p1 = {'Bfield':349.824, 'rb85frac':0, 'Btheta':np.deg2rad(41.557692), 'Etheta':np.deg2rad(69.192077), 'lcell':5e-3, 'T':46.564327, 'Dline':'D2', 'Elem':'Rb'}
p2 = {'Bfield':194.799786, 'rb85frac':0, 'Btheta':np.deg2rad(89.511744), 'Etheta':np.deg2rad(87645), 'lcell':50e-3, 'T':51.071226, 'Dline':'D2', 'Elem':'Rb'}
twoFiltersParams = {'abundance1': 72.17, 'abundance2': 100.0, 'bField1': 1113.728, 'bField2': 794.5498899268161, 'polariser2Angle': 105.1686751032191,
'eTheta1': 33.87852387605819, 'polariser1Angle': 110.94181803271967, 'bPhi1': 143.12956765248276, 'bPhi2': 62.05875144427716, 'temperature2': 23.684958557108178,
'polariserPresent': False, 'temperature1': 89.2117061563239, 'bTheta1': 36.087929981681256, 'bTheta2': 90.09185012392477}
paperParams = {'abundance1': 72.17, 'abundance2': 72.17, 'bField1': 270, 'bField2': 240, 'polariser2Angle': 105.1686751032191,
'eTheta1': 6, 'polariser1Angle': 110.94181803271967, 'bPhi1': 0, 'bPhi2': 0, 'temperature2': 79.0,
'polariserPresent': False, 'temperature1': 86.7, 'bTheta1': 0, 'bTheta2': 90}


# Rubidium abundances which are realisable.
isotopeRbList = [0.0, 72.17, 100.0]

def FilterField(detuning, params, isPolarised, inputE = None):
    """
    Generate the electric field after passing through a filter.
    We always assume that the polariser after the filter is perpendiucular to the input
    angle of the light. The polariser is not always required though.
    TODO: Should this be normalised?
    DEPRECATED.
    """

    if inputE is None:
        # Use the input of the function to determine the polarisation of the input light.
        E_in = np.array([np.cos(params["Etheta"]), np.sin(params["Etheta"]), 0])
    else:
        assert (inputE.shape == (3, len(detuning))), "Incorrect electric field input shape. Are you missing the z-axis?"
        E_in = inputE

    # Call ElecSus to find the output electric field from the cell.
    try:
	    [E_out] = elecsus.calculate(detuning, E_in, params, outputs = ['E_out'])

    except:
        # There was an issue obtaining the field from ElecSus.
	    return 0

    if isPolarised:

        # We must consider the effect of the polariser, which we assume is perpendicular to the input.
        outputAngle = params['Etheta'] + np.pi/2
        J_out = np.matrix([[np.cos(outputAngle)**2, np.sin(outputAngle)*np.cos(outputAngle), 0],
                [np.sin(outputAngle)*np.cos(outputAngle), np.sin(outputAngle)**2, 0],
                [0, 0, 1]])
            
        outputE =  np.array(J_out * E_out)

    else:
        # No polariser needed.
        outputE = E_out

    return outputE

def TwoFiltersPolarised(detuning, params1, polariser1Angle, params2, polariser2Angle):
    """
    This function calculates the figure of merit for a two filter setup, with a polariser in between the filters.
    Note that the angle of the central polariser is arbitrary, but the final polariser must be perpendicular to it.
    This requirement is waived in the case of the central polariser being perpendicular to the first, as off-resonant
    light will have already been filtered out.
    """

    # Input into first filter.
    inputAngle = params1["Etheta"]
    E_in1 = np.array([np.cos(inputAngle), np.sin(inputAngle), 0])

    # Find the output field from ElecSus.
    try:
	    [E_out1] = elecsus.calculate(detuning, E_in1, params1, outputs = ['E_out'])

    except:
        # There was an issue obtaining the field from ElecSus.
        print("There was an issue in ElecSus. Here are the input parameters:")
        print("Input parameters: " + str(params1))
        print("Input field: " + str(E_in1))
        return 0

    # Apply the effect of the central polariser. This matrix is 3x3 to be the correct input for ElecSus
    J_out1 = np.matrix([[np.cos(polariser1Angle)**2, np.sin(polariser1Angle)*np.cos(polariser1Angle), 0],
            [np.sin(polariser1Angle)*np.cos(polariser1Angle), np.sin(polariser1Angle)**2, 0],
            [0, 0, 1]])

    # This is the input into the second filter.
    E_in2 = np.array(J_out1 * E_out1)

    # Find the output field from ElecSus.
    try:
	    [E_out2] = elecsus.calculate(detuning, E_in2, params2, outputs = ['E_out'])

    except:
        # There was an issue obtaining the field from ElecSus.
        print("There was an issue in ElecSus. Here are the input parameters:")
        print("Input parameters: " + str(params2))
        print("Input field: " + str(E_in2))
        return 0

    # Calculate the output of the last polariser.
    if (polariser1Angle == (inputAngle + np.pi/2)):
        # Off-resonant light has already been filtered out.
        outputAngle = polariser2Angle
    else:
        # The final polariser must be crossed to prevent the escape of off-resonant light.
        outputAngle = polariser1Angle + np.pi/2

    J_out2 = np.matrix([[np.cos(outputAngle)**2, np.sin(outputAngle)*np.cos(outputAngle), 0],
            [np.sin(outputAngle)*np.cos(outputAngle), np.sin(outputAngle)**2, 0],
            [0, 0, 1]])

    outputE = np.array(J_out2 * E_out2)

    # Obtain output transmission.
    outputTrans = Transmission(outputE)

    return CalculateFoM(outputTrans, detuning)

def TwoFiltersNonPolarised(detuning, params1, params2):
    """
    This function calculates the figure of merit for a two filter setup, without a polariser in between the filters.
    """

    # Input into first filter.
    E_in1 = np.array([np.cos(params1["Etheta"]), np.sin(params1["Etheta"]), 0])

    # Find the output field from ElecSus.
    try:
	    [E_out1] = elecsus.calculate(detuning, E_in1, params1, outputs = ['E_out'])

    except:
        # There was an issue obtaining the field from ElecSus.
        print("There was an issue in ElecSus. Here are the input parameters:")
        print("Input parameters: " + str(params1))
        print("Input field: " + str(E_in1))
        return 0

    # Find the output field from ElecSus.
    try:
	    [E_out2] = elecsus.calculate(detuning, E_out1, params2, outputs = ['E_out'])

    except:
        # There was an issue obtaining the field from ElecSus.
        print("There was an issue in ElecSus. Here are the input parameters:")
        print("Input parameters: " + str(params2))
        print("Input field: " + str(E_out1))
        return 0

    # Calculate the output of the last polariser, which must be perpendicular to the input polarisation.
    outputAngle = params1['Etheta'] + np.pi/2
    J_out = np.matrix([[np.cos(outputAngle)**2, np.sin(outputAngle)*np.cos(outputAngle), 0],
            [np.sin(outputAngle)*np.cos(outputAngle), np.sin(outputAngle)**2, 0],
            [0, 0, 1]])

    outputE = np.array(J_out * E_out2)

    # Obtain output transmission.
    outputTrans = Transmission(outputE)

    return CalculateFoM(outputTrans, detuning)

def Transmission(eField):
    """
    Calculate the transmission for a given range of detuning.
    """

    return (eField * eField.conjugate()).sum(axis=0)

def CalculateFoM(transmission, detuning):
    """
    Calculate the figure of merit (FoM) for the produced spectrum.
    """

    maxTransmission = np.max(transmission)
    ENBW = integrate(transmission, detuning)/maxTransmission
    FOM = maxTransmission/ENBW # This is in 1/MHz, so we multiply by 1000 for 1/GHz

    if np.isnan(FOM):
        # Occurs if there is just a flat line for the transmission. Usually occurs for high temp and high B field.
        return 0
    else:
        return FOM.real * 1000

def SingleFilter(detuning, params):
    """
    Test the validity of this method by having just a single filter.
    """

    return CalculateFoM(Transmission(FilterField(detuning, params, True)), detuning)

def TestFoM():
        """
        This function aims to reproduce the values of the Table 1 in the paper in Opt. Lett. 2018
        Doc ID: 335953.
        """
        # Define the parameters for each test.
        paramsList = [{'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'},
        {'Bfield':695, 'rb85frac':72.17, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(45), 'lcell':5e-3, 'T':147, 'Dline':'D1', 'Elem':'Rb'},
        {'Bfield':224, 'rb85frac':100, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(90), 'lcell':5e-3, 'T':121, 'Dline':'D2', 'Elem':'Rb'},
        {'Bfield':310, 'rb85frac':100, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(46), 'lcell':5e-3, 'T':139, 'Dline':'D1', 'Elem':'Rb'},
        {'Bfield':849, 'rb85frac':0, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(81), 'lcell':5e-3, 'T':121, 'Dline':'D2', 'Elem':'Rb'},
        {'Bfield':315, 'rb85frac':0, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(45), 'lcell':5e-3, 'T':141, 'Dline':'D1', 'Elem':'Rb'},
        {'Bfield':1120, 'Btheta':np.deg2rad(87), 'Etheta':np.deg2rad(89), 'lcell':5e-3, 'T':127, 'Dline':'D2', 'Elem':'Cs'},
        {'Bfield':338, 'Btheta':np.deg2rad(89), 'Etheta':np.deg2rad(46), 'lcell':5e-3, 'T':121, 'Dline':'D1', 'Elem':'Cs'},
        {'Bfield':88, 'Btheta':np.deg2rad(1), 'Etheta':np.deg2rad(87), 'lcell':5e-3, 'T':150, 'Dline':'D2', 'Elem':'K'},
        {'Bfield':460, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(47), 'lcell':5e-3, 'T':177, 'Dline':'D1', 'Elem':'K'},
        {'Bfield':144, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(0), 'lcell':5e-3, 'T':245, 'Dline':'D2', 'Elem':'Na'},
        {'Bfield':945, 'Btheta':np.deg2rad(88), 'Etheta':np.deg2rad(41), 'lcell':5e-3, 'T':279, 'Dline':'D1', 'Elem':'Na'}]

        for elecsusParams in paramsList:
                print("FoM for " + str(elecsusParams["Elem"]) + ", " + str(elecsusParams["Dline"]) + " line: " + str(SingleFilter(globalDetuning, elecsusParams)))

        return

def TwoFilters(bField1, temperature1, bTheta1, eTheta1, bPhi1, abundance1, polariserPresent, polariser1Angle, bField2, temperature2, bTheta2, bPhi2, abundance2, polariser2Angle):
    """
    The objective function to be optimised by the chosen algorithm.
    """

    cell1Params = p1
    cell1Params["Bfield"] = bField1
    cell1Params["T"] = temperature1
    cell1Params["Btheta"] = np.deg2rad(bTheta1)
    cell1Params["Etheta"] = np.deg2rad(eTheta1)
    cell1Params["Bphi"] = np.deg2rad(bPhi1)
    cell1Params["rb85frac"] = abundance1

    cell2Params = p2
    cell2Params["Bfield"] = bField2
    cell2Params["T"] = temperature2
    cell2Params["Btheta"] = np.deg2rad(bTheta2)
    cell2Params["Bphi"] = np.deg2rad(bPhi2)
    cell2Params["rb85frac"] = abundance2

    if polariserPresent:
        return TwoFiltersPolarised(globalDetuning, cell1Params, polariser1Angle, cell2Params, polariser2Angle)
    else:
        return TwoFiltersNonPolarised(globalDetuning, cell1Params, cell2Params)

def FirstFilter(bField1, temperature1, bTheta1, eTheta1, bPhi1, abundance1):
    """
    This is required for determining the individual figure of merit for each filter.
    """

    cell1Params = p1
    cell1Params["Bfield"] = bField1
    cell1Params["T"] = temperature1
    cell1Params["Btheta"] = np.deg2rad(bTheta1)
    cell1Params["Etheta"] = np.deg2rad(eTheta1)
    cell1Params["Bphi"] = np.deg2rad(bPhi1)
    cell1Params["rb85frac"] = abundance1

    return SingleFilter(globalDetuning, cell1Params)

def SecondFilter(eTheta1, polariserPresent, polariser1Angle, bField2, temperature2, bTheta2, bPhi2, abundance2):
    """
    Same as above, but for the second filter. The input angle is a bit stranger here.
    """

    cell2Params = p2
    cell2Params["Bfield"] = bField2
    cell2Params["T"] = temperature2
    cell2Params["Btheta"] = np.deg2rad(bTheta2)

    if polariserPresent:
        cell2Params["Etheta"] = np.deg2rad(polariser1Angle)
    else:
        cell2Params["Etheta"] = np.deg2rad(eTheta1)

    cell2Params["Bphi"] = np.deg2rad(bPhi2)
    cell2Params["rb85frac"] = abundance2

    return SingleFilter(globalDetuning, cell2Params)

def OptimiseTwoFilters(numItersPerCore, toClear = False):
    """
    Uses the chosen optimisation algorithm to determine the best setup for two filters that optimises the figure of merit.
    We will only optimise the full problem. For rubidium the variables are:
    - 2 Temperatures
    - 2 Magnetic Field Strengths
    - 4 Magnetic Field Angles
    - 2 Rubidium Abundances
    - 1 Input Polarisation
    - Whether to have an intermediate polariser
    - If the polariser is present, it's angle
    - The angle of the final polariser (fixed apart from the special case of the central polariser being crossed to the first).
    Cell lengths are fixed at 5mm and 50mm for the first and second filter respectively. They can be free later.
    """

    # Get MPI information.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Define the bounds of each variable.
    variableBounds = {"bField1": choco.uniform(0, 2000), "temperature1": choco.uniform(20, 400), "bTheta1": choco.uniform(0, 180), "eTheta1": choco.uniform(0, 180), "bPhi1": choco.uniform(0, 180),
    "abundance1": choco.choice(isotopeRbList), "polariserPresent": choco.choice([True, False]), "polariser1Angle": choco.uniform(0, 180), "bField2": choco.uniform(0, 2000),
    "temperature2": choco.uniform(20, 400), "bTheta2": choco.uniform(0, 180), "bPhi2": choco.uniform(0, 180), "abundance2": choco.choice(isotopeRbList), "polariser2Angle": choco.uniform(0, 180)}

    # Set up the connection and clear previous entries.
    connection = choco.SQLiteConnection("sqlite:///two_filters_db.db")
    if toClear:
        if rank == 0:
            connection.clear()
    
    optimiser = choco.QuasiRandom(connection, variableBounds, clear_db = True, skip = int(np.ceil(numItersPerCore/10)), seed = None)

    for iteration in range(numItersPerCore):
                
        # Suggest a point of inquiry.
        token, nextParams = optimiser.next()

        # Check what FoM this gives.
        fEval = abs(TwoFilters(**nextParams))

        if fEval > 10:
            print("FoM greater than 10 for these params:")
            print(str(nextParams))

        # Update the solver.
        optimiser.update(token, fEval)

    # Wait for all the processes to finish.
    comm.Barrier()

    # Obtain results
    results = connection.results_as_dataframe()

    if rank == 0:
        results = results.sort_values(by = "_loss", ascending = False)
        print(results)
        print(results.to_dict('records')[0])

    return

def CombinedFoMColourMap(numItersPerCore):
    """
    Determines the individual figures of merit and the resultant combination. Runs the optimisation as above, with a few steps in between.
    """

    # Get MPI information.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Define the bounds of each variable.
    variableBounds = {"bField1": choco.uniform(0, 2000), "temperature1": choco.uniform(20, 400), "bTheta1": choco.uniform(0, 180), "eTheta1": choco.uniform(0, 180), "bPhi1": choco.uniform(0, 180),
    "abundance1": choco.choice(isotopeRbList), "polariserPresent": choco.choice([True, False]), "polariser1Angle": choco.uniform(0, 180), "bField2": choco.uniform(0, 2000),
    "temperature2": choco.uniform(20, 400), "bTheta2": choco.uniform(0, 180), "bPhi2": choco.uniform(0, 180), "abundance2": choco.choice(isotopeRbList), "polariser2Angle": choco.uniform(0, 180)}

    # Set up the connection and clear previous entries.
    connection = choco.SQLiteConnection("sqlite:///colour_map_db.db")
    if rank == 0:
        connection.clear()
    
    optimiser = choco.QuasiRandom(connection, variableBounds, clear_db = True, skip = int(np.ceil(numItersPerCore/10)), seed = None)

    filter1FoM = []
    filter2FoM = []
    totalFoM = []

    for iteration in range(numItersPerCore):
                
        # Suggest a point of inquiry.
        token, nextParams = optimiser.next()

        # Find the individual figures of merit for each filter.
        filter1Params = dict((key, nextParams[key]) for key in ("bField1", "temperature1", "bTheta1", "eTheta1", "bPhi1", "abundance1") if key in nextParams)
        filter1FoM.append(abs(FirstFilter(**filter1Params)))

        filter2Params = dict((key, nextParams[key]) for key in ("eTheta1", "polariserPresent", "polariser1Angle", "bField2", "temperature2", "bTheta2", "bPhi2", "abundance2") if key in nextParams)
        filter2FoM.append(abs(SecondFilter(**filter2Params)))

        # Check what FoM this gives.
        fEval = abs(TwoFilters(**nextParams))

        totalFoM.append(fEval)

        # Update the solver.
        optimiser.update(token, fEval)

    # Send the result to the main thread.
    if rank != 0:
        comm.send([filter1FoM, filter2FoM, totalFoM], dest = 0)

    # Wait for all the processes to finish.
    comm.Barrier()

    if rank == 0:
        # Gather all the data into one list.
        totalF1List = []
        totalF2List = []
        totalFinalList = []

        # Add the main threads contribution.
        totalF1List.append(filter1FoM)
        totalF2List.append(filter2FoM)
        totalFinalList.append(totalFoM)

        for source in range(comm.Get_size()):
            if source != 0:
                resultLists = comm.recv(source = source)
                totalF1List.append(resultLists[0])
                totalF2List.append(resultLists[1])
                totalFinalList.append(resultLists[2])


        # Flatten the lists into one big one.
        totalF1List = list(itertools.chain.from_iterable(totalF1List))
        totalF2List = list(itertools.chain.from_iterable(totalF2List))
        totalFinalList = list(itertools.chain.from_iterable(totalFinalList))

        # Plot the results on a colourmap.
        fig = plt.figure("Total FoM As a Function of Equivalents FoMs")
        fig.set_size_inches(19.20, 10.80)
        plt.tricontourf(totalF1List, totalF2List, totalFinalList, cmap = "viridis")
        # Plot scatter points for high figure of merit points.
        highFoMVal = 0.15
        highFoM1 = []
        highFoM2 = []
        for index, fom1 in enumerate(totalF1List):
            if(fom1 > highFoMVal or totalF2List[index] > highFoMVal):
                highFoM1.append(fom1)
                highFoM2.append(totalF2List[index])

        plt.scatter(highFoM1, highFoM2, s = 100, marker = "x", c = "black")
        plt.xlabel(r"Filter 1 FoM [GHz$^{-1}$]", fontsize = 16)
        plt.ylabel(r"Filter 2 FoM [GHz$^{-1}$]", fontsize = 16)
        
        cb = plt.colorbar()
        cb.set_label(r"Total FoM [GHz$^{-1}$]", rotation = 270, labelpad = 25, fontsize = 16)
        cb.ax.tick_params(labelsize = 16)

        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.xlim(left = 0)
        plt.ylim(bottom = 0)

        plt.savefig("combined_colourmap.pdf")
        print("Plot saved.")

        plt.show()

    return

def ParamSenstivity():
    """
    This function determines the smallest change in each variable which will cause a 1% change in the figure of merit.
    """

    # Define the optimal parameters for the two filter setup.
    optimalParams = {"bField1": 230, "temperature1": 126, "bTheta1": 83, "eTheta1": 6, "bPhi1": 0, "abundance1": 72.17, "polariserPresent": True, "polariser1Angle": 70,
     "bField2": 230, "temperature2": 126, "bTheta2": 83, "bPhi2": 0, "abundance2": 72.17, "polariser2Angle": 0}

    # Evaluate the function.
    optimalFoM = TwoFilters(**optimalParams)
    print("Optimal figure of merit: " + str(optimalFoM))

    # Each perturbation we will apply. NOTE: This currently assume a symmetric peak, add negative values to test that. TODO: Plot graph?
    perturbations = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5, 10, 20]

    for key, variable in optimalParams.items():
        # Ensure we're not varying something that doesn't make sense.
        if key in ["abundance1", "polariserPresent", "abundance2"]:
            continue
        
        sensitivity = 0

        for perturbation in perturbations:
            # Apply the perturbation.
            sensitivity = perturbation
            perturbDict = optimalParams.copy()
            perturbDict[key] = variable + perturbation
            perturbFoM = TwoFilters(**perturbDict)

            if perturbFoM/optimalFoM <= 0.99 or perturbFoM/optimalFoM >= 1.01:
                break

        print("Sensitivity of " + key + ": Change of +" + str(sensitivity) + " for a change of 1%" + " in the figure of merit")

    return

if __name__ == "__main__":
    # Ensure these methods work.
    #TestFoM()
    
    # Ensure the objective function works by reproducing the paper values for the figure of merit.
    #print(TwoFilters(**paperParams))

    # Run optimisation.
    OptimiseTwoFilters(500, toClear = False)

    # Create a colourmap showing total FoM sensitivity to individual FoMs.
    #CombinedFoMColourMap(300)

    # Determine the sensitivity of each variable.
    #ParamSenstivity()

    # Test a given set of parameters.
    #print(TwoFilters(**twoFiltersParams))