"""
This module will run the Bayesian Optimisation algorithm on multiple cores, using the MPI framework.
"""

import time
from tqdm import tqdm
from elecsus import elecsus_methods as elecsus
import chocolate as choco
import numpy as np
from scipy.integrate import simps as integrate
import matplotlib.pyplot as plt
from mpi4py import MPI

# First we define the functions we will use later on.

# Here we define some global variables so that it is easier to change it for all functions.
# Detuning used for all tests.
globalDetuning = np.arange(-100000, 100000, 10) # MHz
# Input parameters used for all tests.
globalParams = {'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}
# Global target FoM. This is used to test for time how long it takes to reproduce a paper value.
globalFoM = 1.04

def ProduceSpectrum(detuning, params, toPlot=False):
        """
        Produce a simple transmission output using ElecSus.
        We always assume that the polariser after the filter is perpendiucular to the input
        angle of the light.
        """

        # Use the input of the function to determine the polarisation of the input light.
        E_in = np.array([np.cos(params["Etheta"]), np.sin(params["Etheta"]), 0])

        # Determine the effect of the final polariser on the output field using a Jones matrix.
        outputAngle = params['Etheta'] + np.pi/2
        J_out = np.matrix([[np.cos(outputAngle)**2, np.sin(outputAngle)*np.cos(outputAngle)],
                        [np.sin(outputAngle)*np.cos(outputAngle), np.sin(outputAngle)**2]])

        # Call ElecSus to find the output electric field from the cell.
        try:
	        [E_out] = elecsus.calculate(detuning, E_in, params, outputs=['E_out'])

        except:
            # There was an issue obtaining the field from ElecSus.
	        return 0

        transmittedE =  np.array(J_out * E_out[:2])
        transmission =  (transmittedE * transmittedE.conjugate()).sum(axis=0)

        if toPlot:
                # Plot the result.
                plt.plot(detuning, transmission)
                plt.show()

        return transmission

def CalculateFoM(detuning, params):

        # Get the overall transmission.
        transmission = ProduceSpectrum(detuning, params, False)

        maxTransmission = np.max(transmission)
        ENBW = integrate(transmission, detuning)/maxTransmission
        FOM = maxTransmission/ENBW # This is in 1/MHz, so we multiply by 1000 for 1/GHz

        if np.isnan(FOM):
                # Occurs if there is just a flat line for the transmission. Usually occurs for high temp and high B field.
                return 0
        else:
                return FOM.real * 1000

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
                print("FoM for " + str(elecsusParams["Elem"]) + ", " + str(elecsusParams["Dline"]) + " line: " + str(CalculateFoM(globalDetuning, elecsusParams)))

        return

def Objective1D(bField):
    """
    First variable to vary is the magnetic field strength.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField

    return CalculateFoM(globalDetuning, elecsusParams)

def Objective2D(bField, temperature):
    """
    Second variable to vary is the temperature.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField
    elecsusParams["T"] = temperature

    return CalculateFoM(globalDetuning, elecsusParams)

def Objective3D(bField, temperature, bTheta):
    """
    Third variable to vary is the angle the magnetic field makes with the wavevector.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField
    elecsusParams["T"] = temperature
    elecsusParams["Btheta"] = np.deg2rad(bTheta)

    return CalculateFoM(globalDetuning, elecsusParams)

def Objective4D(bField, temperature, bTheta, eTheta):
    """
    Fourth variable to vary is the angle the electric field makes with the wavevector.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField
    elecsusParams["T"] = temperature
    elecsusParams["Btheta"] = np.deg2rad(bTheta)
    elecsusParams["Etheta"] = np.deg2rad(eTheta)

    return CalculateFoM(globalDetuning, elecsusParams)

def Objective5D(bField, temperature, bTheta, eTheta, bPhi):
    """
    The final variable for a single filter is the angle the magnetic field makes with the electric field.
    """

    elecsusParams = globalParams
    elecsusParams["Bfield"] = bField
    elecsusParams["T"] = temperature
    elecsusParams["Btheta"] = np.deg2rad(bTheta)
    elecsusParams["Etheta"] = np.deg2rad(eTheta)
    elecsusParams["Bphi"] = np.deg2rad(bPhi)

    return CalculateFoM(globalDetuning, elecsusParams)

def BayesMPI(dimension = 1, numItersPerCore = 100):
        """
        The main function of this module. This will run the Bayesian optimisation algorithm on as many cores as specified, using the MPI framework.
        The Chocolate library will ensure concurrency as all cores use a shared database to ensure no overlap of function evaluations.
        """

        # Find all the cores being used.
        comm = MPI.COMM_WORLD
    
        # Give each core an ID, called rank.
        rank = comm.Get_rank()

        # Define a dictionary to store each possible case for dimensions. Each entry is the objective function and the respective bounds.
        problemList = {
                1: [Objective1D, {"bField": choco.uniform(0, 2000)}],
                2: [Objective2D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400)}],
                3: [Objective3D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 90)}],
                4: [Objective4D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 90), "eTheta": choco.uniform(0, 90)}],
                5: [Objective5D, {"bField": choco.uniform(0, 2000), "temperature": choco.uniform(20, 400), "bTheta": choco.uniform(0, 90), "eTheta": choco.uniform(0, 90), "bPhi": choco.uniform(0, 90)}]
        }

        problem = problemList.get(dimension)

        # Now have all the cores run the optimisation algorithm.
        bestFoM = 0

        # Set up the database for the chocolate optimiser.
        connection = choco.SQLiteConnection("sqlite:///bayesian_mpi_db.db")

        # Define the optimiser. Clear the database only once (so we start from fresh each iteration.)
        if rank == 0:
                solver = choco.Bayes(connection, problem[1], utility_function = "ei", n_bootstrap = int(numItersPerCore/10), clear_db = True)
        else: 
                solver = choco.Bayes(connection, problem[1], utility_function = "ei", n_bootstrap = int(numItersPerCore/10))
        
        for iteration in range(numItersPerCore):

                # Suggest a point of inquiry.
                token, nextParams = solver.next()

                # Check what FoM this gives.
                fEval = abs(problem[0].__call__(**nextParams))

                # Update the solver. This happens ASAP to ensure other cores have the most up to date info.
                solver.update(token, fEval)

                # Update best FoM.
                if fEval > bestFoM:
                        bestFoM = fEval

        print("Core number " + str(rank) + " best FoM: " + str(bestFoM))

        return
                
        
if __name__ == "__main__":
        BayesMPI(5, 6)