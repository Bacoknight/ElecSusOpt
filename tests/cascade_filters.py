"""
A module which will toy with the idea of two (or more) filters. There can be a polariser in between filters or not.
We will then attempt to optimise it for figure of merit.
"""

import time
from tqdm import tqdm
from elecsus import elecsus_methods as elecsus
import chocolate as choco
import numpy as np
from scipy.integrate import simps as integrate
import matplotlib.pyplot as plt
import lmfit

# Here we define some global variables so that it is easier to change it for all functions.
# Detuning used for all tests.
globalDetuning = np.arange(-100000, 100000, 10) # MHz
# Input parameters used for all tests.
globalParams = {'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}
p1 = {'Bfield':270, 'rb85frac':72.17, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':86.7, 'Dline':'D2', 'Elem':'Rb'}
p2 = {'Bfield':240, 'rb85frac':72.17, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(6), 'lcell':50e-3, 'T':79, 'Dline':'D2', 'Elem':'Rb'}

def FilterField(detuning, params, isPolarised, inputE = None):
    """
    Generate the electric field after passing through a filter.
    We always assume that the polariser after the filter is perpendiucular to the input
    angle of the light. The polariser is not always required though.
    TODO: Should this be normalised?
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

def TwoFilters(detuning, params1, toPolarise, params2):
    """
    Cascade two filters with either a polariser in between or not, returning the figure of merit.
    Note that the last filter must be followed by a polariser.
    """

    outputField1 = FilterField(detuning, params1, toPolarise)
    outputField2 = FilterField(detuning, params2, True, outputField1)

    outputTrans = Transmission(outputField2)

    return CalculateFoM(outputTrans, detuning)

if __name__ == "__main__":

    print(TwoFilters(globalDetuning, p1, False, p2))