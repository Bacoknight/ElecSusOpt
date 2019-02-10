"""
Introduces complex intermediate optical elements through Jones matrices. Waveplates change the polarisation of the light
without attenuation.
"""

import numpy as np
from elecsus import elecsus_methods as elecsus
from scipy.integrate import simps as integrate

# Define global parameters.
globalDetuning = np.linspace(-25000, 25000, 1000)
globalParams = {'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}

def WaveplateTest():
    """
    Introduces the waveplate for a single filter, for a phase difference phaseDiff before using the polariser which is perpendicular to the input polarisation.
    The phase difference is set to 2*pi so there should be no effect. We also assume a linear retarder, to simplify the Jones matrix.
    """

    elecsusParams = globalParams.copy()

    # First generate the output transmission as before.
    inputE = np.array([np.cos(elecsusParams["Etheta"]), np.sin(elecsusParams["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE] = elecsus.calculate(globalDetuning, inputE, elecsusParams, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(elecsusParams))
        print("Input field: " + str(inputE))
        return 0.0
    
    # Apply the waveplate action.
    phaseDiff = np.deg2rad(360)

    waveplateMatrix = np.exp(-1j * phaseDiff/2) * np.matrix([[np.cos(1.2 * np.pi)**2 + (np.exp(-1j*phaseDiff) * np.sin(1.2 * np.pi)**2), (1 - np.exp(-1j*phaseDiff)) * np.cos(1.2 * np.pi) * np.sin(1.2 * np.pi), 0],
                    [(1 - np.exp(-1j*phaseDiff)) * np.cos(1.2 * np.pi) * np.sin(1.2 * np.pi), np.sin(1.2 * np.pi)**2 + (np.exp(-1j*phaseDiff) * np.cos(1.2 * np.pi)**2), 0],
                    [0, 0, 1]])

    waveplateOutput = np.array(waveplateMatrix * outputE)
    
    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = elecsusParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    singleFilterOutputE = np.array(jonesMatrix * waveplateOutput)

    # Get the transmission.
    singleFilterTransmission = (singleFilterOutputE * singleFilterOutputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(singleFilterTransmission, globalDetuning)/singleFilterTransmission.max().real)/1e3).real

    figureOfMerit = (singleFilterTransmission.max()/ENBW).real
    
    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print(str(elecsusParams))
        return 0.0
    else:
        return -1.0 * figureOfMerit

if __name__ == "__main__":
    print(WaveplateTest())