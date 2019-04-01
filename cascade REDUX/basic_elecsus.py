"""
This module will prove that I can call ElecSus properly and use it to recreate various graphs and values from the literature.
"""

from elecsus import elecsus_methods as elecsus
import numpy as np
import matplotlib.pyplot as plt
plt.rc("text", usetex = True)

import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")
import time
from scipy.integrate import simps as integrate

# Quick start parameters.
globalParams = {'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}
globalDetuning = np.linspace(-25000, 25000, 20000)

def ProduceTransmissionSingle(detuning, inputParams):
    """
    Create a transmission graph for a given set of filter parameters. This is only for a single filter, so the polarisers are crossed.
    """ 

    # Start timing.
    startTime = time.time()

    # The parameter 'Etheta' is not actually used in ElecSus explicitly. We use it to define the input angle of the electric field.
    # We assume it has no component in the z direction, as that is just a matter of redefining our axes.
    inputE = np.array([np.cos(inputParams["Etheta"]), np.sin(inputParams["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field.
    [outputE] = elecsus.calculate(detuning, inputE, inputParams, outputs = ["E_out"])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = inputParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    singleFilterOutputE = np.array(jonesMatrix * outputE)

    # Get the transmission.
    singleFilterTransmission = (singleFilterOutputE * singleFilterOutputE.conjugate()).sum(axis=0)

    # Print some stats.
    print("ElecSus output complete!")
    print("Time taken (s): " + str(time.time() - startTime))
    print("Maximum transmission: " + str(singleFilterTransmission.max().real))

    # Plot the output.
    fig = plt.figure("Basic transmission")
    fig.set_size_inches(19.20, 10.80)
    plt.plot(detuning/1e3, singleFilterTransmission * 100)

    # Make the graph pretty.
    plt.xlabel(r'$\Delta$ (GHz)')
    plt.ylabel(r'Transmission (\%)')
    plt.ylim(bottom = 0)
    plt.xlim(detuning[0]/1e3, detuning[-1]/1e3)
    plt.tight_layout()

    #plt.show()

    return

def TestTransmission(detuning):
    """
    Reproduce the transmission graphs from Opt. Lett. 43, 4272-4275 (2018).
    """

    # Define the parameters for each test.
    paramsList = [{'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'},
    {'Bfield':1120, 'Btheta':np.deg2rad(87), 'Etheta':np.deg2rad(89), 'lcell':5e-3, 'T':127, 'Dline':'D2', 'Elem':'Cs'},
    {'Bfield':88, 'Btheta':np.deg2rad(1), 'Etheta':np.deg2rad(87), 'lcell':5e-3, 'T':150, 'Dline':'D2', 'Elem':'K'},
    {'Bfield':144, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(0), 'lcell':5e-3, 'T':245, 'Dline':'D2', 'Elem':'Na'}]

    for params in paramsList:
        ProduceTransmissionSingle(detuning, params)

    return

def ProduceENBW(detuning, inputParams):
    """
    Generate the Equivalent Noise Bandwidth for a single filter.
    """

    # First generate the output transmission as before.
    inputE = np.array([np.cos(inputParams["Etheta"]), np.sin(inputParams["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field.
    [outputE] = elecsus.calculate(detuning, inputE, inputParams, outputs = ["E_out"])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = inputParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    singleFilterOutputE = np.array(jonesMatrix * outputE)

    # Get the transmission.
    singleFilterTransmission = (singleFilterOutputE * singleFilterOutputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(singleFilterTransmission, detuning)/singleFilterTransmission.max().real)/1e3).real

    print("ENBW obtained! Value: " + str(ENBW))

    return

def TestENBW(detuning):
    """
    Reproduce the ENBW values from Opt. Lett. 43, 4272-4275 (2018).
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

    for params in paramsList:
        ProduceENBW(detuning, params)

    return

def ProduceFoM(detuning, inputParams):
    """
    Produce a single figure of merit for a single filter.
    """

    # The steps are similar to above, with one extra step.
    # First generate the output transmission as before.
    inputE = np.array([np.cos(inputParams["Etheta"]), np.sin(inputParams["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field.
    [outputE] = elecsus.calculate(detuning, inputE, inputParams, outputs = ["E_out"])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = inputParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    singleFilterOutputE = np.array(jonesMatrix * outputE)

    # Get the transmission.
    singleFilterTransmission = (singleFilterOutputE * singleFilterOutputE.conjugate()).sum(axis=0)

    ENBW = ((integrate(singleFilterTransmission, detuning)/singleFilterTransmission.max().real)/1e3).real

    figureOfMerit = (singleFilterTransmission.max()/ENBW).real

    print("FoM obtained! Value: " + str(figureOfMerit))

    return

def TestFoM(detuning):
    """
    Reproduce the FoM values from Opt. Lett. 43, 4272-4275 (2018).
    """

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

    for params in paramsList:
        ProduceFoM(detuning, params)

    return

def TwoFilterStats(detuning, inputParams1, inputParams2):
    """
    Generates all the stats for the double filter proposed in the literature. That is, a transmission graph, the ENBW and FoM.
    """

    # First generate the output transmission for the first filter.
    inputE = np.array([np.cos(inputParams1["Etheta"]), np.sin(inputParams1["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field from the first filter.
    [outputE1] = elecsus.calculate(detuning, inputE, inputParams1, outputs = ["E_out"])

    # Call ElecSus to obtain the output field from the second filter.
    [outputE2] = elecsus.calculate(detuning, outputE1, inputParams2, outputs = ["E_out"])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = inputParams1["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * outputE2)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    ENBW = ((integrate(filterTransmission, detuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real

    print("ENBW obtained! Value: " + str(ENBW))
    print("FoM obtained! Value: " + str(figureOfMerit))
    print("Maximum transmission: " + str(filterTransmission.max()))

    # Plot the output.
    fig = plt.figure("Basic transmission")
    fig.set_size_inches(19.20, 10.80)
    plt.plot(detuning/1e3, filterTransmission * 100)

    # Make the graph pretty.
    plt.xlabel(r'$\Delta$ (GHz)')
    plt.ylabel(r'Transmission (\%)')
    plt.ylim(bottom = 0)
    plt.xlim(detuning[0]/1e3, detuning[-1]/1e3)
    plt.tight_layout()

    plt.show()

    return

if __name__ == "__main__":

    # Produce a single transmission output using the global parameters.
    #ProduceTransmissionSingle(globalDetuning, globalParams)

    # # Test the transmission against the four graphs shown in the literature for optimised filters.
    # TestTransmission(globalDetuning)

    # # Produce a single ENBW using the global parameters.
    # ProduceENBW(globalDetuning, globalParams)

    # # Test the ENBW against the literature values.
    # TestENBW(globalDetuning)

    # # Produce a single figure of merit using the global parameters.
    #ProduceFoM(globalDetuning, globalParams)

    # # Test the FoM against the literature values.
    # TestFoM(globalDetuning)

    # # Test the dual filter setup using the (only) literature values.
    inputParams1 = {'Bfield':270, 'rb85frac':72.17, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':86.7, 'Dline':'D2', 'Elem':'Rb'}
    inputParams2 = {'Bfield':240, 'rb85frac':72.17, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(6), 'lcell':50e-3, 'T':79, 'Dline':'D2', 'Elem':'Rb'}
    TwoFilterStats(globalDetuning, inputParams1, inputParams2)