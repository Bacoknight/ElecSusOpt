"""
This module introduces a second filter and tasks the algorithm with reproducing the figure of merit determined in the literature.
This is a simplified version as we only consider the variation of temperatures and magnetic field strengths in each filter.
"""

from elecsus import elecsus_methods as elecsus
import numpy as np
from scipy.integrate import simps as integrate
import skopt
from matplotlib import pyplot as plt
import scipy.optimize as optimise

# Define global parameters.
globalDetuning = np.linspace(-25000, 25000, 1000)
globalParams = {'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}

# Literature parameters.
lit1Params = {'Bfield':270, 'rb85frac':72.17, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':86.7, 'Dline':'D2', 'Elem':'Rb'}
lit2Params = {'Bfield':240, 'rb85frac':72.17, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(6), 'lcell':50e-3, 'T':79, 'Dline':'D2', 'Elem':'Rb'}

def HalfFilter(detuning, inputParams):
    """
    Ensure the dual filter is correct by checking if we get the same results if we split a single filter in half.
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

    ENBWSingle = ((integrate(singleFilterTransmission, detuning)/singleFilterTransmission.max().real)/1e3).real

    figureOfMeritSingle = (singleFilterTransmission.max()/ENBWSingle).real

    print("Single filter computations complete! Here are the stats:")
    print("ENBW value: " + str(ENBWSingle))
    print("FoM value: " + str(figureOfMeritSingle))
    print("Maximum transmission: " + str(singleFilterTransmission.max()))

    # Reproduce these results with a double filter.
    # Create a new set of parameters with a halved cell length.
    dualFilterParams = {**inputParams, 'lcell': inputParams["lcell"]/2.0}
    # First generate the output transmission for the first filter.
    inputE = np.array([np.cos(dualFilterParams["Etheta"]), np.sin(dualFilterParams["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field from the first filter.
    [outputE1] = elecsus.calculate(detuning, inputE, dualFilterParams, outputs = ["E_out"])

    # Call ElecSus to obtain the output field from the second filter.
    [outputE2] = elecsus.calculate(detuning, outputE1, dualFilterParams, outputs = ["E_out"])

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = dualFilterParams["Etheta"] + np.pi/2

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

    print("-----------------------------------------------------------------------------------")
    print("Double filter computations complete! Here are the stats:")
    print("ENBW value: " + str(ENBW))
    print("FoM value: " + str(figureOfMerit))
    print("Maximum transmission: " + str(filterTransmission.max()))

    return

def TwoFilterLiterature(detuning, inputParams1, inputParams2):
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
    plt.plot(detuning/1e3, filterTransmission.real)

    # Make the graph pretty.
    plt.xlabel("Detuning (GHz)")
    plt.ylabel("Transmission")

    plt.show()

    return

def TwoFilterFitness(inputParams):
    """
    The fitness function used by the skopt algorithm. The input is a list of variables. As this is the basic two filter setup, the only inputs are the temperatures and the 
    """

    filter1Params = lit1Params.copy()
    filter2Params = lit2Params.copy()

    filter1Params["Bfield"] = inputParams[0]
    filter1Params["T"] = inputParams[1]
    filter2Params["Bfield"] = inputParams[2]
    filter2Params["T"] = inputParams[3]

    # First generate the output transmission for the first filter.
    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field from the first filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE1] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the first filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    # Call ElecSus to obtain the output field from the second filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE2] = elecsus.calculate(globalDetuning, outputE1, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the second filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE1))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filter1Params["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * outputE2)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    ENBW = ((integrate(filterTransmission, globalDetuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real
    
    if np.isnan(figureOfMerit):
        # Usually occurs in the case of high temperatures and B fields, since the transmission is just a flat line.
        print("Figure of merit is NaN! Here are the input parameters:")
        print("Filter 1 parameters:")
        print(str(filter1Params))
        print("Filter 2 parameters:")
        print(str(filter2Params))
        return 0.0
    else:
        return -1.0 * figureOfMerit

def OptimiseSimpleTwoFilter(numIters):
    """
    Run the (4D) optimisation of the dual filter with the aim of reproducing the literature values.
    """

    problemBounds = [(10, 1300), (40, 230), (10, 1300), (40, 230)]

    result = skopt.forest_minimize(TwoFilterFitness, problemBounds, verbose = True, n_calls = numIters, n_random_starts = int(np.ceil(numIters/10)))

    print("Complete:")
    print(result.fun)

    return

def PlotTransmission(inputParams):
    """
    Plot the transmission of the double filter.
    """

    filter1Params = lit1Params.copy()
    filter2Params = lit2Params.copy()

    filter1Params["Bfield"] = inputParams[0]
    filter1Params["T"] = inputParams[1]
    filter1Params["Etheta"] = np.deg2rad(inputParams[2])
    filter1Params["Btheta"] = np.deg2rad(inputParams[3])
    filter1Params["Bphi"] = np.deg2rad(inputParams[4])
    filter2Params["Bfield"] = inputParams[5]
    filter2Params["T"] = inputParams[6]
    filter2Params["Btheta"] = np.deg2rad(inputParams[7])
    filter2Params["Bphi"] = np.deg2rad(inputParams[8])

    # First generate the output transmission for the first filter.
    inputE = np.array([np.cos(filter1Params["Etheta"]), np.sin(filter1Params["Etheta"]), 0])

    # Call ElecSus to obtain the output electric field from the first filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE1] = elecsus.calculate(globalDetuning, inputE, filter1Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the first filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter1Params))
        print("Input field: " + str(inputE))
        return 0.0

    # Call ElecSus to obtain the output field from the second filter.
    try:
        # There may at times be issues with ElecSus, such as when NaN is entered as a variable.
        [outputE2] = elecsus.calculate(globalDetuning, outputE1, filter2Params, outputs = ["E_out"])
    except:
        print("There was an issue in ElecSus for the second filter, so this iteration will return a figure of merit of 0. Here are the input parameters:")
        print("Input parameters: " + str(filter2Params))
        print("Input field: " + str(outputE1))
        return 0.0

    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = filter1Params["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
								[np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    # Get the output from the filter and the polarisers.
    outputE = np.array(jonesMatrix * outputE2)

    # Get the transmission.
    filterTransmission = (outputE * outputE.conjugate()).sum(axis=0).real

    ENBW = ((integrate(filterTransmission, globalDetuning)/filterTransmission.max().real)/1e3).real

    figureOfMerit = (filterTransmission.max()/ENBW).real

    print("Computation complete! Here are the relevant values:")
    print("ENBW: " + str(ENBW))
    print("Figure of Merit: " + str(figureOfMerit))

    plt.plot(globalDetuning/1e3, filterTransmission)
    plt.xlim(globalDetuning[0]/1e3, globalDetuning[-1]/1e3)
    plt.ylim(bottom = 0)
    plt.xlabel("Detuning (GHz)")
    plt.ylabel("Transmission")
    plt.show()
    
    return

if __name__ == "__main__":

    # # Test the dual filters by splitting in half.
    # HalfFilter(globalDetuning, globalParams)

    # # Reproduce the literature values.
    inputParams1 = {'Bfield':270, 'rb85frac':72.17, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':86.7, 'Dline':'D2', 'Elem':'Rb'}
    inputParams2 = {'Bfield':240, 'rb85frac':72.17, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(6), 'lcell':50e-3, 'T':79, 'Dline':'D2', 'Elem':'Rb'}
    TwoFilterLiterature(globalDetuning, inputParams1, inputParams2)

    # # Test the fitness function.
    # print("Looking for about 1.2...")
    #print(abs(TwoFilterFitness([270, 86.7, 240, 79])))

    # Run the optimisation.
    #OptimiseSimpleTwoFilter(1000)

    # Plot the transmission of a double filter.
    # inputParams = [314, 109, 50, 86, 59, 199, 77, 77, 2]
    # PlotTransmission(inputParams)