"""
A module for plotting various graphs to help humans understand the problem we are trying to solve.
"""

import time
from elecsus import elecsus_methods as elecsus
import chocolate as choco
import numpy as np
from scipy.integrate import simps as integrate
import matplotlib.pyplot as plt
import lmfit
from mpi4py import MPI

# Here we define some global variables so that it is easier to change it for all functions.
# Detuning used for all tests.
globalDetuning = np.arange(-100000, 100000, 10) # MHz

def ProduceSpectrum(detuning, params, toPlot = False):
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

def Objective1D(bField, elecsusParams):
    """
    First variable to vary is the magnetic field strength.
    """

    elecsusParams["Bfield"] = bField

    return CalculateFoM(globalDetuning, elecsusParams)

def PlotFoMvsBGraph():
    """
    This function will plot the variation of figure of merit with magnetic field, for each experiment defined in ElecSus.
    """

    # Find all the cores being used.
    comm = MPI.COMM_WORLD
    
    # Get the 'name' of each core to refer to them.
    rank = comm.Get_rank()

    # Define the parameters for each test.
    paramsList = [{'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb', 'Label': "Rb, Nat. Ab."},
        {'Bfield':695, 'rb85frac':72.17, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(45), 'lcell':5e-3, 'T':147, 'Dline':'D1', 'Elem':'Rb', 'Label': "Rb, Nat. Ab."},
        {'Bfield':224, 'rb85frac':100, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(90), 'lcell':5e-3, 'T':121, 'Dline':'D2', 'Elem':'Rb', 'Label': "Rb-85"},
        {'Bfield':310, 'rb85frac':100, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(46), 'lcell':5e-3, 'T':139, 'Dline':'D1', 'Elem':'Rb', 'Label': "Rb-85"},
        {'Bfield':849, 'rb85frac':0, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(81), 'lcell':5e-3, 'T':121, 'Dline':'D2', 'Elem':'Rb', 'Label': "Rb-87"},
        {'Bfield':315, 'rb85frac':0, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(45), 'lcell':5e-3, 'T':141, 'Dline':'D1', 'Elem':'Rb', 'Label': "Rb-87"},
        {'Bfield':1120, 'Btheta':np.deg2rad(87), 'Etheta':np.deg2rad(89), 'lcell':5e-3, 'T':127, 'Dline':'D2', 'Elem':'Cs', 'Label': "Cs"},
        {'Bfield':338, 'Btheta':np.deg2rad(89), 'Etheta':np.deg2rad(46), 'lcell':5e-3, 'T':121, 'Dline':'D1', 'Elem':'Cs', 'Label': "Cs"},
        {'Bfield':88, 'Btheta':np.deg2rad(1), 'Etheta':np.deg2rad(87), 'lcell':5e-3, 'T':150, 'Dline':'D2', 'Elem':'K', 'Label': "K"},
        {'Bfield':460, 'Btheta':np.deg2rad(90), 'Etheta':np.deg2rad(47), 'lcell':5e-3, 'T':177, 'Dline':'D1', 'Elem':'K', 'Label': "K"},
        {'Bfield':144, 'Btheta':np.deg2rad(0), 'Etheta':np.deg2rad(0), 'lcell':5e-3, 'T':245, 'Dline':'D2', 'Elem':'Na', 'Label': "Na"},
        {'Bfield':945, 'Btheta':np.deg2rad(88), 'Etheta':np.deg2rad(41), 'lcell':5e-3, 'T':279, 'Dline':'D1', 'Elem':'Na', 'Label': "Na"}]

    bRange = np.linspace(0, 0.2e4, num = 2000)

    # Ensure there is a core per calculation.
    assert (comm.Get_size() == len(paramsList)), "You need to run this on " + str(len(paramsList)) + " processes, one for each problem."

    # Pick the experiment being ran on each thread.
    params = paramsList[rank]

    # Create a list to store values.
    fomList = []

    for bField in bRange:
        fomList.append(Objective1D(bField, params))

    if rank != 0:
        comm.send(fomList, dest = 0)

    # Wait for all parallel processes to finish.
    comm.Barrier()

    # Grab all the results and plot it.
    if rank == 0:
        # Set up the plot.
        fig = plt.figure("Figure of Merit vs B Field")
        fig.set_size_inches(19.20, 10.80)
        axD1 = plt.subplot(1, 2, 1)
        axD2 = plt.subplot(1, 2, 2, sharey = axD1)
        # Plot the results from process 0.
        if paramsList[rank]["Dline"] == "D1":
            axD1.plot(bRange, fomList, label = paramsList[rank]["Label"])
        else:
            axD2.plot(bRange, fomList, label = paramsList[rank]["Label"])
        # Grab results and plot them.
        for source in range(comm.Get_size()):
            if source != 0:
                # Receive and plot the data.
                fomData = comm.recv(source = source)
                if paramsList[source]["Dline"] is "D1":
                    axD1.plot(bRange, fomData, label = paramsList[source]["Label"])
                else:
                    axD2.plot(bRange, fomData, label = paramsList[source]["Label"])

        axD1.set_xlabel("Magnetic Field Strength [Gauss]")
        axD1.set_ylabel(r"Figure of Merit [GHz$^{-1}$]")
        axD1.legend()
        axD2.set_xlabel("Magnetic Field Strength [Gauss]")
        axD2.set_ylabel(r"Figure of Merit [GHz$^{-1}$]")
        axD2.legend(prop = {"size": 6})
        axD1.set_xlim(0, 2000)
        axD1.set_ylim(bottom = 0)
        axD2.set_xlim(0, 2000)
        axD2.set_ylim(bottom = 0)
        plt.savefig("fom_vs_bfield.pdf")
        print("Plot saved.")

        plt.show()

    return

if __name__ == "__main__":
    PlotFoMvsBGraph()
