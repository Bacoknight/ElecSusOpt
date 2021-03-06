"""
Here we explicitly plot out the graphs listed on the 'cool graphs' column of the whiteboard.
These are done with explicit values and are not intended to be used for anything programming related.
These will produce graphs which should be ready for inserting into a report or presentation.
"""

import matplotlib.pyplot as plt
plt.rc("text", usetex = True)

import json
from scipy.integrate import simps as integrate
from elecsus import elecsus_methods as elecsus
from tqdm import tqdm
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.image as mpimg
import matplotlib.transforms as mtrans
from sklearn.externals import joblib
import shap

from matplotlib.animation import FuncAnimation
from PIL import Image
from subprocess import check_call

import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

def PlotTreeModels():
    """
    Shows how the ExtraTrees model evolves over a certain number of iterations. It will show the model after the random sampling, half way through the
    optimisation and at the end. It will also show the sparsity of the model after these iterations using the export_to_graphviz function.
    Obtains the data from visualise_iters.txt.
    """

    plotDpi = 100

    with open("visualise_iters.txt") as jsonFile:
        data = json.load(jsonFile)
        numIters = data["numIters"]
        bRange = data["bRange"]
        objVals = data["objVals"]
        randomVals = data["randomVals"]
        halfVals = data["halfVals"]
        endVals = data["endVals"]

    fig = plt.figure("Model Visualisation", dpi = plotDpi)
    fig.set_size_inches(19.20, 10.80)

    # Gridspec allows for more complex axes.
    gridSpec = fig.add_gridspec(3, 2)
    funcPlot = plt.subplot(gridSpec.new_subplotspec((0, 0), rowspan = 3))
    randTreePlot = plt.subplot(gridSpec.new_subplotspec((0, 1)))
    randTreePlot.tick_params(axis = "both", which = "both", bottom = False, top = False, labelbottom = False, right = False, left = False, labelleft = False)
    randTreePlot.set_xlabel("Random seeding")
    randTreePlot.patch.set_visible(False)
    halfTreePlot = plt.subplot(gridSpec.new_subplotspec((1, 1)))
    halfTreePlot.tick_params(axis = "both", which = "both", bottom = False, top = False, labelbottom = False, right = False, left = False, labelleft = False)
    halfTreePlot.set_xlabel(r"50\% complete")
    endTreePlot = plt.subplot(gridSpec.new_subplotspec((2, 1)))
    endTreePlot.tick_params(axis = "both", which = "both", bottom = False, top = False, labelbottom = False, right = False, left = False, labelleft = False)
    endTreePlot.set_xlabel("Final model")

    # Plot the function and their predictions.
    funcPlot.plot(bRange, objVals, label = "True function")
    funcPlot.plot(bRange, randomVals, label = "Random seeding")
    funcPlot.plot(bRange, halfVals, label = r"50\% complete")
    funcPlot.plot(bRange, endVals, label = "Final model")
    funcPlot.set_xlim(bRange[0], bRange[-1])
    funcPlot.set_ylim(bottom = 0)
    funcPlot.set_xlabel("Magnetic Field Strength (Gauss)")
    funcPlot.set_ylabel(r"Figure of Merit (GHz$^{-1}$)")

    # Do this after plotting labels and axes, but before the axes are measured to ensure a nice and tight graph.
    plt.tight_layout()

    # Obtain the size of a single plot in inches to inform the dot picture creator.
    ext = randTreePlot.get_position().transformed(fig.transFigure).extents
    bbox = mtrans.Bbox.from_extents(np.array(ext).astype(int))

    height = bbox.height
    width = bbox.width

    # Turn the DOT files into images and plot them.
    check_call(['dot','-Tpng', "-Gsize={},{}!".format(width/100, height/100), "-Gdpi={}".format(plotDpi), "-Gratio=fill", 'random_tree.dot','-o','random_tree.png'])
    randImg = Image.open("random_tree.png")
    randTreePlot.imshow(randImg, resample = False)

    check_call(['dot','-Tpng', "-Gsize=" + str(width/100) + "," + str(height/100) + "!", "-Gdpi=100", "-Gratio=fill", 'half_tree.dot','-o','half_tree.png'])
    halfImg = Image.open("half_tree.png")
    halfTreePlot.imshow(halfImg, aspect = "auto")

    check_call(['dot','-Tpng', "-Gsize=" + str(width/100) + "," + str(height/100) + "!", "-Gdpi=100", "-Gratio=fill", 'end_tree.dot','-o','end_tree.png'])
    endImg = Image.open("end_tree.png")
    endTreePlot.imshow(endImg, aspect = "auto")

    funcPlot.legend()
    plt.savefig("iter_trees.pdf")
    plt.show()

    return

def PlotSingleMap():
    """
    Plots a graph of temperature and magnetic field strength against figure of merit for simple single filter.
    """

    baseParams = {'Bfield':230, 'rb85frac':72.17, 'Btheta':np.deg2rad(83), 'Etheta':np.deg2rad(6), 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}
    detuning = np.linspace(-25000, 25000, 20000)
    inputE = np.array([np.cos(baseParams["Etheta"]), np.sin(baseParams["Etheta"]), 0])
    # Use a Jones matrix to determine the electric field after the action of the second polariser. As this is a single filter, the two polarisers are crossed.
    polariserAngle = baseParams["Etheta"] + np.pi/2

    # Define the Jones matrix. Though only explicitly defined for the x-y plane, we add the third dimension so that we can use all 3 dimensions of the output field.
    jonesMatrix = np.matrix([[np.cos(polariserAngle)**2, np.sin(polariserAngle)*np.cos(polariserAngle), 0],
                                [np.sin(polariserAngle)*np.cos(polariserAngle), np.sin(polariserAngle)**2, 0],
                                [0, 0, 1]])

    numPoints = 71

    angRange = np.linspace(0., 90., numPoints)
    fieldRange = np.linspace(10., 400., numPoints)

    fomVals = []

    for ang in tqdm(angRange):
        for field in fieldRange:
            inputParams = baseParams.copy()
            inputParams["Bfield"] = field
            inputParams["Btheta"] = np.deg2rad(ang)

            # Call ElecSus to obtain the output electric field.
            [outputE] = elecsus.calculate(detuning, inputE, inputParams, outputs = ["E_out"])

            # Get the output from the filter and the polarisers.
            singleFilterOutputE = np.array(jonesMatrix * outputE)

            # Get the transmission.
            singleFilterTransmission = (singleFilterOutputE * singleFilterOutputE.conjugate()).sum(axis=0)

            ENBW = ((integrate(singleFilterTransmission, detuning)/singleFilterTransmission.max().real)/1e3).real

            fomVals.append((singleFilterTransmission.max()/ENBW).real)
    
    Bs, Ts = np.meshgrid(fieldRange, angRange)
    
    fig = plt.figure("2D Heatmap")
    fig.set_size_inches(19.20, 10.80)
    ax = fig.add_subplot(111)
    im = plt.pcolormesh(Bs, Ts, np.array(fomVals).reshape((len(angRange), len(angRange))), shading = "gouraud", cmap = "YlOrRd")
    ax.set_xlabel("Magnetic Field Strength (Gauss)")
    ax.set_ylabel(r"$\theta_{\textrm{B}}$ (${}^{\circ}$)")
    colorbar = fig.colorbar(im)
    colorbar.set_label(r"Figure of Merit (GHz$^{-1}$)")
    fig.tight_layout()

    plt.show()

    return

def PlotAcqComp():
    """
    Obtain the JSON data from acq_plot.txt and plot it as two graphs showing the iteration path and time taken.
    """

    timeList = []
    errorList = [[], []]
    nameList = []
    barColourList = []

    # Open the file and extract the information from it.
    with open("acq_plot.txt") as jsonFile:
        data = json.load(jsonFile)
        realIters = range(1, data.get("numIters", 100) + 1)
        optimiserList = data.get("optimiserList")

    fig = plt.figure("Acquisition function comparison")
    fig.set_size_inches(19.20, 10.80)
    pathPlot = fig.add_subplot(121)
    timePlot = fig.add_subplot(122)

    for optimiser in optimiserList:
        # Plot the graphs.
        avgTime = np.mean(optimiser[2])
        timeList.append(avgTime)
        errorList[0].append(avgTime - min(optimiser[2]))
        errorList[1].append(max(optimiser[2]) - avgTime)
        nameList.append(optimiser[0])
        barColourList.append(optimiser[1])
        # Plot the average result.
        pathPlot.plot(realIters, np.mean(optimiser[3], axis = 0), c = optimiser[1], marker = ".", markersize = 12, markevery = int(np.ceil(data.get("numIters", 100)/10)), lw = 2, label = optimiser[0])

        for run in range(len(optimiser[2])):
            pathPlot.plot(realIters, optimiser[3][run], c = optimiser[1], alpha = 0.2)

    # Plot the time bar chart.
    timePlot.bar(nameList, timeList, align = "center", color = barColourList, yerr = errorList, capsize = 20)
    timePlot.set_xlabel("Acquisition Function")
    timePlot.set_ylabel("Average runtime for " + str(data.get("numIters", 100)) + " iterations (s)")

    pathPlot.legend()
    pathPlot.set_xlim(1, data.get("numIters", 100))
    pathPlot.set_ylim(bottom = 0)
    pathPlot.set_xlabel("Iteration number")
    pathPlot.set_ylabel(r"Best Figure of Merit (GHz$^{-1}$)")

    plt.tight_layout()
    plt.savefig("acq_plot.pdf")
    plt.show()

    return

def PlotParamTuning():
    """
    Obtain the JSON data from kappa_frac_plot.txt and plot it.
    """

    # Open the file and extract the information from it.
    with open("kappa_frac_plot.txt") as jsonFile:
        data = json.load(jsonFile)
        kappas = data.get("kappas")
        fracs = data.get("fracs")
        avgList = data.get("avgList")

    fig = plt.figure("Acquisition function parameters")
    fig.set_size_inches(19.20, 10.80)
    plt.pcolormesh(kappas, fracs, avgList, shading = "gouraud")
    plt.xlabel("Kappa")
    plt.ylabel("Fraction of iterations used for random sampling")
    cb = plt.colorbar()
    cb.set_label(r"Average Figure of Merit (GHz$^{-1}$)", rotation = 270, labelpad = 15)

    plt.tight_layout()
    plt.savefig("kappa_frac_plot.pdf")
    plt.show()

    return

def PlotVariableImportance():
    """
    Obtain the JSON data from variable_importance.txt and plot a stacked bar chart to compare them.
    """

    # Open the file and extract the data.
    with open("variable_importance.txt") as jsonFile:
        data = json.load(jsonFile)
        resultList = [("Single Filter", data.get("Single Filter")), ("Cascaded Filter\n(No Middle Polariser)", data.get("Cascaded Filter, No Middle Polariser")),
        ("Cascaded Filter\n(Middle Polariser General)", data.get("Cascaded Filter, Middle Polariser General")), ("Cascaded Filter\n(Middle Polariser Perpendicular)", data.get("Cascaded Filter, Middle Polariser Perpendicular"))]

    # Define the sorting list. This is so the graphs look coherent.
    variableOrder = [("B field 1", r"$|\textbf{B}_{\textrm{1}}|$"), ("Temperature 1", r"$T_{\textrm{1}}$"), ("E theta", r"$\theta_{\textrm{E}}$"),
    ("B theta 1", r"$\theta_{\textrm{B}_1}$"), ("B phi 1", r"$\phi_{\textrm{B}_1}$"), ("B field 2", r"$|\textbf{B}_{\textrm{2}}|$"), ("Temperature 2", r"$T_{\textrm{2}}$"),
    ("B theta 2", r"$\theta_{\textrm{B}_2}$"), ("B phi 2", r"$\phi_{\textrm{B}_2}$"), ("Free polariser angle", r"$\theta_{\textrm{P}}$")]

    # Give each variable its own colour.
    colours = cm.viridis(np.linspace(0.25, 1.0, len(variableOrder)))

    # Define bar width.
    width = 0.8

    # Set up the figure.
    stackedFig = plt.figure("Variable importance")
    stackedFig.set_size_inches(19.20, 10.80)

    # Plot each BAR individually, to catch out any KeyValue errors.
    for position, result in enumerate(resultList):
        totHeight = 0
        for index, variable in enumerate(variableOrder):
            if index == 0:
                # Plot the first variable (B field 1) explicitly.
                plt.bar(position, result[1][variable[0]], width = width, color = colours[index], bottom = totHeight)
                if result[1][variable[0]] < 0.075:
                    # Shift the label down a bit so it isn't cut off by the bar.
                    plt.text(position, totHeight + result[1][variable[0]]/3, variable[1], ha = "center", va = "center")
                elif result[1][variable[0]] < 0.25:
                    plt.text(position, totHeight + result[1][variable[0]]/2.5, variable[1], ha = "center", va = "center")
                else:
                    # Shift the label up a bit because it looks weird otherwise.
                    plt.text(position, totHeight + result[1][variable[0]]/2, variable[1], ha = "center", va = "center")
                totHeight += result[1][variable[0]]
            else:
                try:
                    # Stack the next bar on top of the old one.
                    plt.bar(position, result[1][variable[0]], width = width, color = colours[index], bottom = totHeight)
                    if result[1][variable[0]] < 0.075:
                        # Shift the label down a bit so it isn't cut off by the bar.
                        plt.text(position, totHeight + result[1][variable[0]]/3, variable[1], ha = "center", va = "center")
                    elif result[1][variable[0]] < 0.25:
                        plt.text(position, totHeight + result[1][variable[0]]/2.5, variable[1], ha = "center", va = "center")
                    else:
                        # Shift the label up a bit because it looks weird otherwise.
                        plt.text(position, totHeight + result[1][variable[0]]/2, variable[1], ha = "center", va = "center")
                    totHeight += result[1][variable[0]]
                except KeyError:
                    # This variable isn't present for this filter. The way the variable list is set up, it is safe to move to the next result in this case.
                    continue

    # Plot aesthetics.
    plt.ylabel("Variable importance")
    plt.xticks(np.arange(len(resultList)), [result[0] for result in resultList])
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig("param_importance.pdf")
    plt.show()

    return

def PlotShap():
    """
    Plots SHAP force graphs in an attempt to explain the model.
    """

    model, resultDataframe, bestRow = joblib.load("shap_data.pkl")
    
    # Initialise SHAP.
    try:
        # Load the SHAP values and explainer.
        explainer, shapValues = joblib.load("shap_values_explainer.pkl")
    except:
        # Create the values and save them for next time.
        explainer = shap.TreeExplainer(model)
        shapValues = explainer.shap_values(resultDataframe)
        joblib.dump((explainer, shapValues), "shap_values_explainer.pkl")

    # Checks to make sure the SHAP library is working fine.
    # for index, row in resultDataframe.iterrows():
    #     # Should be all True.
    #     print(np.isclose((explainer.expected_value + np.sum(shapValues[index,:])), model.predict(np.array(row).reshape(1, -1))))

    # Generate cool plots.
    #shap.force_plot(-1 * explainer.expected_value, -1 * shapValues[bestRow,:], features = None, feature_names = list(resultDataframe.columns.values),  matplotlib = True)
    shap.summary_plot(shapValues, resultDataframe)

    return

def PlotThreeFilterTrans():
    """
    Plots the transmission from three filters.
    """

    # Load the data.
    detuning, transFin, transSecond, transFirst = joblib.load("three_filter_trans.pkl")

    # Set up the plots.
    fig = plt.figure("Filter Transmissions")
    fig.set_size_inches(19.20, 10.80)
    
    # Create the plot grid, in the form (y, x).
    gridSpec = fig.add_gridspec(2, 2)
    firstPlot = plt.subplot(gridSpec.new_subplotspec((1, 0)))
    secondPlot = plt.subplot(gridSpec.new_subplotspec((0, 0)), sharex = firstPlot)
    finalPlot = plt.subplot(gridSpec.new_subplotspec((0, 1), rowspan = 2))

    firstPlot.set_xlabel("Detuning (GHz)")
    firstPlot.set_ylabel("Transmission")
    firstPlot.set_ylim(0, 0.2)
    firstPlot.set_xlim(-10, 10)
    firstPlot.plot(detuning/1e3, transFirst)

    secondPlot.set_ylabel("Transmission")
    secondPlot.set_ylim(0, 0.2)
    secondPlot.plot(detuning/1e3, transSecond)

    finalPlot.set_xlabel("Detuning (GHz)")
    finalPlot.set_ylabel("Transmission")
    finalPlot.set_ylim(0, 0.7)
    finalPlot.set_xlim(-10, 10)
    finalPlot.plot(detuning/1e3, transFin)

    plt.tight_layout()
    plt.show()

    return

def PlotIterVis2D():
    """
    Plots the 2D model predictions as a heatmap.
    """

    # Open the file and extract the data.
    with open("visualise_iters_2d.txt") as jsonFile:
        data = json.load(jsonFile)
        numIters = data.get("numIters")
        bRange = data.get("bRange")
        tRange = data.get("tRange")
        objVals = data.get("objVals")
        predictVals = data.get("predictVals")

    Bs, Ts = np.meshgrid(bRange, tRange)

    fig = plt.figure("Model Predictions")
    fig.set_size_inches(19.20, 10.80)
    ax = fig.add_subplot(111)
    im = plt.pcolormesh(Bs, Ts, np.array(predictVals[0]).reshape((len(bRange), len(bRange))), shading = "gouraud")
    colorbar = fig.colorbar(im)
    fig.tight_layout()

    def animate(i):
        newData = np.array(predictVals[i]).reshape((len(bRange), len(bRange))).ravel()
        vMax = np.max(newData)
        vMin = np.min(newData)
        im.set_array(newData)
        im.set_clim(vMin, vMax)

    ani = FuncAnimation(fig, animate, frames = len(predictVals))

    plt.show()
    
    # # Set up the plots.
    # fig = plt.figure("Model Predictions")
    # fig.set_size_inches(19.20, 10.80)
    # plt.pcolormesh(Bs, Ts, np.array(objVals).reshape((len(bRange), len(bRange))))
    # plt.xlabel("Magnetic Field (Gauss)")
    # plt.ylabel("Temperature (Celcius)")
    # plt.colorbar()
    # plt.show()

    
if __name__ == "__main__":
    # PlotAcqComp()

    # PlotParamTuning()

    # PlotVariableImportance()

    # PlotTreeModels()

    PlotShap()

    #PlotThreeFilterTrans()

    #PlotIterVis2D()

    #PlotSingleMap()