"""
This module uses pygmo to find the optimum figure of merit for a specific filter setup.
We define an objective function to maximise, which will be the intensity in the y direction (we will start with light in the x direction only [given by E_in parameter])
TODO: Test with optimisation test functions (wiki article).
"""
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pygmo as pg
from elecsus import elecsus_methods as elecsus
from scipy.integrate import simps as integrate


# We begin by generating a class that defines the problem to be solved.
class figureOfMerit(object):

    def fitness(self, x):
        # MANDATORY: The fitness function is the function to be optimised. The returned value must be array-like (wrap it in []).
        # The input x is a multidimensional array of values i.e x = [x0, x1, ...., xN] depending on the dimension of the function (defined in get_bounds).
        # Detuning range.
        d = np.arange(-10000, 10000, 10) # MHz

        # Input parameters. Those represented by the input array x will be optimised.
        #elecsus_params = {'Bfield':x[0], 'rb85frac':0, 'Btheta':x[1], 'lcell':75e-3, 'T':x[2], 'Dline':'D2', 'Elem':'Rb'}
        elecsus_params = {'Bfield':x[0], 'rb85frac':72.17, 'Btheta':83, 'lcell':5e-3, 'T':126, 'Dline':'D2', 'Elem':'Rb'}

        [Iy] = elecsus.calculate(d, [1,0,0], elecsus_params, outputs=['Iy'])
        maxIy = max(Iy)
        ENBW = integrate(Iy, x=d/1e3)/maxIy
        FoM = maxIy/ENBW

        return([-1 * FoM])

    def get_bounds(self):
        # MANDATORY: Here we define the bounds of the function. This is required for pygmo to determine the dimensionality of the function.
        #return([0, 0, -100], [1e4, 90, 100])
        return([0], [1e4])

    def get_name(self):
        # OPTIONAL: Gives the problem a name. Just for aesthetics.
        return "Figure of Merit Generation"

def decorator(originalFitness):
    """
    This function is used to gather more information about the process of optimisation, namely the input values used at each iteration (called the decision vector [dv]).
    Useful for looking within the black box that is pygmo.
    """
    def newFitness(self, dv):
        """
        This function will wrap around the original fitness (objective) function, and will at each iteration log the test values being used.
        """

        fitnessVal = originalFitness(self, dv)

        global inputVals

        try:
            inputVals.append(np.append(dv, fitnessVal))
        except:
            inputVals = [np.append(dv, fitnessVal)]


        return fitnessVal

    return newFitness

# print("Problem loading...")
# problem = pg.problem(pg.decorator_problem(figureOfMerit(), fitness_decorator=decorator))
# print("Problem loaded. Obtaining random seed...")
# seed = random.get_data(data_type='uint16', array_length=1)[0]

# # Use cmaes or sea.
# print("Seed obtained, value: " + str(seed) + ". Planning algorithm...")
# algo = pg.algorithm(pg.simulated_annealing(n_T_adj = 1, seed = seed))
# algo.set_verbosity(1)   
# pop = pg.population(problem, 1)
# print("Timer starting.")
# startTime = time.time()

# pop = algo.evolve(pop)

# print("Elapsed time: {} seconds".format(time.time() - startTime))

# print(pop.champion_f) 
# print(pop.champion_x)

# print("Steps taken: " + str(inputVals))

# bStrength1 = np.array(inputVals)[::5,0]
# FoMs1 = -1 * np.array(inputVals)[::5,1]

# # bStrength2 = np.array(inputVals)[1::2,0]
# # FoMs2 = -1 * np.array(inputVals)[1::2,1]

# # print(bStrength)
# # print(FoMs)

# plt.plot(bStrength1, FoMs1, '-o', color='m')
# plt.plot(bStrength2, FoMs2, '-o', color='c')


def OptPlaneGraph():
    """
    Shows part of the function we are trying to optimise. This will come in handy to check whether our optimisation algorithm got stuck
    in a local minimum. The plan is to have the mayavi library generate a surface upon which we can show the path of the algorithm.
    """
    print("Generating optimisation graph. This could take some time...")

    bRange = np.linspace(0, 1e4, num=1000)
    FoMList = []
    detuning = np.arange(-10000, 10000, 10)

    for b in tqdm(bRange):

        x = [230, 83, 126]
        p_dict = {'Bfield':b, 'rb85frac':72.17, 'Btheta':x[1], 'lcell':5e-3, 'T':x[2], 'Dline':'D2', 'Elem':'Rb'}
        [Iy] = elecsus.calculate(detuning, [1,0,0], p_dict, outputs=['Iy'])
        ENBW = integrate(Iy, x=detuning/1e3)/np.amax(Iy)
        FoMList.append(np.amax(Iy)/ENBW)

    fig, ax = plt.subplots()
    fig.set_size_inches(19.20, 10.80)
    plt.plot(bRange, FoMList, color='c')
    ax.axvline(x[0], color='m')

    plt.xlabel("B field (G)")
    plt.xlim(0, 10000)
    plt.ylim(bottom=0)
    plt.ylabel(r"FoM (GHz$^{-1}$)")

    figName = str(p_dict["Elem"]) + "_" + str(p_dict["Dline"]) + "_" + "varyB" + "_" + str(p_dict["T"]) + "_" + str(p_dict["Btheta"]) + ".pdf"
    fig.tight_layout()
    plt.savefig(figName)
    print("Image saved as: " + figName)

    return

def CompareTimes():
    """
    This function is used to show the user how fast a certain optimisation method takes.
    Each algorithm will be timed for 200 function evaluations, INCLUDING SETUP TIME.
    Just one of a few criterion used to determine which algorithm is the best.
    Note that the same random seed is used for each algorithm to ensure a uniform start.
    Returns the time taken for each algorithm in a single array (useful for determining
    the convergence per unit time).
    """

    problem = pg.problem(figureOfMerit())
    noParticles = 20
    seed = np.random.randint(1e6)
    algoList = [pg.sea(gen = 200, seed = seed),
                pg.bee_colony(gen = 5, seed = seed, limit = 10),
                pg.cmaes(gen = 11, seed = seed),
                pg.simulated_annealing(n_T_adj = 1, seed = seed),
                pg.sade(gen = 10, seed = seed)]

    timeList = []
    nameList = []

    for algo in algoList:

        algo = pg.algorithm(algo)
        algo.set_verbosity(1)

        algoName = algo.get_name()
        print("Starting timing for: " + algoName)
        startTime = time.time()

        pop = pg.population(problem, noParticles)
        pop = algo.evolve(pop)

        elapsedTime = time.time() - startTime
        print("Algorithm complete. Elapsed time: {} seconds".format(elapsedTime))

        timeList.append(elapsedTime)
        nameList.append(algoName)

    # Plot bar graph.
    noAlgos = len(nameList)

    fig, ax = plt.subplots()
    fig.set_size_inches(19.20, 10.80)

    index = np.arange(noAlgos)

    barChart = ax.bar(index, timeList)

    ax.set_xlabel("Algorithm name")
    ax.set_ylabel("Time for 1000 function evaluations (s)")
    ax.set_xticks(index)
    ax.set_xticklabels(algoList)

    fig.tight_layout()
    plt.ylim(bottom=0)
    figName = "algo_time_compare.pdf"
    plt.savefig(figName)
    print("Image saved as: " + figName)

    return timeList

def CompareConvergence():
    """
    This function is used to show the user the optimal value found after 200 function evaluations.
    Just one of a few criterion used to determine which algorithm is the best.
    Note that the same random seed is used for each algorithm to ensure a uniform start.
    Returns the optimal value obtained by each algorith in a single array (useful for
    determining the convergence per unit time). Note that dividing by 200 will give
    the average convergence per evalutation.
    """

    problem = pg.problem(figureOfMerit())
    noParticles = 20
    seed = np.random.randint(1e6)
    algoList = [pg.sea(gen = 200, seed = seed),
                pg.bee_colony(gen = 5, seed = seed, limit = 10),
                pg.cmaes(gen = 11, seed = seed),
                pg.simulated_annealing(n_T_adj = 1, seed = seed),
                pg.sade(gen = 10, seed = seed)]

    champList = []
    fig, ax = plt.subplots()
    fig.set_size_inches(19.20, 10.80)
    
    for algo in algoList:

        pgAlgo = pg.algorithm(algo)
        pgAlgo.set_verbosity(1)

        algoName = pgAlgo.get_name()
        print("Starting algorithm: " + algoName)

        pop = pg.population(problem, noParticles)
        pop = pgAlgo.evolve(pop)

        champVal = pop.champion_f
        print("Algorithm complete. Best FoM: " + str(champVal))
        champList.append(champVal)

        log = np.array(pgAlgo.extract(type(algo)).get_log())
        if "Annealing" in algoName:
            # We are testing Simulated Annealing, which as a different log structure.
            plt.plot(log[:,0], -1 * log[:,1], label = algoName)
        else:
            plt.plot(log[:,1], -1 * log[:,2], label = algoName)

    plt.legend()

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("FoM")

    fig.tight_layout()
    figName = "algo_converge_compare.pdf"
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig(figName)
    print("Image saved as: " + figName)

    return champList
    
if __name__ == '__main__':
    print("Running test cases...")
    #OptPlaneGraph()
    timeList = CompareTimes()
    champList = CompareConvergence()
    efficiencyList = np.divide(champList, timeList)

    print("Convergence per unit time list: ")
    print(efficiencyList)