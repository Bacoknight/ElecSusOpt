""" 

Optimisation of bandpass optical filters for weird geometries



This file in particular - use random restart method with a local optimiser to try and find global optimum



JK, 2017

"""



import numpy as np

import matplotlib.pyplot as plt



import scipy.integrate as integrate

import lmfit as lm

import sys, os, time, copy, glob

import cPickle as pickle



# multiprocessing / parallelisation on local machine cores

import multiprocessing as mp

import psutil



from elecsus import elecsus_methods as em # elecsus should be installed, not local 



"""

Pseudo-code

----------------



1. 	Define FOM (to minimise) - a few options available

			a. ENBW / Transmission 

				i. @ whole spectrum

				ii. @ specific frequency

			b. FWHM / Transmission @ specific peak <<<< this one is used here.

			c. Transmission at multiple frequencies (transmission at one freq AND extinction at another etc)

2.		Define function to calculate FOM from given parameters

			Parameters to vary:

				i.		Cell length <<< fixed for a given run

				ii.		Temperature

				iii.		Magnetic field strength (uniform over cell - possibly include gradient as well? << not yet - too slow)

				iv.	Magnetic field angle

				v.		Initial polarisation, subject to crossed polarisers

3.		Compute multi-dimensional FOM maps

			a. Visualise through matplotlib, interactively (event handler / buttons on plot)

			b. Visualise through scatter matrices (pandas?)

4.		Define optimisation problem based on all of the above and optimise the **** out of it.



"""



def calcFWHM(spectrumDetuning, spectrumTransmission):

	""" 

	Calculate full-width at half-maximum for a given input spectrum.

	Note this is a fairly crude method, and will give non-sensical values for data which

	has multiple closely spaced peaks

	This finds the local half-maximum points which are the first points to the left and right

	of the main peak that have a transmission value which is < 1/2 of the maximum value.

	"""

	

	#fig = plt.figure(); ax = fig.add_subplot(111)

	#ax.plot(spectrumDetuning, spectrumTransmission)

	

	positionOfMax = spectrumDetuning[spectrumTransmission.argmax()]

	#ax.axvline(positionOfMax)

	

	argOfMax = spectrumTransmission.argmax()

	heightOfMax = spectrumTransmission.max()

	

	HM = heightOfMax / 2.

	#ax.axhline(heightOfMax)

	#ax.axhline(HM)

	

	detuningLeft = spectrumDetuning[0:argOfMax]

	specLeft = spectrumTransmission[0:argOfMax]

	halfMaskBoolean = specLeft<HM

	hwLeft = detuningLeft[halfMaskBoolean][-1]

	#ax.axvline(hwLeft, color='k')

	

	detuningRight = spectrumDetuning[argOfMax:]

	specRight = spectrumTransmission[argOfMax:]

	halfMaskBoolean = specRight<HM

	hwRight = detuningRight[halfMaskBoolean][0]

	#ax.axvline(hwRight, color='k')

	

	fwhm = hwRight - hwLeft

	

	return fwhm

	

def calcENBW(spectrumDetuning, spectrumTransmission, targetTransmission=None):

	"""

	Calculate the equivalent noise bandwidth from a set of spectral data

	"""

	# warn if the transmission is not close to zero in the extreme edges of the spectrum

	#if (spectrumTransmission[0] > 1e-3) or (spectrumTransmission[-1] > 1e-3):

	#	print '\nWARNING:\nTransmission not zero at edges of spectrum.\nIt is likely the calculation of ENBW is incorrect!!'

	

	# warn if very few data points for integration

	if len(spectrumDetuning) < 200:

		print('\nWARNING:\nFew data points in the detuning array.\nIt is likely the calculation of ENBW is incorrect!!')

	

	if targetTransmission is None:

		ENBW = integrate.simps(spectrumTransmission, spectrumDetuning) / spectrumTransmission.max()

	else:

		ENBW = integrate.simps(spectrumTransmission, spectrumDetuning) / targetTransmission

	

	return ENBW

	

def fomToverENBW(spectrumDetuning, p_dict, targetDetuning=None):

	"""

	Calculate the equivalent-noise-bandwidth (ENBW) / Peak Transmission figure-of-merit (FOM)

	

	Inputs:

		spectrumDetuning	:	1D array of detuning points for the whole spectrum. Should be a monotonically increasing 

											(or decreasing) set of points.

		p_dict						:	parameter dictionary - passed to ElecSus

	

	Options:

		targetDetuning		:	single detuning point where transmission will be evaluated. Either given as numeric value, 

											or if 	'peak' (default), use the detuning of the peak transmission of the filter profile



	Outputs:

		fom							: 	Single element expressing the figure of merit

		peakDetuning			:	If targetDetuning = 'peak', return the detuning element (from spectrumDetuning) that 

											corresponds to peak transmission. If targetDetuning is anything 

											other than 'peak', peakDetuning == targetDetuning

	"""



	# if targetDetuning is required, add this point on to the spectrumDetuning array to explicitly calculate the transmission at targetDetuning

	if targetDetuning is not None:

		spectrumDetuning = np.append(spectrumDetuning, targetDetuning)

	

	

	# 1. Calculate specturm based on spectrumDetuning and input parameter dictionary

	

	# get (linearly polarised) input electric field

	E_in = np.array([np.cos(p_dict['thetaE']), np.sin(p_dict['thetaE']), 0])

	

	# define Jones matrix for orthogonal output (linear) polariser

	outputAngle = p_dict['thetaE'] + np.pi/2

	J_out = np.matrix( [ [np.cos(outputAngle)**2, np.sin(outputAngle)*np.cos(outputAngle)],

									[np.sin(outputAngle)*np.cos(outputAngle), np.sin(outputAngle)**2] ] )

	

	try:

		[E_out] = em.calculate(spectrumDetuning, E_in, p_dict, outputs=['E_out'])

	except:

		## TO FIX LATER!!!

		# Catch-all for any error that occurs in ElecSus - sometimes the null matrix fails for some reason

		# in this case ignore the error and return zeros

		print '\t\tWARNING: ElecSus error encountered - returning zero FOM value'

		return 0.

		

	# get transmission from electric field

	transmittedE =  np.array(J_out * E_out[:2])

	spectrumTransmission =  (transmittedE * transmittedE.conjugate()).sum(axis=0)



	# if targetDetuning is required, extract this point from the calculated array to explicitly calculate the transmission at targetDetuning

	targetTransmission = None

	if targetDetuning is not None:

		targetTransmission = spectrumTransmission[-1]

		spectrumTransmission = spectrumTransmission[:-1]

		

	# get ENBW

	ENBW = calcENBW(spectrumDetuning, spectrumTransmission, targetTransmission)

	

	# get FOM

	if targetDetuning is not None:

		FOM = targetTransmission / ENBW

	else:

		FOM = spectrumTransmission.max() / ENBW

	

	return FOM.real

	

def testENBW():

	"""

	Test method for checking ENBW calculation. Check it agrees with the sub-GHz Faraday filter paper for Cs D1 line

	(nb: it does)

	"""

	detuning = np.linspace(-40,40,25000)

	p_dict = {'Elem':'Cs', 'Dline':'D1', 'Bfield':45.7, 'T':67.8, 'LCELL':75e-3, 'thetaE':0.0}

	

	[transmissionSpec] = em.calculate(detuning*1e3, [1,0,0], p_dict, outputs=['Iy'])

	

	ENBW = calcENBW(detuning*1e3, transmissionSpec)

	FOM = fomToverENBW(detuning*1e3, p_dict)

	

	print 'ENBW (MHz):', ENBW

	print 'FOM (MHz^-1):', FOM

	

	plt.plot(detuning, transmissionSpec)

	plt.show()



def fomFitFunction(spectrumDetuning, T, Bfield, Btheta, thetaE, 

									p_dict={}, targetDetuning=None, iterationNumber=None, totalIterations=None):

	"""

	Wrapper to fomToverENBW() with explicit parameters to be used with lmfit Model class

	"""

	if iterationNumber is not None:

		print 'Iteration number:\t', iterationNumber+1, ' / ', totalIterations

		print 'T, B, Btheta, thetaE:\t', T, Bfield, Btheta, thetaE

	

	print '.', 

	p_dict['T'] = T

	p_dict['Bfield'] = Bfield

	p_dict['Btheta'] = Btheta

	p_dict['thetaE'] = thetaE

	

	return fomToverENBW(spectrumDetuning, p_dict, targetDetuning=targetDetuning)

	

def fomFitFunctionMPWrapper(args_list):

	""" 

	Wrapper to fomFitFunction() that is compatible with the multiprocessing map() or map_async() methods



	detuning = args_list[0]

	T = args_list[1]

	Bfield = args_list[2]

	Btheta = args_list[3]

	thetaE = args_list[4]

	p_dict = args_list[5]

	targetDetuning = args_list[6]

	iterationNumber = args_list[7]

	"""

	

	return args_list[7], fomFitFunction(*args_list)



def fomCostFunction(params, spectrumDetuning=None, p_dict={},targetDetuning=None):

	""" 

	Wrapper to fomFitFunction that can be used with lmfit's minimize() method

	

	Since we are minimising the cost function, we use the inverse of the FOM defined above

	"""

	# unpack parameters

	parvals = params.valuesdict()

	T = parvals['T']

	Bfield = parvals['Bfield']

	Btheta = parvals['Btheta']

	thetaE = parvals['thetaE']

	

	try:

		cost = 1./fomFitFunction(spectrumDetuning, T,Bfield,Btheta,thetaE, p_dict, targetDetuning)

	except ZeroDivisionError:

		cost = 1e20 # large but not infinite number

	

	return cost

	

def generateSpectrum(spectrumDetuning, p_dict):

	""" 

	Method that calculates a filter spectrum and returns it. 

	Uses ElecSus to calculate the electric field propagation, then adds linear polarisers 

	at a specified angle at the output.

	

	Returns a 1D numpy array of len(spectrumDetuning)

	"""

	print p_dict

	

	outputAngle = p_dict['thetaE'] + np.pi/2

	E_in = np.array([np.cos(p_dict['thetaE']), np.sin(p_dict['thetaE']), 0])

	J_out = np.matrix( [ [np.cos(outputAngle)**2, np.sin(outputAngle)*np.cos(outputAngle)],

								[np.sin(outputAngle)*np.cos(outputAngle), np.sin(outputAngle)**2] ] )



	[E_out, S0] = em.calculate(spectrumDetuning, E_in, p_dict, outputs=['E_out','S0'])



	# get transmission from electric field

	transmittedE =  np.array(J_out * E_out[:2])

	return (transmittedE * transmittedE.conjugate()).sum(axis=0).real



def doSingleOptimisation(guessParamDict, paramBools, paramBounds, 

											detuningParams,

											save_directory, 

											iterationNumber=0, startingIterationNumber=0, 

											showGuess=False):

	""" 

	Main method to run one instance of the minimisation routine. 

	Irritatingly this can't be part of the Optimiser class, as multiprocessing doesn't support it, for reasons I

	still don't quite understand. 

	

	This is called by each process in multiprocessing's map_async() method, but can also be called externally.

	

	Uses lmfit's Minimizer class (https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer) to do the hard work.

	

	Inputs:

	

		guessParamDict:	dictionary of guess parameters that will override the ones stored in the __init__ method

		paramBools:		dictionary - same keys as guessParamDict, but with boolean values, that dictate which 

									parameters are allowed to vary in the fit

		paramBounds:		dictionary - same keys as guessParamDict, but with 2-element lists that specify the min/max 

									values allowed for each parameter in the fit

		save_directory:	directory in which to store the results

		

	Options:

	

		iterationNumber:	int, for naming the save files

		startingIterationNumber: int, offset to iterationNumber for saving files 

												(e.g. on a second run, keep the results of the first run)

		showGuess:		Boolean to plot the guess parameters

		

	Returns:

	

		fom:		Figure of merit for the best-case parameters

		iterationNumber:	iteration number for reference

	

	"""

	

	spectrumDetuning = np.linspace(detuningParams[0], detuningParams[1], detuningParams[2])

	

	# File to save / load optimisation data to / from

	optFn = os.path.join(save_directory,str(iterationNumber+startingIterationNumber)+'_initial.pkl')

	print optFn

	

	#print guessParamDict

	

	# Initialise lmfit parameters class

	params = lm.Parameters()

	params.add('T',value=guessParamDict['T'],\

							min=paramBounds['minT'],max=paramBounds['maxT'],\

							vary=paramBools['T'])

	params.add('Bfield',value=guessParamDict['Bfield'],\

							min=paramBounds['minB'],max=paramBounds['maxB'],\

							vary=paramBools['Bfield'])

	params.add('Btheta',value=guessParamDict['Btheta'],\

							min=paramBounds['minAngleB'],max=paramBounds['maxAngleB'],\

							vary=paramBools['Btheta'])

	params.add('thetaE',value=guessParamDict['thetaE'],\

							min=paramBounds['minAngleE'],max=paramBounds['maxAngleE'],\

							vary=paramBools['thetaE'])

	

	pickle.dump([params, None, None, None, spectrumDetuning, guessParamDict], open(optFn,'wb'))

	#print 'saved...'

	

	if showGuess is True:

		print 'Pausing optimisation to look at guess parameters. Close the plot window to continue.'

		fig = plt.figure("Initial parameter spectrum")

		ax = fig.add_subplot(111)

		#print pd_full

		ax.plot(spectrumDetuning, generateSpectrum(spectrumDetuning, guessParamDict))

		plt.show()

	

	

	## Fitting method: cobyla, leastsq and L-BFGS-B are all local solvers. Differential evolution is global solver, but the results have been a bit hit and miss

	#method = 'leastsq'

	method = 'cobyla'

	#method = 'differential_evolution' # global, but takes forever and often still doesn't return the right result...

	#method = 'L-BFGS-B'

	#method = 'Nelder-Mead'

	

	# Run the fit

	print '\tStarting fit running. Current time: ', time.ctime()

	print '\tRandom restart iteration number: ', iterationNumber

	print '\tUsing method: ', method

	st = time.clock()

	MiniInst = lm.Minimizer(fomCostFunction, params, fcn_kws={'spectrumDetuning':spectrumDetuning, 'p_dict':guessParamDict, 'targetDetuning':None})

	result = MiniInst.scalar_minimize(method=method,params=params)

	

	# print some info on the fit results

	print '\n\tFit Completed. Time elapsed (min): ', (time.clock() - st)/60

	print '\tNumber of function evaluations: ', result.nfev

	print '\tFit successful?:\t', result.success

	

	print '\tBest-case parameters:'

	best_paramvals = result.params.valuesdict()

	for key in best_paramvals:

		print '\t\t',key,':\t', best_paramvals[key]

	

	# update parameter dictionary

	guessParamDict.update(best_paramvals)



	# Calculate best-case FOM

	spectrumDetuning = np.linspace(-60,60,5000)*1e3 # finer grid of points than used to optimise

	fom = fomFitFunction(spectrumDetuning, best_paramvals['T'], best_paramvals['Bfield'], best_paramvals['Btheta'], best_paramvals['thetaE'], guessParamDict)

	

	#Calculate best-case spectrum

	spec = generateSpectrum(spectrumDetuning, guessParamDict)



	# calculate best-case figures of merit

	fwhm = calcFWHM(spectrumDetuning/1e3, spec)

	enbw = calcENBW(spectrumDetuning/1e3, spec)

	

	print '\n\Figures of merit:'

	print '\t\tMax. Transmission:\t', spec.max()

	print '\t\tFWHM (GHz):\t\t', fwhm

	print '\t\tENBW (GHz):\t\t', enbw

	print '\t\tFOM (GHz-1):\t\t', fom*1e3



	# Save calculations to file for further processing and analysis later

	optFn = os.path.join(save_directory,str(iterationNumber+startingIterationNumber)+'_optimised.pkl')

	pickle.dump([result.params, best_paramvals, fom, spec, spectrumDetuning, guessParamDict], open(optFn,'wb'))

	

	print '\nCalculation saved to pickle file\n\n----------------------------------------------------------'

	

	return fom, iterationNumber



def doSingleOptimisationMP(args_list):

	""" Multiprocessing-compatible wrapper for doSingleOptimisation() """

	guessParamDict, paramBools, paramBounds, detuningParams, \

		save_directory, iterationNumber, startingIterationNumber  = args_list

	

	time.sleep(np.random.random()*2) # pause for <= 2 second to allow initial print statements to be printed properly from separate processes - only matters at the start when all processes spawn simultaneously

	return doSingleOptimisation(guessParamDict, paramBools, paramBounds, detuningParams,

											save_directory, iterationNumber, startingIterationNumber)

	

class Optimiser():

	""" Main class to set up and do the optimisation """

	

	def __init__(self, ELEM, DLINE, LCELL, Rb85abundance='NAT'):

		"""

		Initialisation housekeeping - create directory tree for saving/loading outputs if it doesn't exist already

		"""

		

		self.ELEM = ELEM

		self.DLINE = DLINE

		self.LCELL = LCELL

		self.Rb85abundance = Rb85abundance

		# set a label to save data to - catch the case where we deal with different isotopic abundances of Rb

		if ELEM == 'Rb':

			if Rb85abundance == 'NAT':

				self.ELEM_LABEL = 'Rb'

			else:

				self.ELEM_LABEL = str(Rb85abundance)+'Rb85'

		else:

			self.ELEM_LABEL = ELEM

		

		# Define the directory to save results to

		self.save_directory = 'optimised_data\\'+self.ELEM_LABEL+'\\'+self.DLINE+'\\'+str(round(LCELL*1e3,1))+'mm'

		

		# Make sure that directory exists...

		# make directories for saved optimisation data (save result from every random restart seed)

		# try / except to catch if the directory already exists

		try:

			os.makedirs(self.save_directory)

		except:

			print 'Error occurred when trying to make save directory - probably the directory already exists (no action needed)'

			

		

	def runRandomRestartOptimisation(self, startingIterationNumber=0):

		"""

		Method to set up and run the Random Restart optimisation - create pool of workers and spawn random guesses in parameter space, then set the pool going over all processors available. This should be relatively easy to farm out to multiple computers.

		

		After the initial parameter search, select the best ones and use these as starting parameters for a set of local optimisers.

		"""



		# Number of random seeds (starting coordinates in parameter space)

		self.nSeeds = 1500

		self.optSeeds = 150

		

		# Min/max for each parameter (needed for Differential evolution solver, and to give range on random starting parameters in RR)

		self.minB = 10

		self.maxB = 1300

		self.minT = 40

		self.maxT = 230

		self.minAngleB = 0 # - np.pi/40 # 5% overshoot range

		self.maxAngleB = np.pi/2 # + np.pi/40

		self.minAngleE = 0 # - np.pi/40

		self.maxAngleE = np.pi/2 #+ np.pi/40

		

		# group the above into a dictionary

		self.fitBounds = {'minB':self.minB, 'maxB':self.maxB, 'minT':self.minT, 'maxT':self.maxT, 'minAngleB':self.minAngleB, 'maxAngleB':self.maxAngleB, 'minAngleE':self.minAngleE, 'maxAngleE':self.maxAngleE}

		

		# Initial parameters (if they aren't selected randomly)

		self.startingB = 1000.

		self.startingT = 120.

		self.startingAngleB = 88. * np.pi/180

		self.startingAngleE = np.pi/10

		

		# Fixed parameters

		self.p_dict = {'lcell':self.LCELL, 'Elem':self.ELEM, 'Dline':self.DLINE}

		if self.Rb85abundance == 'NAT':

			self.Rb85abundance = 72.17 #%



		# Calculation range (laser frequency detuning)

		detRange = 25e3

		detPoints = 1000

		#self.spectrumDetuning = np.linspace(-detRange,detRange,1500) * 1e3

		

		# Which parameters are varying in the fit

		self.fitBools = {'T':True, 'Bfield':True, 'Btheta':True, 'thetaE':True}

		

		

		# Generate random sets of starting parameters

		if self.fitBools['T']:

			startingTs = np.random.uniform(self.minT, self.maxT, self.nSeeds)

		else:

			startingTs = np.ones(self.nSeeds)*self.startingT

		if self.fitBools['Bfield']:

			startingBs = np.random.uniform(self.minB, self.maxB, self.nSeeds)

		else:

			startingBs = np.ones(self.nSeeds)*self.startingB

		if self.fitBools['Btheta']:

			startingBthetas = np.random.uniform(self.minAngleB, self.maxAngleB, self.nSeeds)

		else:

			startingBthetas = np.ones(self.nSeeds)*self.startingAngleB

		if self.fitBools['thetaE']:

			startingthetaEs = np.random.uniform(self.minAngleE, self.maxAngleE, self.nSeeds)

		else:

			startingthetaEs = np.ones(self.nSeeds)*self.startingAngleE



			

		#

		## 1.  Filter grid of initial coordinates; choose subset of the nSeeds based on best FOM

		#

		

		

		# Create pool of child processes

		pool = mp.Pool()

		

		# Use psutil module to lower process priority so computer is still responsive while calculating!

		parent = psutil.Process()

		parent.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

		for child in parent.children():

			child.nice(psutil.IDLE_PRIORITY_CLASS)

										

		# Create args_list to pass to map_async method

		detuning = np.linspace(-detRange,detRange,detPoints)

		args_list = [[detuning, startingTs[i], startingBs[i], startingBthetas[i], startingthetaEs[i], \

								{'Elem':self.ELEM, 'Dline':self.DLINE, 'lcell':self.LCELL, \

								'rb85frac':self.Rb85abundance}, None, i, self.nSeeds] for i in range(self.nSeeds)]



		# Run multiprocessing loop

		output = pool.map_async(fomFitFunctionMPWrapper, args_list)

		

		# Close pool and join - wait for processes to finish before continuing

		pool.close()

		pool.join()		

		

		result = output.get()		

		# Sort results by FOM value

		resultSorted = sorted(result, key=lambda f: f[1])[::-1] # [::-1] ==> largest to smallest

		

		# Use best intial params to optimise with

		selectedIndices = np.array(resultSorted,dtype=int).T[0][:self.optSeeds]

		startingFOMs = np.array(resultSorted).T[1][:self.optSeeds]

		

		print '\n------------------------------------------------------------------------'

		print '\nInitial parameter search completed. Filtered down to '+str(self.optSeeds)+' sets of seed parameters.'

		print 'Initial FOMs are:'

		for fom in startingFOMs: print '\t',fom*1e3

		print 'Starting optimisation routines...\n'

	

		

		

		#

		## 2.  Use filtered grid of starting parameters to run optimisation routines

		#

		

		

		# Create pool of child processes

		pool = mp.Pool()

		

		# Use psutil module to lower process priority so computer is still responsive while calculating!

		parent = psutil.Process()

		parent.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

		for child in parent.children():

			child.nice(psutil.IDLE_PRIORITY_CLASS)

										

		# Create args_list to pass to map_async method

		args_list = [[{'T':startingTs[selectedIndices[i]], 'Bfield':startingBs[selectedIndices[i]], \

								'Btheta':startingBthetas[selectedIndices[i]], 'thetaE':startingthetaEs[selectedIndices[i]], \

								'Elem':self.ELEM, 'Dline':self.DLINE, 'lcell':self.LCELL, 'rb85frac':self.Rb85abundance}, \

								self.fitBools, self.fitBounds, \

								[-detRange, detRange, detPoints], \

								self.save_directory, i, startingIterationNumber] for i in range(len(selectedIndices))]



		# Run multiprocessing loop

		output = pool.map_async(doSingleOptimisationMP, args_list)

		

		# Close pool and join - wait for processes to finish before continuing

		pool.close()

		pool.join()

				

		# get results back ...

		print 'Output: \n', output.get()

		

		

	def loadOptimisationIteration(self,iterationNumber):

		""" Load one RR result from pickle file, and plot the spectrum """

		

		self.optFn = os.path.join(self.save_directory,str(iterationNumber)+'.pkl')

		params, best_paramvals, fom, spec, spectrumDetuning, p_dict = pickle.load(open(self.optFn,'rb'))

		

		print 'Best parameters:'

		print best_paramvals

		

		# analyse spectrum for FWHM, ENBW etc

		fwhm = calcFWHM(spectrumDetuning/1e3, spec)

		enbw = calcENBW(spectrumDetuning/1e3, spec)

		maxTransmission = spec.max()

		# make plot

		fig = plt.figure(figsize=(6,4.5))

		ax = fig.add_subplot(111)

		

		ax.plot(spectrumDetuning/1e3, spec)

		

		ax.text(0.03,0.92, 'Maximum transmission: '+str(round(maxTransmission,2)), transform=ax.transAxes, ha='left')

		ax.text(0.03,0.86, 'ENBW (GHz): '+str(round(enbw,2)), transform=ax.transAxes, ha='left')

		ax.text(0.03,0.8, 'FWHM of main peak (GHz): '+str(round(fwhm,3)), transform=ax.transAxes, ha='left')

		ax.text(0.03,0.74, 'FOM (GHz$^{-1}$): '+str(round(fom*1e3,2)), transform=ax.transAxes, ha='left')

		

		ax.set_xlim(spectrumDetuning[0]/1e3, spectrumDetuning[-1]/1e3)

		

		ax.set_xlabel('Detuning (GHz)')

		ax.set_ylabel('Filter transmission')

		

		plt.tight_layout()

		

		plt.show()

		

	def loadAllOptimisationResults(self):

		""" Load all RR results from pickle files, and plot the spectra """

		

		pkl_fns = glob.glob(os.path.join(self.save_directory,'*_optimised.pkl'))

		print pkl_fns



		# make plot

		fig = plt.figure(figsize=(6,4.5))

		ax = fig.add_subplot(111)

		#ax2 = fig.add_subplot(212,sharex=ax)



		# Get FOM results

		foms = []

		for pkl_fn in pkl_fns:

			params, best_paramvals, fom, spec, spectrumDetuning, p_dict = pickle.load(open(pkl_fn,'rb'))	

			foms.append(fom*1e3)

			

		# Sort results by FOM in decreasing (large to small) order	

		z = zip(foms, pkl_fns)

		zs = sorted(z, key=lambda f: f[0])[::-1]

		

		# print them

		for fom, pkl_fn in zs:

			print fom, pkl_fn



		# plot the best 4 FOM results with sensible FOM values (weird results where transmission > 1 excluded...)

		foms, pkl_fns = zip(*zs)

		foms = np.array(foms)

		pkl_fns = np.array(pkl_fns)[foms<5]

		foms = foms[foms<5]

		print 'Cropped:', foms

		

		print pkl_fns

		# plot 4 best cases

		for pkl_fn in pkl_fns[:4]:

			params, best_paramvals, fom, spec, spectrumDetuning, p_dict = pickle.load(open(pkl_fn,'rb'))

			ax.plot(spectrumDetuning, spec, label='Iteration number: '+pkl_fn[:2])		

		

		# scale and show plot

		plt.tight_layout()

		plt.show()



def calcAndShowFilterProfile():

	""" Calculate a filter with parameters in p_dict. Plot the profile and print figures of merit """

	

	detuning = np.linspace(-30,30,4000)

	

	p_dict = {'Elem':'Rb', 'Dline':'D2', 'lcell':5e-3, 'T':123.988, 'Bfield':232.70, 'Btheta':1.4282, 'thetaE':0.05347}

	spec = generateSpectrum(detuning*1e3, p_dict)

	

	fwhm = calcFWHM(detuning, spec)

	enbw = calcENBW(detuning, spec)

	maxTransmission = spec.max()

	fom = fomToverENBW(detuning*1e3, p_dict) * 1e3

	

	# make plot

	fig = plt.figure(figsize=(6,4.5))

	ax = fig.add_subplot(111)

	

	ax.plot(detuning, spec,color='C1')

	

	ax.text(0.03,0.92, 'Maximum transmission: '+str(round(maxTransmission,2)), transform=ax.transAxes, ha='left')

	ax.text(0.03,0.86, 'ENBW (GHz): '+str(round(enbw,2)), transform=ax.transAxes, ha='left')

	ax.text(0.03,0.8, 'FWHM of main peak (GHz): '+str(round(fwhm,3)), transform=ax.transAxes, ha='left')

	ax.text(0.03,0.74, 'FOM (GHz$^{-1}$): '+str(round(fom,3)), transform=ax.transAxes, ha='left')

	

	ax.set_xlim(detuning[0], detuning[-1])

	

	ax.set_xlabel('Detuning (GHz)')

	ax.set_ylabel('Filter transmission')

	

	plt.tight_layout()

	

	plt.show()

	

if __name__ == '__main__':

	#calcAndShowFilterProfile()



	Opt = Optimiser('Rb', 'D2', 5e-3)

	Opt.runRandomRestartOptimisation()

	Opt.loadAllOptimisationResults()

	

	#Opt.loadOptimisationIteration(0)

	#Opt.loadOptimisationIteration(1)

	#Opt.loadOptimisationIteration(2)

	#Opt.loadOptimisationIteration(3)