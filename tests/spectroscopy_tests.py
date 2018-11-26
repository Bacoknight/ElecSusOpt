# Copyright 2017 J. Keaveney

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" 
A series of examples, calculating various spectra for different parameter regimes

Last updated 2018-02-19 JK
"""
# py 2.7 compatibility
from __future__ import (division, print_function, absolute_import)

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks as peaks
from scipy.integrate import simps as integrate
from mpl_toolkits.mplot3d import axes3d, Axes3D

import time
import sys

from elecsus.elecsus_methods import calculate as get_spectra
from elecsus.libs.spectra import calc_chi as get_chi

# Global constants
centreFreqR87 = (380.6685) * 1e6 # MHz. For rubidium only!
speedLight = 3e8
centreKR87 = 2 * np.pi/(speedLight/centreFreqR87)

def FourStepGraph():
	""" 
	For a given setup, creates the four graphs used to produce the final result of ElecSus.
	Used to visualise the 'thought process' of ElecSus. The four graphs are:
	- Stick graph showing transition frequencies.
	- Real and imaginary parts of Chi, used to determine the complex refractive index.
	- The resultant phase shift in the outgoing light.
	- The transmission.
	"""
	# Define the frequency range to be inspected in MHz.
	detuning = np.arange(-100000,100000,10)

	# Define the experimental parameters.
	x = [733.00326761, 23.62413798, 97.07244311]
	p_dict = {'Bfield':x[0], 'rb85frac':0, 'Btheta':x[1], 'lcell':75e-3, 'T':x[2], 'Dline':'D2', 'Elem':'Rb'}
	
	# Get the intensity of light perpendicular to the input light (assuming the setup has two crossed polarisers surrounding the filter).
	# [1,0,0] means the incoming light has an E field aligned along the x-axis only.
	spectrumOutput = get_spectra(detuning, [1,0,0], p_dict, outputs=['Iy', 'ChiPlus', 'ChiMinus'])

	Iy = spectrumOutput[0]

	# Extract the sigma plus transition information first.
	absorptionValsPlus = np.imag(spectrumOutput[1])
	phaseShiftValsPlus = np.real(spectrumOutput[1])

	# Note, the following code will not work on Hamilton as it requires a later version of SciPy than is currently installed.
	peakIndicesPlus = peaks(absorptionValsPlus)[0]
	peakFreqPlus = detuning[peakIndicesPlus]
	peakAlphaPlus = absorptionValsPlus[peakIndicesPlus]

	# Now extract sigma minus transition information.
	absorptionValsMinus = np.imag(spectrumOutput[2])
	phaseShiftValsMinus = np.real(spectrumOutput[2])
		
	# Note, the following code will not work on Hamilton as it requires a later version of SciPy.
	peakIndicesMinus = peaks(absorptionValsMinus)[0]
	peakFreqMinus = detuning[peakIndicesMinus]
	peakAlphaMinus = absorptionValsMinus[peakIndicesMinus]
	deltaN = np.array(phaseShiftValsPlus - phaseShiftValsMinus)
	phaseArray = deltaN * 75e-3 * np.pi * np.add(detuning, centreFreqR87)/speedLight

	# Set up the plot.
	fig = plt.figure("Transmission buildup plot")

	ax1 = plt.subplot(4,1,1)
	ax2 = plt.subplot(4,1,3, sharex=ax1)
	ax3 = plt.subplot(4,1,4, sharex=ax2)
	ax4 = plt.subplot(4,1,2, sharex=ax3)
	ax5 = ax4.twinx()

	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.setp(ax2.get_xticklabels(), visible=False)
	plt.setp(ax3.get_yticklabels(), visible=False)
	plt.setp(ax4.get_xticklabels(), visible=False)
		
	ax1.plot(detuning/1e3, Iy, '-', color='navy', lw=2.5)
	ax2.plot(detuning/1e3, absorptionValsPlus, 'r--', color='c', lw=2, label=r'$\sigma^{+}$')
	ax2.plot(detuning/1e3, phaseShiftValsPlus, color='c', lw=2)
	ax4.plot(detuning/1e3, deltaN, color='navy')
	ax5.plot(detuning/1e3, phaseArray, color='red')

	#Note, the following code will not work on Hamilton as it requires a later version of SciPy.
	markerline, stemlines, baseline = ax3.stem(peakFreqPlus/1e3, peakAlphaPlus, '-.', label=r'$\sigma^{+}$')
	
	plt.setp(markerline, alpha=0)
	plt.setp(stemlines, color='c')
	plt.setp(baseline, linewidth=0)

	ax2.plot(detuning/1e3, absorptionValsMinus, 'r--', color='m', lw=2, label=r'$\sigma^{-}$')
	ax2.plot(detuning/1e3, phaseShiftValsMinus, color='m', lw=2)

	#Note, the following code will not work on Hamilton as it requires a later version of SciPy.
	markerline, stemlines, baseline = ax3.stem(peakFreqMinus/1e3, peakAlphaMinus, '-.', label=r'$\sigma^{-}$')
	
	plt.setp(markerline, alpha=0)
	plt.setp(stemlines, color='m')
	plt.setp(baseline, linewidth=0)
	
	ax3.set_xlabel('Detuning (GHz)')
	ax1.set_ylabel('Transmission')
	ax2.set_ylabel(r'$n_{Im}, n_{Re}$')
	ax5.set_ylabel('Phase shift (Radians)')
	ax3.set_ylabel('Relative Transition Strength')
	ax4.set_ylabel(r'$n_{+, Re} - n_{-, Re}$')

	ax2.legend(loc="upper center")
	ax3.legend(loc="upper center")
	
	ax2.set_xlim(-100,100)
	ax3.set_ylim(0)
	ax1.set_ylim(0)

	ENBW = integrate(Iy, x=detuning/1e3)/np.amax(Iy)
	FOM = np.amax(Iy)/ENBW

	print("-----------------")
	print(str(p_dict["Elem"]) + ", " + str(p_dict["Dline"]) + " line:")
	print("-----------------")
	print("Properties:")
	print("B Field Strength (Gauss): " + str(p_dict["Bfield"]))
	print("Temperature (Celcius): " + str(p_dict["T"]))
	print("Angle (Degrees): " + str(p_dict["Btheta"]))
	print("-----------------")
	print("Results:")
	print("Max Transmission: " + str(np.amax(Iy)))
	print("ENBW: " + str(ENBW))
	print("FOM: " + str(FOM))
	print("-----------------")
	
	plt.show()
	
def OptSurfaceGraph():
	"""
	For a given element and magnetic field angle, plots a surface of figure of merit, magnetic field and temperature to show the area of interest.
	"""
	
	# Define the frequency range to be inspected in MHz.
	detuning = np.arange(-8000,8000,25)
	
	# Define the range of magnetic field strengths and temperatures to test.
	tRange = np.linspace(-100, 100, num=20)
	bRange = np.linspace(0, 1e4, num=20)

	T, B = np.meshgrid(tRange, bRange)
	numEvals = 0
	Z = []

	for t in tRange:
		FoMList = []
		for b in bRange:
			numEvals += 1
			if numEvals % 1000 == 0:
				print("Number of evaluations: " + str(numEvals))
			p_dict = {'Bfield':b, 'rb85frac':0, 'Btheta':23.62413798, 'lcell':75e-3, 'T':t, 'Dline':'D2', 'Elem':'Rb'}
			[Iy] = get_spectra(detuning, [1,0,0], p_dict, outputs=['Iy'])
			ENBW = integrate(Iy, x=detuning/1e3)/np.amax(Iy)
			FoMList.append(np.amax(Iy)/ENBW)
		
		Z.append(FoMList)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.plot_surface(T, B, np.array(Z), cmap='nipy_spectral')

	plt.show()

	return

if __name__ == '__main__':
	print('Running Test Cases...')
	FourStepGraph()
	OptSurfaceGraph()