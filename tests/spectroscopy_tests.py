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
plt.rc("text", usetex = True)
import numpy as np
from scipy.signal import find_peaks as peaks
from scipy.integrate import simps as integrate
from mpl_toolkits.mplot3d import axes3d, Axes3D

import time
import sys

from elecsus.elecsus_methods import calculate as get_spectra
from elecsus.libs.spectra import calc_chi as get_chi

import seaborn as sns
sns.set_context("talk")
sns.set_style("ticks")

# Global constants
centreFreqR87 = (508.84871) * 1e6 # MHz. For sodium only!
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
	detuning = np.linspace(-25000, 25000, 20000)

	# Define the experimental parameters.
	p_dict = {'Bfield':144, 'rb85frac':72.17, 'Btheta':0, 'lcell':5e-3, 'T':245, 'Dline':'D2', 'Elem':'Na', 'Etheta': 0}
	
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
	phaseArray = deltaN * centreKR87 * 1e6/(2 * np.pi * 1e3) # Units of pi * 1e3, meaning integers have the worst output, whereas half-integers have the best.

	# Set up the plot.
	fig = plt.figure("Transmission buildup plot")
	fig.set_size_inches(19.20, 10.80)

	ax1 = plt.subplot(224)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(221)
	ax4 = plt.subplot(223)
	ax5 = ax4.twinx()

	# plt.setp(ax1.get_xticklabels(), visible=False)
	# plt.setp(ax2.get_xticklabels(), visible=False)
	plt.setp(ax3.get_yticklabels(), visible=False)
	# plt.setp(ax4.get_xticklabels(), visible=False)
	ax3.yaxis.set_ticks_position("none")
		
	ax1.plot(detuning/1e3, Iy*100, '-', color='navy', lw=2.5)
	ax2.plot(detuning/1e3, absorptionValsPlus * 1e3, 'r--', color='c', lw=2, label=r'$\sigma^{+}$')
	ax2.plot(detuning/1e3, phaseShiftValsPlus * 1e3, color='c', lw=2)
	ax4.plot(detuning/1e3, deltaN * 1e4, color='navy')
	ax5.plot(detuning/1e3, phaseArray * 10, color='navy')
	#ax5.axhline(2, color='red', label=r'$\phi = \pi$')
	#ax5.axhline(1.0005, color='green', label=r'$\phi = \frac{\pi}{2}$')

	#Note, the following code will not work on Hamilton as it requires a later version of SciPy.
	markerline, stemlines, baseline = ax3.stem(peakFreqPlus/1e3, peakAlphaPlus * 1e3, '-.', label=r'$\sigma^{+}$')
	
	plt.setp(markerline, alpha=0)
	plt.setp(stemlines, color='c')
	plt.setp(baseline, linewidth=0)

	ax2.plot(detuning/1e3, absorptionValsMinus * 1e3, 'r--', color='m', lw=2, label=r'$\sigma^{-}$')
	ax2.plot(detuning/1e3, phaseShiftValsMinus * 1e3, color='m', lw=2)

	#Note, the following code will not work on Hamilton as it requires a later version of SciPy.
	markerline, stemlines, baseline = ax3.stem(peakFreqMinus/1e3, peakAlphaMinus * 1e3, '-.', label=r'$\sigma^{-}$')
	
	plt.setp(markerline, alpha=0)
	plt.setp(stemlines, color='m')
	plt.setp(baseline, linewidth=0)
	
	ax3.set_xlabel(r'$\Delta$ (GHz)')
	ax1.set_xlabel(r'$\Delta$ (GHz)')
	ax2.set_xlabel(r'$\Delta$ (GHz)')
	ax4.set_xlabel(r'$\Delta$ (GHz)')
	ax1.set_ylabel(r'Transmission (\%)')
	ax2.set_ylabel(r'$\Re(n) \cdot 10^{3}, \Im(n) \cdot 10^{3}$')
	ax5.set_ylabel(r'Linear polarisation rotation ($100 \pi$)')
	ax3.set_ylabel('Relative Transition Strength')
	ax4.set_ylabel(r'$\Re(n_{\textrm{+}} - n_{\textrm{-}}) \cdot 10^{4}$')

	ax2.legend(loc="best")
	ax3.legend(loc="best")
	#ax5.legend(loc="best")
	
	ax2.set_xlim(detuning[0]/1e3, detuning[-1]/1e3)
	ax1.set_xlim(detuning[0]/1e3, detuning[-1]/1e3)
	ax3.set_xlim(detuning[0]/1e3, detuning[-1]/1e3)
	ax4.set_xlim(detuning[0]/1e3, detuning[-1]/1e3)
	ax3.set_ylim(0)
	ax1.set_ylim(0)

	plt.tight_layout()

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

	figName = str(p_dict["Elem"]) + "_" + str(p_dict["Dline"]) + "_" + str(p_dict["Bfield"]) + "_" + str(p_dict["T"]) + "_" + str(p_dict["Btheta"]) + ".pdf"
	plt.savefig(figName)
	print("Image saved as: " + figName)

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

def ProduceSpectrum(detuning, params, toPlot = True):
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
		[E_out] = get_spectra(detuning, E_in, params, outputs=['E_out'])
	except:
		# There was an issue obtaining the field from ElecSus.
		return 0

	transmittedE =  np.array(J_out * E_out[:2])
	transmission =  (transmittedE * transmittedE.conjugate()).sum(axis=0)

	if toPlot:
		# Plot the result.
		plt.plot(detuning/1e3, transmission * 100)
		plt.show()

	return transmission

if __name__ == '__main__':
	print('Running Test Cases...')
	FourStepGraph()
	#OptSurfaceGraph()

	# Define the frequency range to be inspected in MHz.
	#globalDetuning = np.linspace(-25000, 25000, 1000)

	# Define the experimental parameters.
	# p_dict = {'Bfield':144, 'rb85frac':72.17, 'Btheta':0, 'lcell':5e-3, 'T':245, 'Dline':'D2', 'Elem':'Na', 'Etheta': 0}
	# globalDetuning = np.linspace(-25000, 25000, 20000)
	# print(ProduceSpectrum(globalDetuning, p_dict, True))