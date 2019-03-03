# -*- coding: utf-8 -*-
"""
This is a file that stores the parameters that initialize the genetic algorithm

for spectral decomposition. When the main GA module runs, it imports this

module, which contains links to files containing observed data and reference

spectra. This also contains parameters for axes limits, gaussian functions used

in the evolution cycles, and model constraints.

Author: Vincent Oteyi Kim, PhD, Optoelectronics Group, University of Cambridge
"""

# Observed spectral data

# Set the file name of the data you want to use. The TA data file should be a
# tab-delimited file. The first column should contain the wavelength/energy
# axis values, the first row should contain the time axis values, and the
# values in the matrix should be intensity or DT/T values.
TA_data_filename = "observed_spectrum.txt"
#TA_data_filename = r"C:\Users\Vincent\Documents\Python Scripts\Projects\Genetic Algorithm\SHex\SHex_CN_TS_solutution_540nm1520nm_combined_tab_delim.txt"


# Preparing data for optimization cycles.

# Express spectra using wavelength (nm) or energy (eV) using "wl" or "E".
wl_or_E = "wl"
#
# Set the cutoff wavelength/energy and time. If the value is out of bounds,
# automatically uses the first/last value in the vector
#
t_min = 0  # Start time
t_max = 1700  # End time
x_min = 830  # Minimum value in wavelength/energy
x_max = 960  # Maximum value in wavelength/energy
#
# Subsample the fitted spectra to reduce computational time and, most
# importantly, ensure the algorithm doesn't overfit pixel-to-pixel noise.
# The x axis is divided by the subsampling factor to process few points.
ss_factor = 10
#
# Controls the size of the box of points used for uncertainty calculations of
# each intensity or DT/T data point.
n = 2


# Initial guess and reference files.

# Set the file locations of the spectra that will be modified by the algorithm.
# Use external files as initial guess? Yes(1) / No(0)
#   Random noise/gaussians can still be added whether an external file is used
#   or not. Use init_spe_noise_amplitutde = 0 if no noise is needed, and
#   gauss_prm[1] = 0 if no random gaussians are needed.
#
use_initial_guess = 1
#
# Set the number of spectra to optimize.
num_spectra_to_optimize = 1
#
# Each file must contain axes in the right units. The first column must be
# either wavelegth or energy and the second column either intensity or DT/T).
#initial_guess_filenames = [r"C:\Users\Vincent\Documents\Python Scripts\Projects\Genetic Algorithm\Exciton.txt", r"C:\Users\Vincent\Documents\Python Scripts\Projects\Genetic Algorithm\Polaron.txt"] #{'init_1.txt'};
initial_guess_filenames = [r"References\spectrum1.txt"]
#initial_guess_filenames = []
#
# Set the reference spectra file location
#  These spectra are 'fixed' and will be not modified by the algorithm.
#  If no references are needed, use "reference_filenames = []".
#  Each file must contain axes in the right units. The first (x) column must be
#  the wavelegth or energy and the second column the intensity or DT/T).
# reference_filenames = []
reference_filenames = [r"References\spectrum2.txt"]


# Set initialization parameters for the GA optimization process.
#
#   population_size: The (even) number of individuals in the population
#
#   num_runs: The number of runs of the evolution cycle.
#
#   init_spe_noise_amplitutde: Amplitude of noise added to the spectra
#
#   init_spe_noise_offset: Offset of isn with respect to it's mean (0.5)
#
#   gauss_prm: Parameters of the gaussians used to generate initial spectra
#     [Amplitude, Mean, Width]. Mean and Width are fraction of the total
#     number of points. For example, if the wavelength vector is from 500nm
#     to 900nm and there are 100 points, Width=0.5 corresponds to 50 points
#     (or (900 nm-500 nm) * 0.5 = 200 nm).
#
#   num_gauss: Number of gaussians used to generate initial spectra
#
#   mut.sms: Small mutations strenght of the guassian amplitude
#
#   elitism: The (even) number of best spectra kept without breeding.
#
#   con.nkp: Negative kinetic penalty (constrain), this is a correction
#     applied to the residual to penalize negative kinetics by:
#     1 + negative_kinetic_penalty * (fraction of negative points)
#     Use nkp = 0 if the sign of the spectrum is expected to change.
#
#   species_constraint: Spectrum sign constraints that must be defined for
#     each spectrum. Must be a list whose length equals the number
#     of optimized spectra (e.g. [0, 1, -1]).
#     [0] no constrain
#     [1] positive sectra only (TA features such as GSB,SE)
#     [-1] negative spectra only (TA features such as PIA)

population_size = 100  # The number of individuals (models) in the population.

# Gaussian parameters
init_spe_noise_amplitutde = 0.3  # Set to zero if no noise is needed.
init_spe_noise_offset = 0.5
gauss_prm = [1, 1, 0.25]
num_gauss = 3  # This should be the total number of expected peaks.
sms = 0.05
elitism = 2

# Sign constraints
negative_kinetic_penalty = 5
species_constraint = [1, 1]

# Iteration
num_runs = 1  # The number of runs of the evolution cycle within an iteration.
num_iterations = 1  # Number of iterations the entire GA is performed.

# Dampen the random noise added to the initial guess at the beginning of
# each iteration. The red_noise value is the dampening factor.
red_noise = 1.2
