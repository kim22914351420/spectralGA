# -*- coding: utf-8 -*-
"""
This module provides an environment to analyze spectral data. It allows the

user to observe an evolving spectrum over time at various wavelengths and

intensities. Futhermore, it uses a genetic algorithm to decompose the evolving

spectrum into component spectra that overlap in time and wavelength.


The main motivation of this module is to analyze data from spectroscopy

experiments. Results from this module inform the fundamental physics of how

light interacts with matter, which have continued to directly impact the design

of more efficient solar panels and light-emitting diodes (LEDs).


This module is based on an excellent genetic algorithm program written in

MATLAB by Simon Gelinas during his PhD in the Optoelectronics Group at the

University of Cambridge. The program is discussed in his thesis, and it is a

wonderful resource to the group. Many thanks to Florian Schroeder and Callum

Ward for their valuable contributions to said MATLAB program.


Author: Vincent Oteyi Kim, PhD, Optoelectronics Group, University of Cambridge
"""
import os  # For creating directories for results
import numpy as np  # For manipulating arrays
from scipy import interpolate, signal  # For creating cutoff regions, using
# reference spectra, and filtering data

import matplotlib.pyplot as plt  # For visualization
import seaborn as sns  # For visualization

import parameters as prm  # For easily tuning parameters in separate file
import sys  # For flagging improper reference files.

import time  # For displaying the progress of the genetic algorithm
import datetime

# import spectralLandscapeGenerator as slg


class TAData:
    """Allows the user to interact with spectral data. In practice, these are

    normally results from transient absorption spectroscopy (TAS or TA), which

    is a technique employed extensively in the field of optoelectronics (how

    electronic devices interact with light).
    """

    def __init__(self):

        # Load tab-delimited TA data with the wavelength and time headers.
        TA = np.loadtxt(prm.TA_data_filename, delimiter="\t")

        # The wavelength (nm) or energy (eV) scale
        # where energy = 1240 nm / wavelength
        self.wl_values = TA[1:, 0]
        self.E_values = np.multiply(np.reciprocal(TA[1:, 0]), 1240)

        # The timescale (usually fs, ps, or ns depending on the experiment)
        self.t_values = TA[0, 1:]

        # The DT/T values at the different wavelengths and times
        self.TA_values = TA[1:, 1:] * -1

        # self.wl_values = slg.x ########for demo
        # self.t_values = slg.t
        # self.TA_values = slg.TA_values

    def get_values(self, values):
        """Returns the wavelength scale, the timescale, or intensity values.

        For transient absorption spectroscopy, the third set has DT/T values.
        """

        if values == "wl":
            return self.wl_values

        if values == "E":
            return self.E_values

        if values == "t":
            return self.t_values

        if values == "TA":
            return self.TA_values

        else:
            print("""Please use either "wl", "t", or "TA" as an argument when
                  calling the get_values method.""")


class GeneticAlgorithm(TAData):
    """Represents a genetic algorithm for optimizing the reconstruction of

    observed spectral (TA) data.
    """

    def __init__(self):

        TAData.__init__(self)

        # Make a time-stamped directory for each instance of the algorithm.
        self.directory = os.path.join(
                os.getcwd(),
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        os.makedirs(self.directory)

        # Save and access a list of all values from the parameter file.
        self.metadata = self.save_metadata()

        # The TA data is subsampled to avoid optimizing noise.
        self.ss_factor = prm.ss_factor
        self.ss_region = self.wl_values[0:len(self.wl_values):self.ss_factor]

        # Parameters of gaussians functions used to generate the initial
        # guess spectra
        self.gauss_prm = prm.gauss_prm
        self.init_spe_noise_amplitutde = prm.init_spe_noise_amplitutde

        # Choose axis units of either wavelength or energy.
        if prm.wl_or_E == "wl":
            self.x = self.wl_values

        if prm.wl_or_E == "eng":
            self.x = self.E_values

    def load_initial_guesses_and_reference_spectra(self):
        """Loads the initial guesses and reference spectra from text files.

        These spectra are provided by the user based on theory or previous

        experiments. Initial guesses are modified during optimization whereas

        references are not modified.
        """

        # Create a matrix whose rows hold initial guess spectra and
        # other spectra to optimize whose columns are different wavelengths.
        # Add extra spectra to optimize if there are more expected features
        # than initial guess spectra.

        if prm.num_spectra_to_optimize > len(prm.initial_guess_filenames):
            self.initial_guess_spectra = np.zeros((prm.num_spectra_to_optimize,
                                                   len(self.ss_region)
                                                   )
                                                  )
        else:
            self.initial_guess_spectra = \
             np.zeros((len(prm.initial_guess_filenames), len(self.ss_region)))

        # If the user chose to use initial guess or reference files,
        # check if files exist and are formatted correctly.

        if prm.use_initial_guess:

            if not prm.initial_guess_filenames:
                sys.exit("""User chose to use initial guess files but did not
                         provide any. Program will stop.""")

            else:

                for i in range(len(prm.initial_guess_filenames)):

                    guess_spectrum = np.loadtxt(prm.initial_guess_filenames[i],
                                                delimiter="\t").T

                    if np.ma.size(guess_spectrum, 0) != 2:

                        sys.exit("""Initiation file not valid. Data should be
                                 an n by 2 matrix. Program will stop.""")

                    # Interpolate the initialization file at each subsampled
                    # region (self.ss_region).

                    init_guess_spline_func = \
                        interpolate.interp1d(guess_spectrum[0, :],
                                             guess_spectrum[1, :],
                                             fill_value="extrapolate"
                                             )
                    self.initial_guess_spectra[i, :] = \
                        init_guess_spline_func(self.ss_region)

        np.savetxt("initial_guess_spectra.txt",
                   self.initial_guess_spectra,
                   delimiter="\t"
                   )

        # If the user chose not to use initial guess or reference files,
        # set the reference spectra to an empty list.
        if not prm.reference_filenames:
            self.ref = []
            print("No reference used.")

        else:
            # Create an array whose rows hold reference spectra and whose
            # columns are different wavelengths.

            self.ref = np.zeros((len(prm.reference_filenames),
                                 len(self.ss_region))
                                )

            for i in range(len(prm.reference_filenames)):

                guess_spectrum = np.loadtxt(prm.reference_filenames[i],
                                            delimiter="\t"
                                            ).T

                if np.ma.size(guess_spectrum, 0) != 2:
                    sys.exit("""Initiation file not valid. Data should be an n
                             by 2 matrix. Program will stop."""
                             )

                # Interpolate the initialization file at each value of x.
                self.ss_spline(self.ref,
                               np.ma.size(self.ref, axis=0),
                               guess_spectrum[0, :],
                               guess_spectrum[1, :],
                               self.ss_region
                               )

                # Normalize the reference spectra.
                self.ref[i, :] /= np.amax(np.absolute(self.ref[i, :]))

        # Plot the initial guess and reference spectra.
        plt.figure()
        plt.title("Initial Guess/Reference Spectra")

        # Choose to label units by wavelength or energy.
        if prm.wl_or_E == "wl":
            plt.xlabel("Wavelength (nm)")

        if prm.wl_or_E == "eng":
            plt.xlabel("Energy (eV)")

        plt.ylabel("Light Intensity (Normalized)")

        plt.xlim(prm.x_min, prm.x_max)

        for i in range(len(self.ref)):
            plt.plot(self.ss_region, self.ref[i, :])

        for i in range(len(self.initial_guess_spectra)):
            plt.plot(self.ss_region, self.initial_guess_spectra[i, :])

        print("Initial guess and/or reference spectra:")

        plt.show()

    def uncertainty(self):
        """Uses the standard deviation around each point to estimate the

        uncertainty of that point locally.
        """

        # The default uncertainty case is where every TA data point has the
        # same weight.
        self.unc = np.ones((len(self.wl_values), len(self.t_values)))

        # Estimate the error of each point using the standard deviation of the
        # points contained in a 2*n+1 square around the point.
        n = prm.n

        for i in range(len(self.wl_values)):

            for j in range(len(self.t_values)):

                # Define local bound for uncertainty.
                bnd = [max(0, i + 1 - n), min(len(self.wl_values), i + 1 + n),
                       max(0, j + 1 - n), min(len(self.t_values), j + 1 + n)
                       ]

                # Calculate standard deviation in the region of interest.
                roi = self.TA_values[bnd[0]:bnd[1] + 1, bnd[2]:bnd[3] + 1]

                self.unc[i, j] = np.std(roi)

        # Normalize the uncertainty by the maximum value.
        self.unc = self.unc / np.amax(self.unc)

        # Prevent dividing by zero during a weighted residual calculation.
        # Setting it to one means those residual values will not receive a
        # penalty. High-uncertainty points are weighted less.

        self.unc[self.unc == 0] = 1

    def smooth(self):
        """Reduces noise in the TA data before running the genetic algorithm.

        Smoothing is currently done in another program, but this could be

        expanded to integrate with other spectral analysis modules.
        """

        sm_zone = (np.ones(self.TA_values.shape)
                   / (np.ma.size(self.TA_values, 0)
                      * np.ma.size(self.TA_values, 1)
                      )
                   )
        print(self.TA_values.shape)
        print(sm_zone.shape)

        # Python equivalent to MATLAB's filter2 function.
        # Equivalent to TA = filter2(sm_zone, TA) in MATLAB.

        self.TA_values = signal.convolve2d(self.TA_values,
                                           sm_zone,
                                           mode='valid'
                                           )

    def cut_off(self):
        """Sets the region of interest of the TA data for running the genetic

        algorithm.
        """

        # Check that the region of interest is entirely within the TA data.
        # If not, the boundry of the TA data becomes the boundry of the region
        # of interest.
        if prm.x_min < self.wl_values[0]:
            x_min = self.wl_values[0]
            print("x_min parameter is out of bounds")

        else:
            x_min = prm.x_min

        if prm.t_min < self.t_values[0]:
            t_min = self.t_values[0]
            print("t_min parameter is out of bounds")

        else:
            t_min = prm.t_min

        if prm.x_max > self.wl_values[-1]:
            x_max = self.wl_values[-1]
            print("x_max parameter is out of bounds")

        else:
            x_max = prm.x_max

        if prm.t_max > self.t_values[-1]:
            t_max = self.t_values[-1]
            print("t_max parameter is out of bounds")

        else:
            t_max = prm.t_max

        # Get the cutoff positions in the vectors.
        x_co_interp_function = interpolate.interp1d(self.x,
                                                    list(range(len(self.x)))
                                                    )

        x_co = x_co_interp_function([x_min, x_max]).astype(int)

        t_co_interp_function \
            = interpolate.interp1d(self.t_values,
                                   list(range(len(self.t_values)))
                                   )

        t_co = t_co_interp_function([t_min, t_max]).astype(int)

        # Cut the at selected positions.
        self.x = self.x[x_co[0]:x_co[1]]
        self.t_values = self.t_values[t_co[0]:t_co[1]]
        self.TA_values = self.TA_values[x_co[0]:x_co[1], t_co[0]:t_co[1]]

        # Update the reference spectra.
        if prm.reference_filenames:
            self.ref = self.ref[:, x_co[0]:x_co[1]]

        # Interpolate the reference spectra over the entire cutoff range so
        # that their sizes match the sizes of the guess spectra.

        self.ref_full = np.zeros((len(prm.reference_filenames),
                                 len(self.x))
                                 )

        self.ss_spline(self.ref_full,
                       np.ma.size(self.ref, axis=0),
                       self.ss_region,
                       self.ref[0, :],
                       self.x
                       )
        # Normalize the reference spectra to one.
        for i in range(np.ma.size(self.ref_full, 0)):
            self.ref_full[i, :] /= np.amax(np.absolute(self.ref_full[i, :]))

        # Update the cutoff of the uncertainty array.
        self.unc = self.unc[x_co[0]:x_co[1], t_co[0]:t_co[1]]

        # Update the gaussian parameters used to generate the initial spectra.
        self.gauss_prm[1] = self.gauss_prm[1] * len(self.ss_region)
        self.gauss_prm[2] = self.gauss_prm[2] * len(self.ss_region)

    def run_GA(self):
        """Constructs and runs the genetic algorithm on the TA data."""

        # Monitor the convergence.
        best_residual_log = np.zeros((1, prm.num_runs * prm.num_iterations))

        # Start tracking the runtime of genetic algorithm.
        start_time = time.time()

        # Run the GA for the number of iterations given in the parameter file.
        for i in range(prm.num_iterations):

            GAopt1 = self.GA_optimization()

            # The best_spectra matrix contains the best spectra by wavelength
            # (or energy). The brs is the best residual trace to show
            # convergence at each rr time step.

            self.best_spectra, best_residual = GAopt1[0], GAopt1[1]
            print("best residual: " + str(best_residual))

            # Log the residual. The best_residual rows have the residual values
            # while the 0th column is the list of residuals. The number of rows
            # will be prm.num_runs which will equal the length of the
            # best_residual_log slice.

            best_residual_log[0, (i) * prm.num_runs: (i + 1) * prm.num_runs] \
                = best_residual[:, 0]

            # Update the Gaussian parameters
            self.gauss_prm[0] /= prm.red_noise
            self.init_spe_noise_amplitutde /= prm.red_noise

            # Make the optimized spectra the new initial guesses.
            self.initial_guess_spectra = self.best_spectra

            # Print the best residual, elapsed runtime, and remaining runtime.
            elapsed = time.time() - start_time

            if i == prm.num_iterations - 1:
                print("\nTotal progress: "
                      + str(round((100 * (i + 1) / prm.num_iterations)))
                      + "%. Total runtime: "
                      + str(datetime.timedelta(seconds=int(elapsed)))
                      + " (Hours:Mins:Secs).\n"
                      )
            else:
                print("\nTotal progress: "
                      + str(round((100 * (i + 1) / prm.num_iterations)))
                      + "%. Current runtime: "
                      + str(datetime.timedelta(seconds=int(elapsed)))
                      + ". Remaining runtime: "
                      + str(datetime
                            .timedelta(seconds=round(elapsed *
                                                     (prm.num_iterations
                                                      - i + 1
                                                      ) / (i + 1))
                                       )
                            )
                      + ". (Hours:Mins:Secs)."
                      )

            # Save the results.
            self.save_results()

            print("Best residual log: ", best_residual_log)
            print("Best residual: " + str(np.amax(best_residual_log)))

        print("""Genetic Algorithm has completed. The results have been saved
              to separate files in the directory."""
              )

    def GA_optimization(self):
        """Generates initial populations, noise, Gaussian functions, and new

        populations over many generations.
        """

        # Generate an initial random population of spectra in the subsampled
        # region.
        self.random_population = np.zeros((prm.population_size,
                                           np.ma.size(self
                                                      .initial_guess_spectra,
                                                      axis=0
                                                      ),
                                           len(self.ss_region)
                                           )
                                          )

        # A Gaussian function where p is a list of parameters (amplitude, mean
        # and width).
        gaussian = lambda p, x: p[0] * np.exp(-(x - p[1]) ** 2 / p[2] ** 2)

        # Add a dimension to the array for handling matrices.
        if self.initial_guess_spectra.ndim == 1:
            self.initial_guess_spectra = self.initial_guess_spectra[np.newaxis,
                                                                    :
                                                                    ]

        for j in range(np.ma.size(self.initial_guess_spectra, axis=0)):

            # Preserve one untouched set of initialization spectrum.
            self.random_population[0, j, :] = self.initial_guess_spectra[j, :]

            # Modify the remaining spectra in the generation population.
            for i in range(1, prm.population_size):

                # Add the noise pointwise in a random selection along the
                # subsampled x axis.
                nbr = np.ceil(np.random.rand()
                              * len(self.ss_region)
                              * 0.1
                              ).astype(int)

                # Assign <10% of entries a random value [-0.15, 2.01].
                self.random_population[i,
                                       j,
                                       ((np.ceil(len(self.ss_region)
                                        * np.random.rand(nbr, 1))).astype(int)
                                        - 1
                                        )
                                       ] = (self.init_spe_noise_amplitutde
                                            * (np.log(np.random.rand(nbr, 1))
                                               / -5 - prm.init_spe_noise_offset
                                               )
                                            )

                # Create an array that stores sets of random gaussian
                # parameters (Amplitude, Mean, and Width) in the rows. Each row
                # is like a gauss_prm array.

                # Make an array of initial random fluctuations.
                rand_gauss_prm_sets = np.random.rand(prm.num_gauss, 3)

                # Change the amplitude of the noise accross the generation.
                rand_gauss_prm_sets[:, 0] = (self.gauss_prm[0]
                                             * (rand_gauss_prm_sets[:, 0]
                                                - 0.5
                                                )
                                             * (prm.population_size - i - 1)
                                             / prm.population_size
                                             )

                # Slightly randomize the mean.
                rand_gauss_prm_sets[:, 1] = (self.gauss_prm[1]
                                             * rand_gauss_prm_sets[:, 1]
                                             )

                # Slightly randomize the width.
                rand_gauss_prm_sets[:, 2] = (self.gauss_prm[2]
                                             * rand_gauss_prm_sets[:, 2]
                                             )

                # Construct the Gaussian noise landscape by adding the
                # Gaussians that have sightly randomized parameters and by
                # assigning them to the random population.

                for k in range(prm.num_gauss):

                    self.random_population[i, j, :] = \
                        (np.squeeze(self.random_population[i, j, :])

                         + gaussian(rand_gauss_prm_sets[k, :],

                                    np.array(list(range(len(self.ss_region))))
                                    )
                         )

                # Add the Gaussian landscape to the starting guess to finish
                # constructing the random population.

                self.random_population[i, j, :] = \
                    (np.squeeze(self.random_population[i, j, :])
                     * np.random.rand(1)
                     + self.initial_guess_spectra[j, :]
                     )

        # Store the best residual at each iteration to monitor convergence.
        # For long runs with many points, this significantly slows down
        # optimization by taking lots of memory and can lead to crashing. The
        # two columns have a list of ranked residuals and its mean.

        best_residual = np.zeros((prm.num_runs, 2))

        # Add a dimension to the array if the squeeze removed a singleton
        # dimension for when there is only one spectrum to optimize.

        if self.random_population.ndim == 1:
            self.random_population = self.random_population[:, np.newaxis, :]

        # Begin the evolution by calling the evolution method.
        print("Evolution started...")

        for i in range(prm.num_runs):

            evolution_results = self.evolution()

            # Update the random population as the new population from the
            # results of the evolution.

            self.random_population = evolution_results[0]
            br = evolution_results[1]
            best_residual[i, :] = br
            print("Current iteration progress: "
                  + str(100 * (i + 1) / prm.num_runs)
                  + "%\n Runs completed: " + str(i + 1)
                  + "\n Runs remaining: "
                  + str(prm.num_runs - i - 1))

        best_spectra = np.squeeze(self.random_population[0, :, :])

        # Add a dimension to the array if the squeeze removed a singleton
        # dimension for when there is only one spectrum to optimize.

        if best_spectra.ndim == 1:
            best_spectra = best_spectra[np.newaxis, :]

        print("Evolution completed.")
        return best_spectra, best_residual

    def evolution(self):
        """Holds a tournament for the best gene selections."""

        # Create and evaluate a list of fitness values with the fitness method.
        fitness_list = np.zeros(prm.population_size)

        for k in range(len(fitness_list)):

            # Use vectorised interpolation. Interpolate the guesses for use in
            # fitness calculations from the subsampled region to the entire
            # window.

            d = self.random_population.shape

            # Stack all of the guess spectra from the various individuals of
            # the population on top of each other. This drops the distinction
            # of the first dimension that contains individuals of the
            # population, but the order is still maintained due to the
            # predictable repeating structure of the guess spectra.

            reshaped_rp = np.transpose(np.reshape(self.random_population,
                                                  (d[0] * d[1], d[2])
                                                  )
                                       )

            # Interpolate the x values for each guess spectra (previously the
            # third dimension but now the second dimension).

            spline_reshaped = np.zeros(reshaped_rp.shape)

            reshaped_interp_func = \
                interpolate.interp1d(self.ss_region,
                                     reshaped_rp,
                                     axis=0,
                                     fill_value="extrapolate"
                                     )

            spline_reshaped = reshaped_interp_func(self.x)

            # Return the array to its original shape.
            interpolated_random_population = np.reshape(spline_reshaped,
                                                        (d[0], d[1], -1)
                                                        )

            # Calculate the fitness on the interpolated array.
            for i in range(prm.population_size):
                fitness_list[i] = \
                    self.fitness(interpolated_random_population[i, :, :])

        # Generate a sorted population.
        # Reverse numpy's sort for descending values.

        sorted_fitness_values = np.sort(fitness_list)[::-1]

        sorted_fitness_indeces = np.argsort(fitness_list)[::-1]

        sorted_population = \
            self.random_population[sorted_fitness_indeces, :, :]

        # Return the best residual.
        br = [sorted_fitness_values[0], np.mean(sorted_fitness_values)]

        # Initialize the new population matrix.
        new_population = np.zeros(sorted_population.shape)

        # Automatically keep the best two out of the population.
        new_population[:prm.elitism, :, :] = \
            sorted_population[:prm.elitism, :, :]

        # Make a tournament for the remaining new population and randomly
        # select the pairs that will couple. Note: the fitness here is actually
        # the remainder of the minimization problem, so it must be minimized.
        # Generate a random selection of pairs and some additional random
        # numbers that will be used later.

        # Generate all the random numbers used in the evolution algorithm.
        evol_rand_num_sets = np.random.rand(int(prm.population_size / 2),
                                            np.ma.size(sorted_population,
                                                       axis=1
                                                       ),
                                            8
                                            )
        evol_rand_num_sets[:, :, :4] = np.floor(evol_rand_num_sets[:, :, :4]
                                                * prm.population_size
                                                )

        # Cycle through population
        for i in range(int(prm.elitism / 2), int(prm.population_size / 2)):
            # Cycle through spectra to optimize
            for j in range(np.ma.size(sorted_population, 1)):

                # Hold a scaled fitness values tournament
                # Choose random parents from the random indices in
                # evol_rand_num_sets[:,:,0:4].

                p1 = sorted_population[
                                int(np.minimum(evol_rand_num_sets[i, j, 0],
                                               evol_rand_num_sets[i, j, 1]
                                               )
                                    ), j, :]

                p2 = sorted_population[
                                int(np.minimum(evol_rand_num_sets[i, j, 2],
                                               evol_rand_num_sets[i, j, 3]
                                               )
                                    ), j, :]

                # Define the adaptative crossover probability (cp).
                # br[0] is the best residual.
                # br[1] is the average residual.

                f1 = sorted_fitness_values[
                                int(np.amin(evol_rand_num_sets[i, j, :4]))]

                # If the selected fitness value is greater than the mean...
                if f1 >= br[1]:
                    cp = (br[0] - f1) / (br[0] - br[1])
                else:
                    cp = 1

                # Check if the parents have children. If yes, mix genes (swap
                # part of the children's genes) with a Gaussian process.

                if evol_rand_num_sets[i, j, 4] < cp:
                    # Create a randomised 'gene' selection window.
                    a = (np.exp(-(np.square(np.subtract(

                      list(range(len(self.ss_region))),
                      (len(self.ss_region) * evol_rand_num_sets[i, j, 5])

                                                        )  # np.subtract
                                            )  # np.square
                      / (len(self.ss_region) * evol_rand_num_sets[i, j, 6] / 2)
                      ** 2
                                  )  # "-"
                                )  # np.exp
                         )

                    new_population[2 * (i) - 2, j, :] = (a
                                                         * np.squeeze(p2)
                                                         + (1 - a)
                                                         * np.squeeze(p1)
                                                         )

                    new_population[2 * (i) - 1, j, :] = (a
                                                         * np.squeeze(p1)
                                                         + (1 - a)
                                                         * np.squeeze(p2)
                                                         )
                else:
                    # Copy the parents' genes.
                    new_population[2 * i - 2, j, :] = p1
                    new_population[2 * i - 1, j, :] = p2

                # Mutate the population.
                if evol_rand_num_sets[i, j, 7] < 0.05:
                    # Number of points to change.
                    nb = int(np.ceil((i + 1)
                                     / prm.population_size
                                     * 2 * len(self.ss_region) * 0.1
                                     )
                             )

                    # Randomly select which points along the x axis (the 3rd
                    # dimension) to change (0, :) and by how much to change
                    # them (1, :).

                    nbr = np.random.rand(2, nb)

                    new_population[2 * i - 2, j, ((np.ceil(nbr[0, :]
                                                   * len(self.ss_region))
                                                   ).astype(int) - 1
                                                  )  # 3rd dimension
                                   ] = \
                    np.squeeze(new_population[2 * i - 2, j,
                                              (np.ceil(nbr[0, :]
                                                       * len(self.ss_region)
                                                       )  # np.ceil
                                               ).astype(int) - 1
                                              ]  # new_population
                               ) + prm.sms * (nbr[1, :] - 0.5)  # squeeze

        # Check if there is a sign constrain on every optimizable spectrum.
        # Set all points of the incorrect sign to zero.
        for j in range(np.ma.size(sorted_population, 1)):

            # There is no sign constraint on the guess spectrum.
            if prm.species_constraint[j] == 0:
                pass

            # The spectrum must be positive.
            elif prm.species_constraint[j] == 1:
                new_population[:, j, :] = \
                    np.multiply(new_population[:, j, :],
                                new_population[:, j, :] > 0
                                )

            # The spectrum must be negative.
            elif prm.species_constraint[j] == -1:
                new_population[:, j, :] = \
                    np.multiply(new_population[:, j, :],
                                new_population[:, j, :] < 0
                                )

            else:
                # This is designed to make the user be explicit in their choice
                # of constraints and to avoid silent errors in optimization.
                sys.exit("""Please use either "0", "-1", or "1" for species
                         constraints."""
                         )

        return new_population, br

    def fitness(self, spectrum):
        """Calculates how the reconstructed spectra compare to the observed

        spectra.
        """

        # Optimize the time traces (tt) associated to the spectra calculates
        # the weighted squared residuals. Calculate the time traces from the
        # spectra available.
        if len(prm.reference_filenames) > 0:
            refSP = np.concatenate((spectrum, self.ref_full))
        else:
            refSP = spectrum

        # Perform MATLAB's right matrix division as TA / refSP. There is no
        # equivalent in numpy, so solve for tt in TA' * tt = refSP' instead of
        # tt * TA = refSP.

        # Instead of solving self.TA_values / refSP,
        # solve (refSP.T \ self.TA_values.T).T using QR.
        # Transpose self.TA_values to accommodate data collection.
        self.TA_values.T
        Q, R = np.linalg.qr(np.transpose(refSP))

        QrefSP = np.matmul(Q.T, self.TA_values)
        self.tt = np.linalg.lstsq(R, QrefSP, rcond=None)[0]

        # Generate the 2D matrix from the data.
        model = np.matmul(np.transpose(self.tt), refSP)

        # Calculate the residual. Transpose self.unc to match self.TA_values.
        res = np.mean(np.square((self.TA_values - np.transpose(model)))
                      / self.unc
                      )

        # Implement a penalty for switching from positive to negative. This way
        # ground-state bleach (GSB) features stay positive and photo-induced
        # absorbtion (PIA) features stay negative. The spectra are kept all
        # positive/negative at the end of the "evolution" method,
        # right after the guess spectra are mutated.

        # The fraction of tt points below 0 is calculated and the residual is
        # increased accordingly.
        res = 1 / (res
                   * (1
                      + prm.negative_kinetic_penalty
                      * np.sum(self.tt[self.tt < 0])
                      / self.tt.size
                      )
                   )

        return res

    def ss_spline(self, spectra_set, loop_range, x_old, y_old, x_new):
        """This creates an interpolation fucntion that is used when switching

        between the entire axis and the subsampled axis.

        """

        for i in range(loop_range):
            interp_spe_func = interpolate.interp1d(x_old,
                                                   y_old,
                                                   fill_value="extrapolate"
                                                   )
            spectra_set[i, :] = interp_spe_func(x_new)

    def QR_solve(self, A, B):
        """A Python implementation of MATLAB's right matrix division ("/").

        QR_solve(A, B) performs the equivalent of MATLAB's A / B that solves

        for x in the equation Ax = B. In Python, instead of performing A / B,

        it computes (B' \ A')' using QR factorization.
        """

        Q2, R2 = np.linalg.qr(np.transpose(B))

        Q2B = np.matmul(Q2.T, np.transpose(A))

        X = np.linalg.lstsq(R2, Q2B, rcond=None)[0]

        return X

    def save_results(self):
        """Saves the results of the genetic algorithm in text files."""

        # Save x in the correct type of units (wavelength or energy).
        if prm.wl_or_E == "wl":
            x_output_name = "wl"

        if prm.wl_or_E == "eng":
            x_output_name = "E"

        # Generate the spectra matrix and concatenate the reference
        # (unmodified) spectra to self.spe.

        self.spe = np.zeros((np.ma.size(self.best_spectra, 0), len(self.x)))

        if self.best_spectra.ndim > 1:
            for i in range(np.ma.size(self.best_spectra, 0)):
                interp_spe_func = interpolate.interp1d(self.ss_region,
                                                       self.best_spectra[i, :],
                                                       fill_value="extrapolate"
                                                       )
                self.spe[i, :] = interp_spe_func(self.x)

            # Normalize the spectra to one.
            for i in range(np.ma.size(self.spe, 0)):
                self.spe[i, :] /= np.amax(np.absolute(self.spe[i, :]))

            if prm.reference_filenames:
                self.spe = np.concatenate((self.spe, self.ref_full))

        else:
            interp_spe_func = interpolate.interp1d(self.ss_region,
                                                   self.best_spectra,
                                                   fill_value="extrapolate"
                                                   )
            self.spe = interp_spe_func(self.x)

            # Normalize the spectra to one.
            self.spe /= np.amax(np.absolute(self.spe))

            if prm.reference_filenames:
                self.spe = np.concatenate((self.spe[None, :], self.ref))

        # Calculate self.TA_values / self.spe using QR factorization.
        print("num NaN in spe: "
              + str(self.spe.size - np.sum(np.isfinite(self.spe)))
              )
        print("num NaN in TA_values: "
              + str(self.TA_values.size - np.sum(np.isfinite(self.TA_values)))
              )
        self.tt_output = \
            np.transpose(self.QR_solve(np.transpose(self.TA_values), self.spe))

        print("tt_output shape:", self.tt_output.shape)

        # Normalize the kinetics to one.
        for i in range(np.ma.size(self.tt_output, 1)):
            self.tt_output[:, i] /= np.amax(np.absolute(self.tt_output[:, i]))

        # "spe.txt" contains the normalized spectral components over wavelength
        # (or energy) scale x. tt contains the time traces of the components
        # over timescale t.
        np.savetxt(os.path.join(self.directory, "{}.txt")
                   .format(x_output_name),
                   self.x,
                   delimiter="\t"
                   )
        np.savetxt(os.path.join(self.directory, "spe.txt"),
                   np.transpose(self.spe),
                   delimiter="\t"
                   )
        np.savetxt(os.path.join(self.directory, "t.txt"),
                   self.t_values,
                   delimiter="\t"
                   )
        np.savetxt(os.path.join(self.directory, "tt.txt"),
                   self.tt_output,
                   delimiter="\t"
                   )

        # Plot the results

        # Plot the normalized component spectra.
        print("Best spectral features:")

        plt.figure()
        plt.title("Normalized Component Spectra")
        for i in range(np.ma.size(self.spe, 0)):
            plt.plot(self.x, np.transpose(self.spe[i, :]),
                     label="Feature {}".format(i + 1)
                     )

        plt.legend()
        # Choose to label units by wavelength or energy.
        if prm.wl_or_E == "wl":
            plt.xlabel("Wavelength (nm)")

        if prm.wl_or_E == "eng":
            plt.xlabel("Energy (eV)")

        plt.ylabel("Light Intensity (arb. unit)")

        plt.xlim(prm.x_min, prm.x_max)

        plt.savefig(os.path.join(self.directory, "spectra.png"),
                    bbox_inches='tight', dpi=1200
                    )
        plt.savefig(os.path.join(self.directory, "spectra.pdf"),
                    bbox_inches='tight'
                    )

        # Plot the intensity of each component spectrum over time.
        plt.figure()
        plt.title("Intensities of Component Spectra over Time")
        for i in range(np.ma.size(self.tt_output, 1)):
            plt.plot(self.t_values,
                     self.tt_output[:, i],
                     label="Feature {}".format(i + 1)
                     )
        plt.legend()
        plt.xlim(0.05, prm.t_max)
        plt.ylim(0.1, 1.05)
        plt.xscale("log")
        plt.xlabel("Time (ps)")
        plt.ylabel("Light Intensity (arb. unit)")
        plt.savefig(os.path.join(self.directory, "kinetics.png"),
                    bbox_inches="tight", dpi=1200
                    )
        plt.savefig(os.path.join(self.directory, "kinetics.pdf"),
                    bbox_inches="tight"
                    )
        plt.show()

    def save_metadata(self):
        """Saves parameters for an execution of the genetic algorithm."""

        metadata = \
            [["TA_data_filename:", prm.TA_data_filename],
             ["wl_or_E:", prm.wl_or_E],
             ["t_min:", prm.t_min],
             ["t_max:", prm.t_max],
             ["x_min:", prm.x_min],
             ["x_max:", prm.x_max],
             ["ss_factor:", prm.ss_factor],
             ["use_initial_guess:", prm.use_initial_guess],
             ["num_spectra_to_optimize:", prm.num_spectra_to_optimize],
             ["initial_guess_filenames:", prm.initial_guess_filenames],
             ["reference_filenames:", prm.reference_filenames],
             ["n:", prm.n],
             ["population_size:", prm.population_size],
             ["num_runs:", prm.num_runs],
             ["init_spe_noise_amplitutde:", prm.init_spe_noise_amplitutde],
             ["init_spe_noise_offset:", prm.init_spe_noise_offset],
             ["gauss_prm:", prm.gauss_prm],
             ["num_gauss:", prm.num_gauss],
             ["sms:", prm.sms],
             ["elitism:", prm.elitism],
             ["negative_kinetic_penalty:", prm.negative_kinetic_penalty],
             ["species_constraint:", prm.species_constraint],
             ["num_iterations:", prm.num_iterations],
             ["red_noise:", prm.red_noise]
             ]

        # Create the metadata file.
        with open(os.path.join(self.directory, 'metadata.txt'), 'w') as f:
            for parameter in metadata:
                for item in parameter:
                    row = parameter[0] + " " + str(parameter[1])
                f.write("{}\n".format(row))

        return metadata

    def timeslices(self):
        """Plots time slices of the complete observed spectrum."""

        plt.figure()
        plt.title("Total Spectrum at Various Times")

        # Choose times to view the total spectrum.
        timeslice1 = 1
        timeslice2 = 50
        timeslice3 = 180

        plt.plot(GA1.x,
                 GA1.TA_values[:, 1] * 10 ** 4,
                 label="{:.1f} ps".format(self.t_values[timeslice1])
                 )
        plt.plot(self.x,
                 self.TA_values[:, 50] * 10 ** 4,
                 label="{:.1f} ps".format(self.t_values[timeslice2])
                 )
        plt.plot(self.x,
                 self.TA_values[:, 180] * 10 ** 4,
                 label="{:.0f} ps".format(self.t_values[timeslice3])
                 )

        # Place the legend outside the graph.
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Label units in either wavelength or energy.
        if prm.wl_or_E == "wl":
            plt.xlabel("Wavelength (nm)")

        if prm.wl_or_E == "eng":
            plt.xlabel("Energy (eV)")

        plt.xlim(prm.x_min, prm.x_max)

        # Display units in scientific notation.
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ylabel("Light Intensity (arb. unit)")

        # Save the data as raster (bitmap) and vector image files.
        plt.savefig(os.path.join(self.directory, 'timeslices.png'),
                    bbox_inches='tight', dpi=1200
                    )
        plt.savefig(os.path.join(self.directory, 'timeslices.pdf'),
                    bbox_inches='tight'
                    )

        print("Time slices of the total spectrum: ")

        plt.show()

        return

    def surface_plots(self):
        """Plots surface maps of the total and compenent spectra."""

        TA_surface_plot = plt.figure()
        ha = TA_surface_plot.add_subplot(111, projection='3d')
        # "plot_surface" expects `x` and `y` data to be 2D
        X, T = np.meshgrid(self.x, self.t_values)

        ha.plot_surface(X, T, np.transpose(self.TA_values * 10 ** 4),
                        cmap=plt.cm.viridis,
                        linewidth=0,
                        antialiased=False
                        )

        # Choose the orientation of the surface plot.
        ha.view_init(30, 135)
        #  Label the graph.
        plt.title("Complete Spectrum\n")
        plt.xticks(rotation=20,
                   rotation_mode="anchor",
                   ha="right",
                   va="bottom"
                   )

        ax = TA_surface_plot.gca(projection='3d')
        # Create x labels in the correct type of units (wavelength or energy).
        if prm.wl_or_E == "wl":
            ax.set_xlabel("\n\nWavelength (nm)")
        if prm.wl_or_E == "eng":
            ax.set_xlabel("\n\nEnergy (eV)")

        ax.set_ylabel("\n\nTime (picoseconds)")
        ax.set_zlabel("Intensity (arb. unit)")

        plt.gcf().subplots_adjust(right=1.5, bottom=-0.5)

        plt.savefig(os.path.join(self.directory, 'surfaceplot.png'),
                    bbox_inches='tight', dpi=1200
                    )
        plt.savefig(os.path.join(self.directory, 'surfaceplot.pdf'),
                    bbox_inches='tight'
                    )
        plt.show()

        return


# Set up figure styles.
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_palette(sns.color_palette("colorblind"), 10)
sns.set_style('whitegrid')
plt.rcParams['lines.linewidth'] = 5

# =============================================================================
# Optional contour plots of the observed spectra data.
# TAdata1 = TAData()
# plt.imshow(TAdata1.TA_values, cmap="plasma")
# plt.show()
#
# # Plot a contour map TA Data
# TAdata1 = TAData()
# plt.contourf(TAdata1.t_values, TAdata1.wl_values, TAdata1.TA_values)
# plt.show
#
# =============================================================================

# Create an instance of the model and prepare the data.
GA1 = GeneticAlgorithm()
GA1.load_initial_guesses_and_reference_spectra()
print("spectra loaded")
GA1.uncertainty()
print("uncertainty calculated")
GA1.cut_off()
print("cutoff established")

GA1.surface_plots()
GA1.timeslices()
GA1.run_GA()
