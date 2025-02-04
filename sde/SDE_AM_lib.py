# could just sqrt(abs()) be optimal for "diagonal"

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt
import time

from numpy import linalg as LA
from matplotlib.colors import Normalize
from multiprocessing import Pool
from scipy.linalg import sqrtm
from scipy.integrate import nquad
from numpy.linalg import cholesky

tfd = tfp.distributions


class NNHyperparameters:
    def __init__(self, K=2**6, M_min=0, M_max=100, lambda_reg=2e-3, gamma=1, delta=0.1, name=None):
        self.K = K
        self.M_min = M_min
        self.M_max = M_max
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.delta = delta
        self.name = name


class SDEAMTrain:
    def __init__(self, n_dimensions=1, x_min=None, x_max=None, omega_drift=None, amp_drift=None, z_mean=0, z_std=1,
                 omega_diff=None, amp_diff=None, diff_std=1, diff_type="diagonal", constant_diff=False, resampling=True):
        self.d = n_dimensions
        self.tri = n_dimensions * (n_dimensions + 1) // 2
        self.x_min = x_min
        self.x_max = x_max
        self.omega_drift = omega_drift
        self.amp_drift = amp_drift
        self.z_mean = z_mean
        self.z_std = z_std
        self.omega_diff = omega_diff
        self.amp_diff = amp_diff
        self.diff_std = diff_std
        self.diff_type = diff_type
        self.constant_diff = constant_diff
        self.resampling = resampling
        self.history = {'loss': None, 'val_loss': None, 'training_time': None}

    @staticmethod
    def normalise_z(z):
        z_mean = np.mean(z, axis=0),
        z_std = np.std(z, axis=0)
        z_norm = (z - z_mean) / z_std
        return z_norm, z_mean, z_std

    @staticmethod
    def normalise_diff_vectors(diff_vectors):
        diff_std = np.mean(diff_vectors, axis=0)
        diff_vectors_norm = diff_vectors / diff_std
        return diff_vectors_norm, diff_std

    @staticmethod
    def split_data(validation_split, *inputs):
        num_samples = inputs[0].shape[0]
        valid_sample_size = int(num_samples * validation_split)

        # Generate random indices for the validation set
        valid_indices = np.random.choice(num_samples, size=valid_sample_size, replace=False)

        # Generate masks to separate training and validation data
        mask = np.ones(num_samples, dtype=bool)
        mask[valid_indices] = False

        # Apply the mask to all inputs at once
        inputs_train = tuple(data[mask] for data in inputs)
        inputs_valid = tuple(data[~mask] for data in inputs)

        return inputs_train, inputs_valid

    @staticmethod
    def S(x, omega):
        x_omega = np.matmul(x, omega)
        S_ = np.exp(1j * x_omega)
        return S_

    @staticmethod
    def beta(x, omega, amp):
        beta_ = np.real(np.matmul(SDEAMTrain.S(x, omega), amp))
        return beta_

    @staticmethod
    def matrix_sqrtm(matrix):
        return sqrtm(matrix)

    @staticmethod
    def matrix_cholesky(matrix):
        return cholesky(matrix)

    def diff(self, x):
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        diff_vectors = SDEAMTrain.beta(x_norm, self.omega_diff, self.amp_diff) * self.diff_std
        diff_matrix = np.zeros((x.shape[0], self.d, self.d))
        
        if self.diff_type == "diagonal":
            idx = np.arange(self.d)
            diff_matrix[:, idx, idx] = np.sqrt(np.abs(diff_vectors))
        else:
            lower_triangle_indices = np.tril_indices(self.d)
            diff_matrix[:, lower_triangle_indices[0], lower_triangle_indices[1]] = diff_vectors[:, :self.tri]
            diff_matrix[:, lower_triangle_indices[1], lower_triangle_indices[0]] = diff_vectors[:, :self.tri]
            if self.diff_type == "triangular":
                with Pool() as pool:
                    diff_matrix = np.array(pool.map(SDEAMTrain.matrix_cholesky, diff_matrix))
            else:
                with Pool() as pool:
                    diff_matrix = np.array(pool.map(SDEAMTrain.matrix_sqrtm, diff_matrix))
                
        return diff_matrix

    def drift(self, x):
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        drift_ = (SDEAMTrain.beta(x_norm, self.omega_drift, self.amp_drift) * self.z_std + self.z_mean)
        return drift_

    def drift_diffusivity(self, x, _):
        return SDEAMTrain.drift(self, x), SDEAMTrain.diff(self, x)

    def get_diff_vectors(self, y_n, y_np1, x, step_sizes):
        f = y_np1 - (y_n + step_sizes * SDEAMTrain.drift(self, x))
        if self.diff_type == "diagonal":
            diff_vectors = f ** 2 / step_sizes
        else:
            f_reshape = f[:, :, np.newaxis]
            f_square = f_reshape @ f_reshape.transpose(0, 2, 1)
            f_square_h = f_square / step_sizes.reshape(-1, 1, 1)

            lower_triangle_indices = np.tril_indices(self.d)
            diff_vectors = f_square_h[:, lower_triangle_indices[0], lower_triangle_indices[1]]

        return diff_vectors

    @staticmethod
    def get_amp(x, y_np1, lambda_reg, omega, K):
        St = SDEAMTrain.S(x, omega)
        cm = np.matmul(np.transpose(np.conj(St)), St) + x.shape[0] * lambda_reg * np.identity(K)
        amp = LA.solve(cm, np.matmul(np.transpose(np.conj(St)), y_np1))
        return amp

    def get_loss(self, true_drift, true_diffusivity, y_n, y_np1, x, step_sizes):
        drift_ = SDEAMTrain.drift(self, x)
        diffusivity_ = SDEAMTrain.diff(self, x)
        #drift_ = true_drift(x)
        #diffusivity_ = true_diffusivity(x)

        loc = y_n + step_sizes * drift_
        scale = np.sqrt(step_sizes).reshape(-1, 1, 1) * diffusivity_
        if self.diff_type == "spd":
            scale = tf.linalg.matmul(scale, tf.linalg.matrix_transpose(scale))
            scale = tf.linalg.cholesky(scale)
        
        loc = tf.convert_to_tensor(loc, dtype=tf.float64)
        scale = tf.convert_to_tensor(scale, dtype=tf.float64)
        
        if self.diff_type == "diagonal":
            approx_normal = tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=np.diagonal(scale, axis1=1, axis2=2),
                name="approx_normal"
            )
        else:
            approx_normal = tfd.MultivariateNormalTriL(
                loc=loc,
                scale_tril=scale,
                name="approx_normal"
            )

        log_prob = approx_normal.log_prob(y_np1)
        sample_distortion = -tf.reduce_mean(log_prob, axis=-1)
        distortion = tf.reduce_mean(sample_distortion)

        additive_constant = 2 * self.d * np.log(2 * np.pi) / np.log(10)
        loss = distortion + additive_constant
        return loss

    def am_train(self, param, x, y_norm, validation_split):
        start_time = time.time()
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        (x_norm, y_norm), (x_norm_valid, y_norm_valid) = SDEAMTrain.split_data(validation_split, x_norm, y_norm)

        K = param.K
        omega = np.zeros((x.shape[1], K))
        amp = SDEAMTrain.get_amp(x_norm, y_norm, param.lambda_reg, omega, K)

        ve = np.zeros(param.M_max)
        ve_min = float('inf')
        moving_avg = np.zeros(param.M_max)
        min_moving_avg = float('inf')
        moving_avg_len = 5
        min_index = 0
        break_iterations = 5

        for i in range(param.M_max):
            # train omega
            omega_prime = omega + param.delta * np.random.normal(0, 1, size=(x.shape[1], K))
            amp_prime = SDEAMTrain.get_amp(x_norm, y_norm, param.lambda_reg, omega_prime, K)
            for k in range(0, K - 1):
                D = (np.linalg.norm(amp_prime[k, :]) / np.linalg.norm(amp[k, :])) ** param.gamma
                if D >= np.random.random():
                    amp[k, :] = amp_prime[k, :]
                    omega[:, k] = omega_prime[:, k]        

            if self.resampling:
                amp_pmf = np.linalg.norm(amp, axis=1) / np.sum(np.linalg.norm(amp, axis=1))
                omega = omega[:, np.random.choice(K, K, p=amp_pmf)]

            amp = SDEAMTrain.get_amp(x_norm, y_norm, param.lambda_reg, omega, K)

            # calculate validation loss
            ve[i] = np.mean(np.abs(SDEAMTrain.beta(x_norm_valid, omega, amp) - y_norm_valid) ** 2)

            # break loop when validation loss stagnates
            if i < moving_avg_len:
                moving_avg[i] = np.mean(ve[:i+1])
            else:
                moving_avg[i] = np.mean(ve[i-moving_avg_len+1:i+1])

            if moving_avg[i] < min_moving_avg:
                min_moving_avg = moving_avg[i]
                min_index = i

            if min_index + break_iterations < i and i > param.M_min:
                break

            # storing values for lowest validation loss
            if ve[i] < ve_min:
                end_time = time.time()
                ve_min = ve[i]
                setattr(self, f'omega_{param.name}', np.array(np.copy(omega)))
                setattr(self, f'amp_{param.name}', np.array(np.copy(amp)))

            print(f"\r{param.name} epoch: {i}", end='')
        print()

        return ve[:i], moving_avg[:i], end_time-start_time

    def train_model(self, drift_param, diff_param, true_drift, true_diffusivity, y_n, y_np1, x=None, step_sizes=None, validation_split=0.1, AM_validation_split=0.1, XP=True, plot=False):
        if x is None:
            x = y_n
        elif XP:
            x = np.concatenate((y_n, x), axis=1)

        (y_n, y_np1, x, step_sizes), (y_n_valid, y_np1_valid, x_valid, step_sizes_valid) = SDEAMTrain.split_data(validation_split, y_n, y_np1, x, step_sizes.reshape(-1, 1))

        self.x_min = np.min(x, axis=0)
        self.x_max = np.max(x, axis=0)

        # calculate z with updated diffusion
        z_start = time.time()
        z = (y_np1 - y_n)/step_sizes
        z_norm, self.z_mean, self.z_std = SDEAMTrain.normalise_z(z)
        z_time = time.time() - z_start

        # train for z using AM
        ve_drift, moving_avg_drift, minima_time_drift = SDEAMTrain.am_train(self, drift_param, x, z_norm, AM_validation_split)

        if plot:
            if x.shape[1] == 1:
                SDEAMTrain.plot_1D(self, true_drift, x, z_norm, drift_param.name)
            elif x.shape[1] == 2:
                SDEAMTrain.plot_2D(self, true_drift, x, z_norm, drift_param.name)
            SDEAMTrain.plot_loss(ve_drift, moving_avg_drift)

        # calculate point-wise diffusion with updated drift
        diff_vector_start = time.time()
        diff_vectors = SDEAMTrain.get_diff_vectors(self, y_n, y_np1, x, step_sizes)
        diff_vectors_norm, self.diff_std = SDEAMTrain.normalise_diff_vectors(diff_vectors)
        diff_vector_time = time.time() - diff_vector_start

        # train for global diffusion using AM
        if not self.constant_diff:
            ve_diff, moving_avg_diff, minima_time_diff = SDEAMTrain.am_train(self, diff_param, x, diff_vectors_norm, AM_validation_split)
            if plot:
                if x.shape[1] == 1:
                    SDEAMTrain.plot_1D(self, true_diffusivity, x, diff_vectors_norm, diff_param.name)
                elif x.shape[1] == 2:
                    SDEAMTrain.plot_2D(self, true_diffusivity, x, diff_vectors_norm, diff_param.name)
                SDEAMTrain.plot_loss(ve_diff, moving_avg_diff)
        else:
            self.omega_diff = np.zeros((x.shape[1], diff_param.K))
            self.amp_diff = np.ones((diff_param.K, self.tri))/diff_param.K
            minima_time_diff = 0
            if plot:
                print('Trained Diffusion =', SDEAMTrain.diff(self, x[0, :].reshape(1, -1)))

        # calculate losses
        self.history['loss'] = SDEAMTrain.get_loss(self, true_drift, true_diffusivity, y_n, y_np1, x, step_sizes)
        self.history['val_loss'] = SDEAMTrain.get_loss(self, true_drift, true_diffusivity, y_n_valid, y_np1_valid, x_valid, step_sizes_valid)
        self.history['training_time'] = z_time + diff_vector_time + minima_time_drift + minima_time_diff

        print(f"\rObserved loss: {self.history['loss']}")
        print(f"\rObserved validation loss: {self.history['val_loss']}")
        print(f"\r Training time: {self.history['training_time']}")
        return self
        
    # plot functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_1D(self, true_func, x, y_norm, name):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'hspace': 0.12, 'wspace': 0.19})

        # grid
        x_div = 500
        x_grid = np.linspace(self.x_min, self.x_max, x_div).reshape((x_div, 1))
        x_grid_norm = np.linspace(0, 1, x_div).reshape((x_div, 1))

        # plot training data
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        ax[0].scatter(x_norm[:, 0], y_norm, alpha=0.5)

        # plot intermediate function
        omega = getattr(self, f"omega_{name}")
        amp = getattr(self, f"amp_{name}")
        ax[1].plot(x_grid_norm, SDEAMTrain.beta(x_grid_norm, omega, amp))

        # plot trained drift/diffusivity
        func = getattr(SDEAMTrain, name)
        ax[2].plot(x_grid, func(self, x_grid).reshape((x_div, 1)))

        # plot actual drift/diffusivity
        ax[2].plot(x_grid, true_func(x_grid).reshape((x_div, 1)))

        # Set labels
        ax[0].set_ylabel(r'$f(x_0)$', fontsize=12)

        ax[0].set_xlabel(r'$\bar{x_0}$', fontsize=12)
        ax[1].set_xlabel(r'$\bar{x_0}$', fontsize=12)
        ax[2].set_xlabel(r'$x_0$', fontsize=12)

        plt.show()

    def plot_2D(self, true_func, x, y_norm, name):
        output_dim = y_norm.shape[1]
        
        # grid
        x_div = 30
        y_div = 30
        x_1, x_2 = np.meshgrid(np.linspace(self.x_min[0], self.x_max[0], x_div), np.linspace(self.x_min[1], self.x_max[1], y_div))
        x_1_norm, x_2_norm = np.meshgrid(np.linspace(0, 1, x_div), np.linspace(0, 1, y_div))
        x_grid = np.column_stack((x_1.ravel(), x_2.ravel()))
        x_norm_grid = np.column_stack((x_1_norm.ravel(), x_2_norm.ravel()))
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        
        omega = getattr(self, f"omega_{name}")
        amp = getattr(self, f"amp_{name}")
        func = getattr(SDEAMTrain, name)

        intermediate = SDEAMTrain.beta(x_norm_grid, omega, amp)
        trained = func(self, x_grid)
        true_ = true_func(x_grid)

        if name == "diff" and output_dim != 1:
            if self.diff_type == "diagonal":
                trained = trained[:, [0, 1], [0, 1]]
                true_ = true_[:, [0, 1], [0, 1]]
            else:
                trained = trained[:, [0, 0, 1], [0, 1, 1]]
                true_ = true_[:, [0, 0, 1], [0, 1, 1]]
        elif output_dim == 1:
            true_ = true_.reshape(-1, 1)

        fig, ax = plt.subplots(output_dim, 4, figsize=(20, 4*output_dim), gridspec_kw={'hspace': 0.12, 'wspace': 0.19})

        if output_dim == 1:
            ax = np.expand_dims(ax, axis=0)
        
        for j in range(output_dim):
            norms_1_2 = Normalize(vmin=min(np.real(y_norm[:, j]).min(), np.real(intermediate[:, j]).min()), 
                      vmax=max(np.real(y_norm[:, j]).max(), np.real(intermediate[:, j]).max()))
            norms_3_4 = Normalize(vmin=min(np.real(trained[:, j]).min(), np.real(true_[:, j]).min()), 
                          vmax=max(np.real(trained[:, j]).max(), np.real(true_[:, j]).max()))

            # plot training data with normalization
            ax[j, 0].scatter(x_norm[:, 0], x_norm[:, 1], c=y_norm[:, j].real, cmap='viridis', s=20, norm=norms_1_2)

            # plot trained g(x) with normalization
            int = ax[j, 1].imshow(np.real(np.reshape(intermediate[:, j], (x_div, -1))), cmap='viridis', extent=[0, 1, 0, 1], origin='lower', aspect='auto', norm=norms_1_2)
        
            # plot trained drift/diffusivity with normalization
            tr = ax[j, 2].imshow(np.real(np.reshape(trained[:, j], (x_div, -1))), cmap='viridis', extent=[self.x_min[0], self.x_max[0], self.x_min[1], self.x_max[1]], origin='lower', aspect='auto', norm=norms_3_4)

            # plot actual drift/diffusivity with normalization
            ax[j, 3].imshow(np.reshape(true_[:, j], (x_div, -1)), cmap='viridis', extent=[self.x_min[0], self.x_max[0], self.x_min[1], self.x_max[1]], origin='lower', aspect='auto', norm=norms_3_4)

            # add color bars
            cbar_1_2 = fig.colorbar(int, ax=ax[j, 0], orientation='vertical', fraction=0.02, pad=0.04)
            #cbar_1_2.set_label(f'Row {j+1} Color Scale (1 & 2)')

            cbar_3_4 = fig.colorbar(tr, ax=ax[j, 2], orientation='vertical', fraction=0.02, pad=0.04)
            #cbar_3_4.set_label(f'Row {j+1} Color Scale (3 & 4)')

            ax[j, 0].set_ylabel(r'${name}_{j}(x_0)$', fontsize=12)

        # format
        for axes in ax.flatten():
            axes.tick_params(axis='both', which='both', labelsize=6)
        plt.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.03)

        # Set labels
        ax[0, 0].set_title('Training Data', fontsize=12)
        ax[0, 1].set_title('Intermediate', fontsize=12)
        ax[0, 2].set_title('Trained', fontsize=12)
        ax[0, 3].set_title('True', fontsize=12)

        plt.show()
    
    @staticmethod
    def plot_loss(ve, moving_avg):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), gridspec_kw={'hspace': 0.12, 'wspace': 0.19})

        ax.semilogy(ve, label="Validation Error")
        ax.semilogy(moving_avg, label="Moving Average")

        ax.set_title('ARFF Loss', fontsize=12)
        ax.set_xlabel(r'$M$', fontsize=12)
        ax.legend()

        plt.show()


class theoretical_mean_loss_estimate:
    @staticmethod
    def integrand(*args):
        step_size = args[-2]
        true_diffusivity = args[-1]
        x = np.array(args[:-2])
        diffusivity = true_diffusivity(x)
        if diffusivity.ndim == 1:
            diffusivity = diffusivity[:, np.newaxis, np.newaxis]

        diffusivity_squared = np.dot(diffusivity, diffusivity.T)
        scaled_matrix = step_size * diffusivity_squared
        determinant = np.linalg.det(scaled_matrix)
        return np.log(np.abs(determinant))

    @staticmethod
    def get_TMLE(true_diffusivity, n_dimensions, n_pts, validation_split, xlim, step_size, input_dimensions=None):
        if input_dimensions == None:
            input_dimensions = n_dimensions
            
        bounds = [(xlim[0], xlim[1]) for i in range(input_dimensions)]
        integral_result, _ = nquad(theoretical_mean_loss_estimate.integrand, bounds, args=(step_size, true_diffusivity))

        domain_volume = np.prod([xlim[1] - xlim[0] for i in range(input_dimensions)])
        additive_constant = 2 * n_dimensions * np.log(2 * np.pi) / np.log(10)
        TMLE = 0.5 * integral_result / domain_volume + 0.5 * n_dimensions * (1 + np.log(2 * np.pi)) + additive_constant

        TMLSD = np.sqrt(0.5 * n_dimensions / (n_pts*(1 - validation_split)))
        TMLSD_val = np.sqrt(0.5 * n_dimensions / (n_pts*validation_split))
        return TMLE, TMLSD, TMLSD_val



