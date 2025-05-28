import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import scipy.spatial.distance

from sde.sde_learning_network import \
(
    SDEIntegrators
)


def sample_data(drift_diffusivity, step_size, n_dimensions, low, high, n_pts, rng,
                n_subsample=10, param_low=None, param_high=None):
    x_data = rng.uniform(low=low, high=high, size=(n_pts, n_dimensions))
    if param_high is not None:
        n_param = np.atleast_1d(param_high).shape[0]
        p_data = rng.uniform(low=param_low, high=param_high, size=(n_pts, n_param))
        y_data = x_data.copy()
        for k in range(n_subsample):
            y_data = np.row_stack([
                    SDEIntegrators.euler_maruyama(y_data[k, :],
                                                  step_size / n_subsample,
                                                  drift_diffusivity,
                                                  rng,
                                                  p_data[k, :])
                    for k in range(x_data.shape[0])
            ])

        return x_data, y_data, p_data
    else:
        y_data = x_data.copy()
        for k in range(n_subsample):
            y_data = np.row_stack([
                    SDEIntegrators.euler_maruyama(y_data[k, :],
                                                  step_size / n_subsample,
                                                  drift_diffusivity,
                                                  rng,
                                                  None)
                    for k in range(x_data.shape[0])
            ])

        return x_data, y_data


def plot_results_functions(apx_drift_diffusivity, true_drift_diffusivity,
                           x_data,
                           p_data=None,
                           data_transform_network=None,
                           data_transform_true=None,
                           fig=None, ax=None, color_approx="red", color_true="black", rows_per_function=1):
    if data_transform_network is None:
        def data_transform_network(x):
            return x
    if data_transform_true is None:
        def data_transform_true(x):
            return x

    mean_network, std_network = apx_drift_diffusivity(data_transform_network(x_data).astype(np.float32), p_data)

    mean_network = keras.backend.eval(mean_network)
    std_network = keras.backend.eval(std_network)

    if std_network.shape != mean_network.shape:
        std_network = keras.backend.eval(tf.linalg.diag_part(std_network))

    n_dimensions = x_data.shape[1]

    ms = 0.25  # marker size

    x_data_transformed = data_transform_true(x_data)
    true_drift_evaluated, true_std_evaluated = true_drift_diffusivity(x_data_transformed, p_data)

    if true_std_evaluated.shape != true_drift_evaluated.shape:
        true_std_evaluated = tf.linalg.diag_part(true_std_evaluated)

    if n_dimensions == 1:
        mean_network = mean_network.reshape(-1, 1)
        std_network = std_network.reshape(-1, 1)
        
        idx_ = np.argsort(x_data_transformed.ravel())

        if fig is None:
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].plot(x_data_transformed[idx_], true_drift_evaluated[idx_], "-", color=color_true, label="true")
        ax[0].plot(x_data_transformed[idx_], mean_network[idx_, 0], ".:", markevery=len(x_data_transformed)//21,
                   color=color_approx, label="approximated")
        ax[0].set_xlabel(r'$x_0$', fontsize=11)
        ax[0].set_ylabel("Drift")
        ax[0].legend()

        ax[1].plot(x_data_transformed[idx_], true_std_evaluated[idx_], "-", color=color_true, label="true")
        ax[1].plot(x_data_transformed[idx_], std_network[idx_, 0], ".:", markevery=len(x_data_transformed)//21,
                   color=color_approx, label="approximated")
        ax[1].set_xlabel(r'$x_0$', fontsize=11)
        ax[1].set_ylabel("Diffusion")
        ax[1].legend()
    else:
        figures_per_row = ((n_dimensions+rows_per_function-1) // rows_per_function)
        if fig is None:
            fig, ax = plt.subplots(2 * rows_per_function, figures_per_row, figsize=(n_dimensions * 3, 6 * rows_per_function))

        print("figures_per_row", figures_per_row)
        for k in range(n_dimensions):
            k_row = k // figures_per_row
            identity_pts = np.linspace(np.min([np.min(mean_network[:, k]), np.min(true_drift_evaluated)]),
                                       np.max([np.max(mean_network[:, k]), np.max(true_drift_evaluated)]),
                                       10)

            ax[k_row, k % figures_per_row].scatter(mean_network[:, k], true_drift_evaluated[:, k], s=ms, label="approximation")
            ax[k_row, k % figures_per_row].plot(identity_pts, identity_pts, 'k--', label="identity")
            ax[k_row, k % figures_per_row].set_xlabel(r"network drift $f_{" + str(k + 1) + "}$")
            ax[k_row, k % figures_per_row].set_ylabel(r"true drift $f_{" + str(k + 1) + "}$")
            ax[k_row, k % figures_per_row].legend()

            identity_pts = np.linspace(np.min([np.min(std_network[:, k]), np.min(true_std_evaluated)]),
                                       np.max([np.max(std_network[:, k]), np.max(true_std_evaluated)]),
                                       10)

            ax[rows_per_function + k_row, k % figures_per_row].scatter(std_network[:, k], true_std_evaluated[:, k], s=ms, label="approximation")
            ax[rows_per_function + k_row, k % figures_per_row].plot(identity_pts, identity_pts, 'k--', label="identity")
            ax[rows_per_function + k_row, k % figures_per_row].set_xlabel(r"network diffusion $\sigma_{" + str(k + 1) + "}$")
            ax[rows_per_function + k_row, k % figures_per_row].set_ylabel(r"true diffusion $\sigma_{" + str(k + 1) + "}$")
            ax[rows_per_function + k_row, k % figures_per_row].legend()
    fig.tight_layout()
    return fig, ax


def plot_parameter_functions(apx_drift_diffusivity, true_drift_diffusivity,
                             x_data,
                             p_data=None,
                             data_transform_network=None,
                             data_transform_true=None,
                             fig=None, ax=None, color_approx="red", color_true="black"):
    if data_transform_network is None:
        def data_transform_network(x):
            return x
    if data_transform_true is None:
        def data_transform_true(x):
            return x

    #mean_network, std_network = apx_drift_diffusivity(data_transform_network(x_data).astype(np.float32), p_data)
    mean_network, std_network = apx_drift_diffusivity(np.concatenate((x_data, p_data), axis=1), None)

    mean_network = keras.backend.eval(mean_network)
    std_network = keras.backend.eval(std_network)

    if std_network.shape != mean_network.shape:
        std_network = keras.backend.eval(tf.linalg.diag_part(std_network))

    n_dimensions = x_data.shape[1]

    ms = 0.25  # marker size

    x_data_transformed = data_transform_true(x_data)
    true_drift_evaluated, true_std_evaluated = true_drift_diffusivity(x_data_transformed, p_data)

    if true_std_evaluated.shape != true_drift_evaluated.shape:
        true_std_evaluated = tf.linalg.diag_part(true_std_evaluated)

    if n_dimensions == 1:
        mean_network = mean_network.reshape(-1, 1)
        std_network = std_network.reshape(-1, 1)
        
        idx_ = np.argsort(x_data_transformed.ravel())

        if fig is None:
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].plot(p_data[idx_, 0], true_drift_evaluated[idx_], "x", color=color_true, label="true")
        ax[0].plot(p_data[idx_, 0], mean_network[idx_, 0], ".", markevery=len(x_data_transformed)//100,
                   color=color_approx, label="approximated")
        ax[0].set_xlabel(r'$x_0$', fontsize=11)
        ax[0].set_ylabel("Drift")
        ax[0].legend()

        ax[1].plot(p_data[idx_, 0], true_std_evaluated[idx_], "x", color=color_true, label="true")
        ax[1].plot(p_data[idx_, 0], std_network[idx_, 0], ".", markevery=len(x_data_transformed)//100,
                   color=color_approx, label="approximated")
        ax[1].set_xlabel(r'$x_0$', fontsize=11)
        ax[1].set_ylabel("Diffusion")
        ax[1].legend()
    else:
        if fig is None:
            fig, ax = plt.subplots(2, n_dimensions, figsize=(n_dimensions * 3, 6))

        for k in range(n_dimensions):
            identity_pts = np.linspace(np.min([np.min(mean_network[:, k]), np.min(true_drift_evaluated)]),
                                       np.max([np.max(mean_network[:, k]), np.max(true_drift_evaluated)]),
                                       10)

            ax[0, k].scatter(mean_network[:, k], true_drift_evaluated[:, k], s=ms, label="approximation")
            ax[0, k].plot(identity_pts, identity_pts, 'k--', label="identity")
            ax[0, k].set_xlabel(r"network drift $f_" + str(k + 1) + "$")
            ax[0, k].set_ylabel(r"true drift $f_" + str(k + 1) + "$")
            ax[0, k].legend()

            identity_pts = np.linspace(np.min([np.min(std_network[:, k]), np.min(true_std_evaluated)]),
                                       np.max([np.max(std_network[:, k]), np.max(true_std_evaluated)]),
                                       10)

            ax[1, k].scatter(std_network[:, k], true_std_evaluated[:, k], s=ms, label="approximation")
            ax[1, k].plot(identity_pts, identity_pts, 'k--', label="identity")
            ax[1, k].set_xlabel(r"network diffusion $\sigma_" + str(k + 1) + "$")
            ax[1, k].set_ylabel(r"true diffusion $\sigma_" + str(k + 1) + "$")
            ax[1, k].legend()
    fig.tight_layout()
    return fig, ax


def histogram_data(drift_diffusivity, step_size, time, n_dimensions, random_seed, ylim, plim=None, coupled_func=None):

    rng = np.random.default_rng(random_seed)
    
    # Parameters
    sim_No = 10000
    active_indices = np.arange(sim_No)
    I = int(time / step_size)  # Number of time steps

    # Array to store all simulations
    if plim is None: 
        xlim = ylim
    else:
        xlim = np.concatenate([ylim, plim], axis=0)
    X = np.zeros((sim_No, xlim.shape[0], I + 1))  
    
    # Initialize starting points 
    X[:, :, 0] = rng.uniform(xlim[:, 0], xlim[:, 1], size=(sim_No, xlim.shape[0]))
    #print(I)
    # Simulate all paths simultaneously
    for i in range(I):
        #print(i)
        X_active = X[active_indices, :, i]

        #print('first', X_active)
        # perform coupled equation e.g p1 = p0 + h*x0
        if plim is not None:
            Y_active = X_active[:, :ylim.shape[0]]
            P_active = X_active[:, -plim.shape[0]:]
            P_active = coupled_func(Y_active, P_active, step_size)
            X_active[:, -plim.shape[0]:] = P_active
            X[active_indices, -plim.shape[0]:, i + 1] = P_active

        #print('second', X_active)
        
        # if Type == "coupled":
        #     X_active_ = X_active[~outliers]
        #     X_active_[:, 1] += X_active_[:, 0]*step_size
        #     X[active_indices, 1, i + 1] = X_active_[:, 1]

        # get euler-maruyama components
        dW = rng.normal(loc=0, scale=np.sqrt(step_size), size=(len(active_indices), n_dimensions))
        drift_, diff_ = drift_diffusivity(X_active, None)
        drift_ = drift_.reshape(-1, n_dimensions)          
        diff_ = diff_.reshape(-1, n_dimensions, n_dimensions)
        
        if plim is None:
            X[active_indices, :, i + 1] = X_active + step_size * drift_ + np.einsum('ijk,ik->ij', diff_, dW)
        else:
            X[active_indices, :ylim.shape[0], i + 1] = (X_active[:, :ylim.shape[0]] + step_size * drift_ + np.einsum('ijk,ik->ij', diff_, dW))
        
        # if Type == "coupled":            
        #     X[active_indices, 0, i + 1] = (X_active_[:, 0].reshape(-1, 1) + step_size * drift_ + np.einsum('ijk,ik->ij', diff_, dW)).reshape(-1)
        # else:
        #     X[active_indices, :ylim.shape[0], i + 1] = X_active[~outliers] + step_size * drift_ + np.einsum('ijk,ik->ij', diff_, dW)


        outlier_mask = ((X[active_indices, :, i + 1] <= xlim[:, 0]) | (X[active_indices, :, i + 1] >= xlim[:, 1])).any(axis=1)
    
        # Mark outliers as NaN for all future steps
        X[active_indices[outlier_mask], :, (i+1):] = np.NaN
    
        # Update the active set
        active_indices = active_indices[~outlier_mask]
        if len(active_indices) == 0:
            break

    
    return X


def plot_histogram(X, step_size, ylim, plim=None):
    if plim is None: 
        xlim = ylim
    else:
        xlim = np.concatenate([ylim, plim], axis=0)
        
    sim_No, dim, I = X.shape
    Y = np.tile(np.arange(I), sim_No)

    fig, axes = plt.subplots(dim, 1, figsize=(5, 4 * dim), gridspec_kw={'hspace': 0.12, 'wspace': 0.19})

    # Ensure axes is always a 1D array, even if dim == 1
    if dim == 1:
        axes = [axes]  # Wrap single AxesSubplot into a list
    
    # Plot a histogram for each dimension
    for d in range(dim):
        X_flat = X[:, d, :].flatten()  # Flatten the dimension d across time
        bins_0 = np.arange(-0.5, I + 0.5, 1)
        bins_1 = np.linspace(xlim[d, 0], xlim[d, 1], 100)
        hist, x_edges, y_edges = np.histogram2d(
            Y, X_flat, 
            bins=[bins_0, bins_1]
        )

        # Plot the histogram
        axes[d].imshow(
            hist.T,  # Transpose because imshow assumes (row, col)
            origin='lower',  # Origin at the bottom-left
            cmap='inferno',  # Color map
            aspect='auto',   # Automatic aspect ratio
            extent=[-0.5 * step_size, (I - 0.5) * step_size, xlim[d, 0], xlim[d, 1]]  # Set axis limits to match bins
        )
        #plt.colorbar(label="Frequency")  # Add a color bar
        axes[d].set_ylabel(f"$x_{d}$")  # Label for dimension d

    fig.supxlabel("Time (s)")
    plt.show()


def histogram_data_ex6(drift_diffusivity, true_paths, step_sizes_layered, n_dimensions, rng, ARFF):
    X = [0] * (len(step_sizes_layered) + 1)
    X[0] = np.array([path[0] for path in true_paths])
    
    for l in range(len(step_sizes_layered)):
        dW = step_sizes_layered[l].reshape(-1,1) * rng.normal(loc=0, scale=1, size=(len(step_sizes_layered[l]), n_dimensions))
        drift_, diff_ = drift_diffusivity(X[l][:len(step_sizes_layered[l])], None)

        if not ARFF:
            diff_new = np.zeros((len(step_sizes_layered[l]), n_dimensions, n_dimensions))
            indices = np.arange(n_dimensions)
            diff_new[:, indices, indices] = diff_
            diff_ = diff_new
        
        X[l+1] = X[l][:len(step_sizes_layered[l])] + step_sizes_layered[l].reshape(-1,1) * drift_ + np.einsum('ijk,ik->ij', diff_, dW)

    return X
    

def plot_histogram_ex6(y, times,  max_time):

    times_all = np.concatenate(times)  
    y_all = np.concatenate(y, axis=0)  

    bins_0 = np.linspace(times_all.min() + 0.01, max_time, 100)
    bins_1 = np.linspace(0, 1, 100)
    
    fig, axes = plt.subplots(2, 1, figsize=(5, 8), gridspec_kw={'hspace': 0.12})
    
    for d in range(2):
        hist, x_edges, y_edges = np.histogram2d(times_all, y_all[:, d], bins=[bins_0, bins_1])
        
        axes[d].imshow(
            hist.T,
            origin='lower',
            cmap='inferno',
            aspect='auto',
            extent=[times_all.min(), max_time, 0, 1]
        )
        axes[d].set_ylabel(f"$y_{d}$")
    
    fig.supxlabel("Time (s)")
    plt.show()



