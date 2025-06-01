import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import scipy.spatial.distance
from scipy.integrate import nquad
import os

from sde.sde_learning_network import \
(
    SDEIntegrators
)


def euler_maruyama_batch(drift_diffusion, step_size, rng, yk, pk=None):
    if pk is not None:
        xk = np.concatenate([yk, pk], axis=1)
    else:
        xk = yk

    dW = rng.normal(loc=0, scale=np.sqrt(step_size), size=yk.shape)
    fk, sk = drift_diffusion(xk)  # expect fk: (N, d), sk: (N, d) or (N, d, d)

    if sk.ndim == 2: # if (N, d) not (N, d, d) 
        skW = sk * dW 
    else:
        skW = np.einsum('nij,nj->ni', sk, dW)

    return yk + step_size * fk + skW


def sample_data(drift_diffusion, step_size, n_pts, n_subsample, rng, ylim, plim=None):
    y_np_data = rng.uniform(low=ylim[:,0], high=ylim[:,1], size=(n_pts, ylim.shape[0]))
    y_np1_data = y_np_data.copy()

    if plim is not None:
        p_data = rng.uniform(low=plim[:,0], high=plim[:,1], size=(n_pts, plim.shape[0]))
    else:
        p_data = None

    for _ in range(n_subsample):
        y_np1_data = euler_maruyama_batch(drift_diffusion, step_size / n_subsample, rng, y_np1_data, p_data)

    return y_np_data, y_np1_data, p_data


class PlotResults:
    def __init__(self, script_dir, filename=None, n_subsample=1):
        self.script_dir = script_dir
        self.filename = filename
        self.n_subsample = n_subsample

    @staticmethod
    def integrand(*args):
        step_size = args[-2]
        true_diffusion = args[-1]
        x = np.array(args[:-2])
        
        diffusion = true_diffusion(x)
        diffusion = np.atleast_2d(diffusion)
    
        diffusion_matrix = diffusion @ diffusion.T
        scaled_matrix = step_size * diffusion_matrix
        det = np.linalg.det(scaled_matrix)
            
        return np.log(det)
    
    
    def mean_min_loss(self, true_diffusion, n_pts, validation_split, step_size, ylim, xlim=None, YinX=True, save=False):
        if xlim is None: 
            xlim = ylim
        elif YinX:
            xlim = np.concatenate([ylim, xlim], axis=0)
    
        bounds = [(xlim[i, 0], xlim[i, 1]) for i in range(xlim.shape[0])]
        integral_result, _ = nquad(PlotResults.integrand, bounds, args=(step_size, true_diffusion))
    
        domain_volume = np.prod([xlim[i, 1] - xlim[i, 0] for i in range(xlim.shape[0])])
        MML = 0.5 * integral_result / domain_volume + 0.5 * ylim.shape[0] * (1 + np.log(2 * np.pi))
    
        SD = np.sqrt(0.5 * ylim.shape[0] / (n_pts*(1 - validation_split)))
        SD_val = np.sqrt(0.5 * ylim.shape[0] / (n_pts*validation_split))
        
        print('Theoretical mean min loss:', MML)
        print('Loss standard deviation:', SD)
        print('Validation loss standard deviation:', SD_val)
        if save:
            output_dir = os.path.join(self.script_dir, 'saved_results/loss_v_time_data')
            output_path = os.path.join(output_dir, f"{self.filename}_SS{self.n_subsample}_min_loss.txt")
            with open(output_path, 'w') as file:
                file.write(f"{MML}\n")
        

        


    
    def plot_results_functions(self, apx_drift_diffusivity, true_drift_diffusivity,
                               x_data,
                               p_data=None,
                               data_transform_network=None,
                               data_transform_true=None,
                               fig=None, ax=None, color_approx="red", color_true="black", rows_per_function=1, save=False):
        if data_transform_network is None:
            def data_transform_network(x):
                return x
        if data_transform_true is None:
            def data_transform_true(x):
                return x
    
        mean_network, std_network = apx_drift_diffusivity(data_transform_network(x_data).astype(np.float32))
    
        mean_network = keras.backend.eval(mean_network)
        std_network = keras.backend.eval(std_network)
    
        if std_network.shape != mean_network.shape:
            std_network = keras.backend.eval(tf.linalg.diag_part(std_network))
    
        n_dimensions = x_data.shape[1]
    
        ms = 0.25  # marker size
    
        x_data_transformed = data_transform_true(x_data)
        true_drift_evaluated, true_std_evaluated = true_drift_diffusivity(x_data_transformed)
    
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
    
        if save:
            output_dir = os.path.join(self.script_dir, 'saved_results/trained_v_true_plots')
            output_path = os.path.join(output_dir, f"{self.filename}_SS{self.n_subsample}.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

    def plot_parameter_functions(self, apx_drift_diffusivity, true_drift_diffusivity,
                                 x_data,
                                 p_data=None,
                                 data_transform_network=None,
                                 data_transform_true=None,
                                 fig=None, ax=None, color_approx="red", color_true="black", save=False):
        if data_transform_network is None:
            def data_transform_network(x):
                return x
        if data_transform_true is None:
            def data_transform_true(x):
                return x
    
        #mean_network, std_network = apx_drift_diffusivity(data_transform_network(x_data).astype(np.float32), p_data)
        mean_network, std_network = apx_drift_diffusivity(np.concatenate((x_data, p_data), axis=1))
    
        mean_network = keras.backend.eval(mean_network)
        std_network = keras.backend.eval(std_network)
    
        if std_network.shape != mean_network.shape:
            std_network = keras.backend.eval(tf.linalg.diag_part(std_network))
    
        n_dimensions = x_data.shape[1]
    
        ms = 0.25  # marker size
    
        x_data_transformed = data_transform_true(x_data)
        true_drift_evaluated, true_std_evaluated = true_drift_diffusivity(np.concatenate((x_data_transformed, p_data), axis=1))
    
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
            ax[0].set_xlabel("Parameter")
            ax[0].set_ylabel("Drift")
            ax[0].legend()
    
            ax[1].plot(p_data[idx_, 0], true_std_evaluated[idx_], "x", color=color_true, label="true")
            ax[1].plot(p_data[idx_, 0], std_network[idx_, 0], ".", markevery=len(x_data_transformed)//100,
                       color=color_approx, label="approximated")
            ax[1].set_xlabel("Parameter")
            ax[1].set_ylabel("Diffusivity")
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
                ax[1, k].set_xlabel(r"network diffusivity $\sigma_" + str(k + 1) + "$")
                ax[1, k].set_ylabel(r"true diffusivity $\sigma_" + str(k + 1) + "$")
                ax[1, k].legend()
        fig.tight_layout()
        
        if save:
            output_dir = os.path.join(self.script_dir, 'saved_results/trained_v_true_plots')
            output_path = os.path.join(output_dir, f"{self.filename}_SS{self.n_subsample}.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

    @staticmethod
    def histogram_data(drift_diffusion, step_size, time, rng, ylim, plim, coupled_func, sim_No):
        
        # Parameters
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
    
        # Simulate all paths simultaneously
        for i in range(I):
            X_active = X[active_indices, :, i]
    
            # perform coupled equation e.g p1 = p0 + h*x0
            if plim is not None:
                Y_active = X_active[:, :ylim.shape[0]]
                P_active = X_active[:, -plim.shape[0]:]
                P_active = coupled_func(Y_active, P_active, step_size)
                X_active[:, -plim.shape[0]:] = P_active
                X[active_indices, -plim.shape[0]:, i + 1] = P_active
    
            # get euler-maruyama components
            dW = rng.normal(loc=0, scale=np.sqrt(step_size), size=(len(active_indices), ylim.shape[0]))
            drift_, diff_ = drift_diffusion(X_active)
            drift_ = drift_.reshape(-1, ylim.shape[0])          
            diff_ = diff_.reshape(-1, ylim.shape[0], ylim.shape[0])
            
            if plim is None:
                X[active_indices, :, i + 1] = X_active + step_size * drift_ + np.einsum('ijk,ik->ij', diff_, dW)
            else:
                X[active_indices, :ylim.shape[0], i + 1] = (X_active[:, :ylim.shape[0]] + step_size * drift_ + np.einsum('ijk,ik->ij', diff_, dW))
            
            outlier_mask = ((X[active_indices, :, i + 1] <= xlim[:, 0]) | (X[active_indices, :, i + 1] >= xlim[:, 1])).any(axis=1)
        
            # Mark outliers as NaN for all future steps
            X[active_indices[outlier_mask], :, (i+1):] = np.NaN
        
            # Update the active set
            active_indices = active_indices[~outlier_mask]
            if len(active_indices) == 0:
                break
    
        return X
    
    def plot_histogram(self, drift_diffusion, step_size, time, rng, ylim, plim=None, coupled_func=None, sim_No=10000, name=None, save=False):
        X = PlotResults.histogram_data(drift_diffusion, step_size, time, rng, ylim, plim, coupled_func, sim_No)
        
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
            axes[d].set_ylabel(f"$x_{d}$", fontsize=12)  # Label for dimension d

        axes[0].set_title(f"{name}")
        axes[-1].set_xlabel("Time (s)", fontsize=11)
        plt.show()

        if save:
            output_dir = os.path.join(self.script_dir, 'saved_results/trajectory_histograms')
            output_path = os.path.join(output_dir, f"{self.filename}_SS{self.n_subsample}_{name}.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    def loss_stats(self, TT, VL, save=False):
        # Calculate the mean
        mean_TT = np.mean(TT)
        mean_VL = np.mean(VL)
        
        # Calculate the standard deviation above and below the mean for training_time
        TT_above = TT[TT > mean_TT] - mean_TT
        TT_below = mean_TT - TT[TT < mean_TT]
        
        std_TT_above = np.std(np.concatenate((TT_above, -TT_above)))
        std_TT_below = np.std(np.concatenate((TT_below, -TT_below)))
        
        # Calculate points above and below the mean for val_loss
        VL_above = VL[VL > mean_VL] - mean_VL
        VL_below = mean_VL - VL[VL < mean_VL]
        
        std_VL_above = np.std(np.concatenate((VL_above, -VL_above)))
        std_VL_below = np.std(np.concatenate((VL_below, -VL_below)))
        
        # Plot scatter points
        plt.scatter(TT, VL, color='blue', label='Data Points')
        
        # Add non-symmetrical error bars (standard deviations)
        plt.errorbar(
            mean_TT, mean_VL,
            xerr=[[std_TT_below], [std_TT_above]],  # Non-symmetrical x error
            yerr=[[std_VL_below], [std_VL_above]],  # Non-symmetrical y error
            fmt='x', color='red', ecolor='black', elinewidth=1.5, capsize=4, label='Mean ± STD'
        )

        plt.ylabel("Min Loss")
        plt.xlabel("Training Time (s)")
        plt.show()

        print("Mean Min Loss: ", mean_VL)
        print("Mean Training Time: ", mean_TT, "s")
        if save:
            output_dir = os.path.join(self.script_dir, 'saved_results/loss_v_time_data')
            output_path = os.path.join(output_dir, f"{self.filename}_SS{self.n_subsample}.txt")
            with open(output_path, 'w') as file:
                file.write(f"{mean_TT},{mean_VL},{std_TT_above},{std_TT_below},{std_VL_above},{std_VL_below}\n")

    def loss_v_time(self, TT, VL, save=False):
        N_EPOCHS = TT.shape[1]
        
        # Initialize arrays to store results
        mean_TT = np.mean(TT, axis=0)
        mean_VL = np.mean(VL, axis=0)
        std_TT_above = np.zeros(N_EPOCHS)
        std_TT_below = np.zeros(N_EPOCHS)
        std_VL_above = np.zeros(N_EPOCHS)
        std_VL_below = np.zeros(N_EPOCHS)
        
        # Iterate over epochs
        for i in range(N_EPOCHS):
            TT_ = TT[:, i]
            VL_ = VL[:, i]
        
            # Calculate deviations for training_time
            TT_above = TT_[TT_ > mean_TT[i]] - mean_TT[i]
            TT_below = mean_TT[i] - TT_[TT_ < mean_TT[i]]
            std_TT_above[i] = np.std(np.concatenate((TT_above, -TT_above)))
            std_TT_below[i] = np.std(np.concatenate((TT_below, -TT_below)))
        
            # Calculate deviations for val_loss
            VL_above = VL_[VL_ > mean_VL[i]] - mean_VL[i]
            VL_below = mean_VL[i] - VL_[VL_ < mean_VL[i]]
            std_VL_above[i] = np.std(np.concatenate((VL_above, -VL_above)))
            std_VL_below[i] = np.std(np.concatenate((VL_below, -VL_below)))
        
        plt.errorbar(
            mean_TT, mean_VL,
            xerr=[std_TT_below, std_TT_above],  # Non-symmetrical x error
            yerr=[std_VL_below, std_VL_above],  # Non-symmetrical y error
            fmt='o', color='red', ecolor='black', elinewidth=1.5, capsize=4, label='Mean ± STD'
        )
        
        plt.xlabel('Mean Training Time')
        plt.ylabel('Mean Validation Loss')
        plt.title('Error Bars for Training Time and Validation Loss Across Epochs')
        plt.legend()
        plt.grid(True)
    
        # Optionally save
        if save:
            data = pd.DataFrame({
                "cum_time": mean_TT,
                "loss": mean_VL,
                "std_training_time_above": std_TT_above,
                "std_training_time_below": std_TT_below,
                "std_loss_above": std_VL_above,
                "std_loss_below": std_VL_below
            })
            output_dir = os.path.join(self.script_dir, 'saved_results/loss_v_time_data')
            output_path = os.path.join(output_dir, f"{self.filename}_SS{self.n_subsample}.csv")
            data.to_csv(output_path, index=False)



























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
























    # moving_avg = np.zeros(N_EPOCHS)
    # min_moving_avg = float('inf')
    # moving_avg_len = 5
    # min_index = 0
    # break_iterations = 5
    # for j in range(N_EPOCHS):
    #     if j < moving_avg_len:
    #         moving_avg[j] = np.mean(val_losses[i,:j+1])
    #     else:
    #         moving_avg[j] = np.mean(val_losses[i,j-moving_avg_len+1:j+1])

    #     if moving_avg[j] < min_moving_avg:
    #         min_moving_avg = moving_avg[j]
    #         min_index = j

    #     if min_index + break_iterations < j:
    #         break

    # val_loss_array = val_losses[i,:j]
    # val_loss_min_index = np.argmin(val_loss_array)
    # training_time[i] = cumulative_times[i,val_loss_min_index]
    # val_loss[i] = val_losses[i,val_loss_min_index]

























