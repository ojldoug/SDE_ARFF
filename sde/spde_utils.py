import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from mpl_toolkits.axes_grid1 import make_axes_locatable

import keras
import keras.backend as K

import tensorflow_probability as tfp

import numpy as np


class SPDEUtils:
    """
    Provides several utility methods for learning SPDEs.
    """

    @staticmethod
    def integrate_stochastic_wave(u0, v0, f, g, time, space, random_seed=1, periodic_boundary=False):
        """
        Numerical scheme to integrate the (deterministically and stochastically) forced wave equation.
        From: https://personal.math.ubc.ca/~walsh/waveq.pdf

        u0: initial values for u
        v0: initial values for du/dt
        f: deterministic forcing, Lipschitz
        g: stochastic forcing, Lipschitz. Will be multiplied with the noise term $\dot{W}$.
        time: time linspace.
        space: space linspace.
        random_seed: seed for the noise term, default: 1.
        """
        rng = np.random.default_rng(random_seed)
        step_size = time[1] - time[0]
        square_size = 2 * step_size ** 2
        noise_std = np.sqrt(square_size)

        # odd and even index grid
        idx_even = np.arange(len(space))[::2]
        idx_odd = np.arange(len(space))[1::2]

        # create buffer
        u0_tmp = np.zeros((len(idx_even) + 2,))
        u0_large = u0_tmp.copy()
        u0_large[1:-1] = u0[idx_even]
        if periodic_boundary:
            u0_large[0] = u0[idx_even][-1]
            u0_large[-1] = u0[idx_even][0]
        else:
            u0_large[0] = 0
            u0_large[-1] = 0

        # approximate integral at t=-h over v0 by constant
        u0_m1 = u0_large[:-2]
        u0_p1 = u0_large[2:]
        um1_i = 0.5 * (u0_m1 + u0_p1) - step_size * v0[idx_odd]

        f0 = f(space[idx_even], 0, 0.5 * (u0_m1 + u0_p1))
        g0 = g(space[idx_even], 0, 0.5 * (u0_m1 + u0_p1))
        W0 = rng.normal(loc=0, scale=noise_std / np.sqrt(2), size=(len(idx_even),))
        u1_i = u0_m1 + u0_p1 - um1_i + step_size ** 2 / 2 * f0 + 0.5 * g0 * W0

        u = np.zeros((len(time), len(idx_even)))
        u[0, :] = um1_i  # t = -h
        u[1, :] = u0[idx_even]  # t =  0
        u[2, :] = u1_i  # t = +h
        for j in range(2, len(time) - 1):
            if j % 2 == 0:
                idx_jm1 = idx_even
                idx_j = idx_odd
            else:
                idx_jm1 = idx_odd
                idx_j = idx_even

            u0_large = u0_tmp.copy()
            u0_large[1:-1] = u[j, :]
            if periodic_boundary:
                u0_large[0] = u[j, :][-1]
                u0_large[-1] = u[j, :][0]
            else:
                u0_large[0] = 0
                u0_large[-1] = 0
            uim1_t = u0_large[:-2]
            uip1_t = u0_large[2:]
            ft = f(space[idx_j], time[j], 0.5 * (uim1_t + uip1_t))
            gt = g(space[idx_j], time[j], 0.5 * (uim1_t + uip1_t))
            Wt = rng.normal(loc=0, scale=noise_std, size=(len(idx_jm1),))
            u[j + 1, :] = uim1_t + uip1_t - u[j - 1, :] + step_size ** 2 * ft + 0.5 * gt * Wt
        return u

    @staticmethod
    def split_forced_wave_spde_data(ut, space, time):
        """
        Split the given solution data u(t) into the form required by the SODE learning framework.
        The split is based on the numerical integration scheme from "integrate_stochastic_wave".

        Parameters
        ----------
        ut solution, with (NT+2) * (NX+2) rows and columns (time * space).
        space the spatial coordinates used in the solution. Shape (NX+2,)
        time the temporal coordinates used in the solution. Shape (NT+2,)
        step_size time step size between ut[1,:] and ut[2,:]

        Returns
        -------
        u_{n}    "current state" 0.5*(u[j,i-1]+u[j,i+1])
        p_{n}    "current parameters" (time[j], space[i])
        u_{n+1}  "next state" (u[j+1,i] + u[j-1,i]) / step_size
        """
        u_jm1 = []
        u_j = []
        p_n = []
        u_np1 = []
        for j in range(1, ut.shape[0] - 1):
            u_jm1.append(ut[j - 1, 1:-1])
            u_j.append([ut[j, :-2] + ut[j, 2:]])
            u_np1.append(ut[j + 1, 1:-1])

            p_j = np.zeros((len(space)-2, 2))
            p_j[:, 0] = time[j]
            p_j[:, 1] = space[1:-1]
            p_n.append(p_j)
        u_jm1 = np.row_stack(u_jm1).ravel()
        u_j = np.row_stack(u_j).ravel()
        u_np1 = np.row_stack(u_np1).ravel()
        p_n = np.row_stack(p_n)

        # reformulate using the numerical scheme, so that the SDE learning framework can be applied
        u_np1 = 0.5 * (u_np1 + u_jm1)
        return 0.5 * u_j.reshape(-1, 1), p_n, u_np1.reshape(-1, 1)

    def plot_data(ut, space, time):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
    
        im1 = ax.imshow(ut, extent=[np.min(space), np.max(space), np.min(time), np.max(time)], origin='lower')#, vmin=-1, vmax=1)
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        fig.colorbar(im1, cax=cax, orientation='vertical')
        fig.tight_layout()

        plt.show()

