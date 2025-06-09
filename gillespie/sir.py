
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle
from itertools import zip_longest
import os

import gillespie.gillespie_c as gsc

from joblib import Parallel, delayed

"""
Original MATLAB code from Alexei

% Gillespie algorithm for the SIRS sde_model

function SA_SIRS_1

N = 1024;       % system size; number of sites
%NActs = 3;      % number of elementary events
k1 = 1.0;       % rate constant of event 1
k2 = 1.0;       % rate constant of event 2
k3 = 0.0;       % rate constant of event 3

y0(1) = 0.02;   % initial condition for y1 (I, infected species)
y0(2) = 0.00;   % initial condition for y2 (R, recovered species)

time_max = 4;  % max time
tstep = 0.01;     % output dt
rng('shuffle','twister');           % seed for RNG
%rand('state',1);
%______________________________

k1 = 4.0*k1;
CN = 1.0 / N;
curtime = 0.0;
NP = time_max/tstep;
time = zeros(NP,1);
y = zeros(NP,2);
N1 = uint64(y0(1)*N); 
N2 = uint64(y0(2)*N);
i = 0;

while (curtime <= time_max)
    
    % calculate all rates and their Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))sum
    y1 = double(N1) * CN;    % I concentration
    y2 = double(N2) * CN;    % R concentration
    R(1) = k1*y1*(1-y1-y2);  % I + S --> I + I 
    R(2) = k2*y1;            % I --> R
    R(3) = k3*y2;            % R --> S
    RSum = R(1)+R(2)+R(3);
    
    % call RNG (0,1)
    x = rand*RSum;
    
    % select one elementary event
    RA = R(1);
    Act = 1;
    while (RA < x)
        Act = Act + 1;
        RA = RA + R(Act);
    end;
    
    % update N's according to the selected event 
    switch Act 
    case 1
        N1 = N1 + 1;
    case 2
        N1 = N1 - 1;
        N2 = N2 + 1;
    case 3
        N2 = N2 - 1;
    end
    
    % update time
    dt = -log(rand)/(RSum*N);
    %dt = 1.0/(RSum*N);
    curtime = curtime + dt;
    
    % save solution
    if (curtime >= tstep*i)
        i = i + 1;
        time(i) = curtime;
        y(i, 1) = y1;
        y(i, 2) = y2;
    end
    
end

NP = i;
figure;
plot(time(1:NP),y(1:NP,1),'black', 'LineWidth', 2);
hold on
plot(time(1:NP),y(1:NP,2),'red', 'LineWidth', 2);


fid = fopen('SA_SIRS_1.dat', 'w');
for j = 1 : NP;
  fprintf(fid, '%.12e %.12e %.12e\r\n', time(j), y(j,1), y(j,2));
end;
fclose(fid);

var(y)
mean(y)
%__________________________________________________________________________


"""
    

class SIRG:
    """
    Python implementation of the SIR Gillespie simulation.
    """
    
    def __init__(self, N = 1024, k1 = 1.0, k2 = 1.0, k3 = 0.0, random_state = 1):
        self.N = N       # system size; number of sites
        self.k1 = k1       # rate constant of event 1
        self.k2 = k2       # rate constant of event 2
        self.k3 = k3       # rate constant of event 3
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        
    def simulate(self, y0, time_max=4, time_step=0.01, parallel=False):
        if len(y0.shape)==1 or y0.shape[0]==1:
            t_y = self.simulate_single(y0, time_max, time_step)
            return t_y[:,0], t_y[:,1:]
        
        if parallel:
            # use all but one CPU core
            time_y_list = Parallel(n_jobs=-2)(delayed(lambda k: self.simulate_single_c(y0[k,:], time_max, time_step, rng=np.random.default_rng(self.random_state+k)))(i) for i in range(y0.shape[0]))
        
            time = [time_y_list[k][:,0] for k in range(len(time_y_list))]
            y = [time_y_list[k][:,1:] for k in range(len(time_y_list))]
        else:
            time, y = gsc.simulate(y0, self.k1, self.k2, self.k3, self.N, self.random_state, time_max, time_step)
                
        return time, y
    
    def simulate_single_c(self, y0, time_max=4, time_step=0.01, rng=None):
        if rng is None:
            rng = self.rng
        return gsc.simulate_single(y0, self.k1, self.k2, self.k3, self.N, rng, time_max=time_max, time_step=time_step)
        
    def simulate_single(self, y0, time_max=4, time_step=0.01, rng=None):
        """
        y0[0]     initial condition for y1 (I, infected species)
        y0[1]     initial condition for y2 (R, recovered species)
        time_max  max time
        tstep     output dt
        """
        
        if rng is None:
            rng = self.rng
        
        # initialize internal parameters
        k1 = 4.0*self.k1
        k2 = self.k2
        k3 = self.k3
        
        CN = 1.0 / self.N
        curtime = 0.0
        NP = int(np.ceil(time_max/time_step))
        time = np.zeros((NP,))
        y = np.zeros((NP,2))
        R = np.zeros((3,))
        N1 = int(y0[0]*self.N)
        N2 = int(y0[1]*self.N)
        i = 0
        
        while (curtime <= time_max):
    
            # calculate all rates and their sum
            y1 = np.clip(N1 * CN, 0, 1);    # I concentration
            y2 = np.clip(N2 * CN, 0, 1);    # R concentration
            R[0] = k1*y1*(1-y1-y2);  # I + S --> I + I 
            R[1] = k2*y1;            # I --> R
            R[2] = k3*y2;            # R --> S
            RSum = np.sum(R)
            
            if RSum == 0: # happens if y1 is zero
                break
            if i >= NP:
                break

            # call RNG (0,1)
            x = rng.uniform(0,1)*RSum;

            # select one elementary event
            RA = R[0]
            Act = 0 # python is zero based...
            while (RA < x and Act < len(R)-1):
                Act = Act + 1
                RA = RA + R[Act]

            # update N's according to the selected event 
            # Python does not have a switch/case keyword structure
            if Act==0:
                    N1 = N1 + 1
            if Act==1:
                    N1 = N1 - 1
                    N2 = N2 + 1
            if Act==2:
                    N2 = N2 - 1

            # update time (clip the argument to be on the safe side)
            if RSum == 0:
                print("RSum error: is zero...")
                RSum = 1
            dt = -np.log(rng.uniform(1e-10,1))/(RSum*self.N)
            # dt = 1.0/(RSum*self.N);
            curtime = curtime + dt

            # save solution
            if (curtime >= time_step*i):
                time[i] = curtime
                y[i, 0] = y1
                y[i, 1] = y2
                i = i + 1
                
        time = time[0:i]
        y = y[0:i,:]
        return np.column_stack([time, y])

    def sample_data_SIRG(self, n_trajectories, time_max, time_step, n_skip_steps):
        y, time_g = SIRG.generate_trajectories(self, n_trajectories, time_max, time_step, n_skip_steps)

        x_data = []
        y_data = []
        times = []
        step_sizes = []
        
        for k in range(len(y)):
            if len(time_g[k]) > 2:
                times.extend(time_g[k][:-1])
                step_sizes.extend(np.gradient(time_g[k])[:-1])
                x_data.append(y[k][:-1,:])
                y_data.append(y[k][1:,:])
        
        x_data = np.row_stack(x_data)
        y_data = np.row_stack(y_data)
        step_sizes = np.array(step_sizes).reshape(-1, 1)
        times = np.array(times)
        
        # work with (theta0,theta2) data instead of (theta1,theta2) data (the SDE equations we use to compare later are in terms of theta0,2)
        # x_data = SIRG.theta02(x_data)
        # y_data = SIRG.theta02(y_data)
    
        return x_data, y_data, step_sizes
    
    def generate_trajectories(self, n_trajectories, time_max, time_step, n_skip_steps):
        
        y0_all = []
        
        for k in range(n_trajectories):
            # randomly sample the unit cube
            y0 = self.rng.uniform(low=0.0, high=1.0, size=(3,))
        
            # transform to sample more points on the boundary
            # y0 = (np.tanh((y0-.5)*5)+1)/2
        
            # make sure we only sample admissible initial conditions
            y0 = np.clip(y0, 0, 1)
            y0 = y0/np.sum(y0)
        
            # only take the two
            y0 = y0[:2]  
        
            y0_all.append(y0)
            
        y0_all = np.row_stack(y0_all)
        time_g, y = SIRG.simulate(self, y0_all, time_max=time_max, time_step=time_step)
    
        for k in range(len(y)):
            # skip simulated time steps for the training data, so that the individual points are further apart in time
            time_g[k] = time_g[k][::n_skip_steps]
            y[k] = y[k][::n_skip_steps,:]

        # work with (theta0,theta2) data instead of (theta1,theta2) data (the SDE equations we use to compare later are in terms of theta0,2)
        y = SIRG.theta02(y)

        return y, time_g
            
    # @staticmethod
    # def theta02(traj):
    #     """
    #     Convert theta12 data to theta02 data (theta0 has "more interesting" trajectories)
    #     """
    #     traj_new = traj.copy()
    #     traj_new[:,0] = 1-(traj[:,0]+traj[:,1])
    #     return traj_new

    @staticmethod
    def theta02(traj):
        """
        Convert theta12 data to theta02 data (theta0 has "more interesting" trajectories)
        """
        traj_new = []
        for t in traj:
            t_new = t.copy()
            t_new[:, 0] = 1 - (t[:, 0] + t[:, 1])
            traj_new.append(t_new)
        return traj_new


def histogram_data_SIRG(drift_diffusion, y, time_g, n_dimensions, rng, ARFF):
    # filter out paths with fewer than 2 entries
    y_filtered = []
    time_g_filtered = []
    for y_i, t_i in zip(y, time_g):
        if y_i.shape[0] >= 2:
            y_filtered.append(y_i)
            time_g_filtered.append(t_i)
    
    # compute step sizes for each path
    step_sizes = [np.gradient(t)[:-1] for t in time_g_filtered]
    step_lengths = [len(s) for s in step_sizes]

    # sort paths by descending step size length
    sorted_indices = np.argsort(step_lengths)[::-1]
    y_sorted = [y_filtered[i] for i in sorted_indices]
    time_g_sorted = [time_g_filtered[i] for i in sorted_indices]
    step_sizes_sorted = [step_sizes[i] for i in sorted_indices]
    
    # layer time, step sizes, and data
    step_sizes_layered = [np.array([x for x in group if x is not None]) for group in zip_longest(*step_sizes_sorted)]
    time_g_layered = [np.array([x for x in group if x is not None]) for group in zip_longest(*time_g_sorted)]

    # initialise
    X = [0] * (len(step_sizes_layered) + 1)
    X[0] = np.array([path[0] for path in y_sorted]) 

    # simulate evolution layer by layer
    for l, steps in enumerate(step_sizes_layered):
        N = len(steps)
        dW = steps.reshape(-1, 1) * rng.normal(size=(N, n_dimensions))
        drift, diffusion = drift_diffusion(X[l][:N])

        if not ARFF:
            diffusion_new = np.zeros((len(step_sizes_layered[l]), n_dimensions, n_dimensions))
            idx = np.arange(n_dimensions)
            diffusion_new[:, idx, idx] = diffusion
            diffusion = diffusion_new
        
        X[l+1] = X[l][:N] + steps.reshape(-1,1) * drift + np.einsum('ijk,ik->ij', diffusion, dW)

    return X, time_g_layered
    

def plot_histogram_SIRG(y, times,  max_time, title, script_dir, filename, n_skip_steps, save=False):

    name = [r'Susceptible $~\tilde{x}_0$', 
        r'Infected $~\tilde{x}_1$', 
        r'Recovered $~\tilde{x}_2$']

    times_all = np.concatenate(times)  
    y_all = np.concatenate(y, axis=0)  
    y_all = np.hstack((y_all, 1 - np.sum(y_all, axis=1, keepdims=True)))
    y_all[:, [1, 2]] = y_all[:, [2, 1]]

    bins_0 = np.linspace(times_all.min() + 0.01, max_time, 100)
    bins_1 = np.linspace(0, 1, 100)

    dim = y_all.shape[1]
    fig, axes = plt.subplots(dim, 1, figsize=(5, 4 * dim), gridspec_kw={'hspace': 0.12})
    
    for d in range(dim):
        hist, x_edges, y_edges = np.histogram2d(times_all, y_all[:, d], bins=[bins_0, bins_1])
        
        axes[d].imshow(
            hist.T,
            origin='lower',
            cmap='inferno',
            aspect='auto',
            extent=[times_all.min(), max_time, 0, 1]
        )
        axes[d].set_ylabel(name[d], fontsize=13)

    axes[0].set_title(f"{title}")
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    plt.show()

    if save:
        output_dir = os.path.join(script_dir, 'saved_results/trajectory_histograms')
        output_path = os.path.join(output_dir, f"{filename}_SS10_{title}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

