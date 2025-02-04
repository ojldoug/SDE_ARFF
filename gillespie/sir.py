
import numpy as np
import scipy
import matplotlib.pyplot as plt

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
