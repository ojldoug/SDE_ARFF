
import cython

import numpy as np
cimport numpy as np

from libc.math cimport ceil, log
from libc.stdlib cimport srand, rand, RAND_MAX

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t


@cython.cdivision(True)
cdef double clip(double a, double min_value, double max_value):
    return min(max(a, min_value), max_value)

@cython.cdivision(True)
cdef double rand_double():
    return rand() / <double>RAND_MAX

def simulate( y0, k1, k2, k3, N, random_state, time_max=4, time_step=0.01):
    if len(y0.shape)==1 or y0.shape[0]==1:
        y0 = y0.reshape(1,-1)

    NP = int(np.ceil(time_max/time_step))
    _R = np.zeros((3,))
    _t = np.zeros((NP,))
    _y = np.zeros((NP,y0.shape[1]))
    time = []
    y = []
    
    simulate_c(time, y, _t, _y, _R, y0, k1, k2, k3, N, random_state, time_max, time_step)
    return time, y
    
@cython.boundscheck(False)
cdef void simulate_c(list time, list y, double[:] _t, double[:,:] _y, double[:] _R, double[:,:] y0, int k1, int k2, int k3, int N, int random_state, double time_max, double time_step):
    cdef int k = 0
    cdef int i = 0
    cdef double[:] y0_current
    
    for k in range(y0.shape[0]):
        y0_current = y0[k,:]
        i = simulate_single(_y, _t, _R, y0_current, k1, k2, k3, N,
                               random_state+k, time_max, time_step)
        time.append(np.asarray(_t[:i]).copy())
        y.append(np.asarray(_y[:i,:]).copy())
   
@cython.cdivision(True)
@cython.boundscheck(False)
cdef int simulate_single(double[:, :] y, double[:] time, double[:] R, double[:] y0, int k1, int k2, int k3, int N, int random_state, double time_max=4, double time_step=0.01):
        """
        y0[0]     initial condition for y1 (I, infected species)
        y0[1]     initial condition for y2 (R, recovered species)
        time_max  max time
        tstep     output dt
        """
        
        srand(random_state)
        
        # initialize internal parameters
        k1 = 4*k1
        
        cdef double CN = 1.0 / N
        cdef double curtime = 0.0
        cdef int NP = int(ceil(time_max/time_step))
        cdef int N1 = int(y0[0]*N)
        cdef int N2 = int(y0[1]*N)
        cdef int i = 0
        cdef double RA = 0.0
        cdef double RSum
        cdef double y1
        cdef double y2
        cdef int Act
        
        while (curtime <= time_max):
    
            # calculate all rates and their sum
            y1 = clip(N1 * CN, 0, 1);    # I concentration
            y2 = clip(N2 * CN, 0, 1);    # R concentration
            R[0] = k1*y1*(1-y1-y2);  # I + S --> I + I 
            R[1] = k2*y1;            # I --> R
            R[2] = k3*y2;            # R --> S
            RSum = R[0]+R[1]+R[2]
            
            if RSum == 0: # happens if y1 is zero
                break
            if i >= NP:
                break

            # call RNG (0,1)
            x = rand_double()*RSum;
            # x = 0.1 *RSum;

            # select one elementary event
            RA = R[0]
            Act = 0 # python is zero based...
            while (RA < x and Act < 2):
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
            dt = -log(clip(rand_double(), 1e-10, 1))/(RSum*N)
            # dt = 1.0 /(RSum*N);
            curtime = curtime + dt

            # save solution
            if (curtime >= time_step*i):
                time[i] = curtime
                y[i, 0] = y1
                y[i, 1] = y2
                i = i + 1
        
        return i