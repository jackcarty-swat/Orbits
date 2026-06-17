import numpy as np 
from numba import njit
@njit
def B_field(x):  # defining function for unifrom magnetic field (in z direction)
    B0=1.0 # Tesla
    return np.array([0,0,B0])


# Lorentz acceleration
# dv/dt = v x B
@njit
def f(state): 
    deriv=np.empty(6) # derivative vector to hold time derivatives of position and velocity
    x=state[:3] # position vector (x,y,z)
    v=state[3:] # velocity vector (vx,vy,vz)
    deriv[:3] =v
    deriv[3:]=np.cross(v,B_field(x))
    return deriv



# Runge-Kutta 4th order method for updating position and velocity after time step dt
@njit
def rk4_step(state,dt):
    temp =np.empty_like(state)
    k1=f(state) # compute initial derivative at current state
    temp[:]=state + 0.5*dt*k1  # using temp[:] results in in-place update of temp array, avoiding creation of new array
    k2=f(temp) # compute derivative at midpoint using k1
    temp[:]=state + 0.5*dt*k2
    k3=f(temp)
    temp[:]=state+dt*k3 
    k4=f(temp)
    new_state=state.copy()
    new_state += (dt/6)*(k1 + 2*k2 + 2*k3 + k4) # combining increments to update state
    return new_state

# using Boris method for updating position and velocity after time step dt
@njit
def boris_push(x,v,B,dt):
    t =0.5*(B*dt) # half of the magnetic rotation during time step
    t_mag2 =np.dot(t,t) # magnitude squared of t vector
    s = (2*t)/(1+t_mag2) # converts t into full corrected rotation
    v_minus=v
    v_prime= v_minus + np.cross(v_minus,t) # rotate velocity
    v_plus=v_minus + np.cross(v_prime,s) # final rotated velocity
    v_new=v_plus
    x_new=x + v_new*dt 
    return x_new, v_new



def wire_field(x):
    X,Y,Z=x
    r2 = X**2 + Y**2
    eps = 1e-6
    r2=max(r2, eps)
    Bx=-Y/ r2
    By = X/ r2
    Bz= 0
    return np.array([Bx,By,Bz])




# Defining dipole magnetic field
def dipole_field(x,r_star):
   mu = r_star**3 # dipole moment in normalized units
   X,Y,Z= x
   r2 = X**2 + Y **2 +Z**2
   if r2<1e-20:
       return np.array([0.0,0.0,0.0])
   r = np.sqrt(r2)
   r5 = r2*r2*r
   factor = mu/r5
   Bx = factor * 3.0 *X*Z
   By=factor * 3.0 * Y*Z
   Bz= factor*(2.0*Z**2-X**2-Y**2)
   return np.array([Bx,By,Bz])
