import orbit_functions
import numpy as np
import matplotlib.pyplot as plt
import time # for measuring execution time of each method
import pandas as pd # For creating tables
x0=np.array([1.0,0.0,0.0]) # initial position 
v0=np.array([0.0,1.0,0.0])# initial velocity (perpendicular to magnetic field)



# simulation parameters
dt=0.01 # time step (small enough to capture motion accurately
omega=1 # cyclotron frequency for proton in 1 Tesla field 
tau =(2*np.pi)/omega # period of one orbit 
nsteps=int(round(tau/dt)) # number of time steps to simulate 
trajectory_rk4=np.zeros((nsteps,3))
trajectory_boris =np.zeros((nsteps,3))
 # initialize list to store trajectory points

# main loop to update position and velocity using RK4 method
state_rk4 =np.empty(6) # we have 6 components to track: 3 for position and 3 for velocity
state_rk4[:3]=x0
state_rk4[3:]=v0
for i in range (nsteps):
    trajectory_rk4[i]=state_rk4[:3]  
    state_rk4=orbit_functions.rk4_step(state_rk4,dt) # Calling on RK4 function from orbit_functions.py


state_boris=np.empty(6)
state_boris[:3]=x0
state_boris[3:]=v0
B=orbit_functions.B_field(x0)
for j in range(nsteps):
    x_curr= state_boris[:3]
    v_curr =state_boris[3:]
    trajectory_boris[j]=x_curr.copy()
    x_new, v_new=orbit_functions.boris_push(x_curr,v_curr,B,dt)
    state_boris[:3]=x_new
    state_boris[3:]=v_new



# plotting trajectories using each method

# rk4 trajectory
plt.figure(figsize=(12,6))
plt.subplot(1,2,1) # significance of subplot: 1 row, 2 columns, 1st plot(for visualization)
plt.plot(trajectory_rk4[:,0], trajectory_rk4[:,1])
plt.xlabel("position x")
plt.ylabel("position y")
plt.title("One Orbit of Proton using RK4 Method")
plt.axis('equal')

# boris trajectory
plt.subplot(1,2,2) # 1 row, 2 columns, 2nd plot
plt.plot(trajectory_boris[:,0], trajectory_boris[:,1])
plt.xlabel("position in x")
plt.ylabel("position in y")
plt.title("One Orbit of Proton using Boris Method")
plt.axis('equal')
plt.savefig("one_orbit.png", dpi=300)
plt.close()
print("Figure for one orbit successfully saved")


"""
The above code simulates the motion of a proton in a uniform magnetic field using RK4 and Boris methods.
Respective plots show the trajectory of a proton for one compete orbit,
which should be circular due to the nature of the magenetic field and initial conditions.
"""


#Repeating simulations for 1000 orbits to test long-term stability of both methods 
n_orbits=1000 
total_time =n_orbits*tau # total simulation time for 1000 orbits 
n_steps=int(total_time/dt) 
trajectory_rk4=np.zeros((n_steps,3)) 
velocity_rk4=np.zeros((n_steps,3))

# timing and simulation of 1000 orbits using 1000 orbits
start_rk4=time.perf_counter() 
for i in range(n_steps): 
    trajectory_rk4[i]=state_rk4[:3] 
    velocity_rk4[i]=state_rk4[3:]
    state_rk4=orbit_functions.rk4_step(state_rk4,dt) 
end_rk4=time.perf_counter() 
rk4_time=end_rk4-start_rk4 # total execution time for RK4 method


# Boris simulation for 1000 orbits
trajectory_boris=np.zeros((n_steps,3))
velocity_boris=np.zeros((n_steps,3))
B=orbit_functions.B_field(x0)

# timing for Boris method
start_boris=time.perf_counter() # accurately measures time interval
for j in range(n_steps):
        trajectory_boris[j]=state_boris[:3]
        velocity_boris[j]=state_boris[3:]
        x_new, v_new=orbit_functions.boris_push(
        state_boris[:3], 
        state_boris[3:],
        B,
        dt
    )
        state_boris[:3]=x_new
        state_boris[3:]=v_new
end_boris=time.perf_counter()
boris_time = end_boris - start_boris # total execution time for Boris method




# plotting orbit evolutions

fig, ax = plt.subplots(1,2,figsize=(14,6))

# RK4 plot
ax[0].plot(
    trajectory_rk4[:,0],
    trajectory_rk4[:,1],
    linewidth=0.8,
    label="RK4"
)

ax[0].set_title("RK4 Method")
ax[0].set_xlabel("x position")
ax[0].set_ylabel("y position")
ax[0].axis("equal")
ax[0].grid(True)


# Boris
ax[1].plot(
    trajectory_boris[:,0],
    trajectory_boris[:,1],
    linewidth=0.8
)

ax[1].set_title("Boris Method")
ax[1].set_xlabel("x position")
ax[1].set_ylabel("y position")
ax[1].axis("equal")
ax[1].grid(True) # turns on the background grid lines


# fixing spacing between the two plots
fig.tight_layout()

plt.savefig(
    "orbit_comparison_1000_orbits.png",
    dpi=300
)
plt.close()
print("1000-orbits figure saved successfully")


"""
So far, we have created plots of particle trajectories using both methods repeated over 1000 cyclotron orbits
The figure gives us a nice circular orbit for both methods, suggesting that both methods are excellent at conserving orbits regardless
of the number of orbits traced or run-time.
"""

# computing the gryo-radius for each method for 1000 orbits

# Tracking gyro-radius
xc_rk4=np.mean(trajectory_rk4[:,0])
yc_rk4=np.mean(trajectory_rk4[:,1])
r_rk4=np.sqrt(
     (trajectory_rk4[:,0]-xc_rk4)**2 +
     (trajectory_rk4[:,1]-yc_rk4)**2
)
xc_boris = np.mean(trajectory_boris[:,0])
yc_boris=np.mean(trajectory_boris[:,1])
r_boris=np.sqrt(
     (trajectory_boris[:,0]-xc_boris)**2 +
     (trajectory_boris[:,1]-yc_boris)**2
)

# creating time array
time_array = np.arange(n_steps)*dt
orbit_number = time_array/tau


"""
Checking if I am actually calculating the actual radius,
turns out I have been calculating the wrong radius all this time. 
Also am going to be veriying the guiding center right below 
"""
steps_per_orbit=int(tau/dt)
orbit_sample = orbit_number[:: steps_per_orbit]
radius_error_rk4=np.abs(r_rk4[:: steps_per_orbit]-1.0) #[:: steps_per_orbit]
radius_error_boris=np.abs(r_boris[:: steps_per_orbit]-1.0) # [::steps_per_orbit]
#r_rk4_sampled=r_rk4[::steps_per_orbit]
#r_boris_sampled=r_boris[:: steps_per_orbit]
#print(r_rk4_sampled[:10])
#print(r_boris_sampled[:10])
#print(r_rk4_sampled.min(),r_rk4_sampled.max())
#print(r_boris_sampled.min(), r_boris_sampled.max())


# plotting R-1 Vs. orbit number, (R-1) shows how much the numerical method drifts from the correct orbit


# plotting R-1 vs. orbit number
plt.figure(figsize=(8,6))
plt.plot(
     orbit_sample,
     radius_error_rk4,
     label="RK4"
)

plt.plot(
     orbit_sample,
     radius_error_boris,
     label="Boris"
)

plt.xlabel("Number of Orbits")
plt.ylabel("R-1")
plt.title("Gyro-radius Error")
plt.yscale("log") # using a logarithmic y-axis which helps us see small errors
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.savefig(
     "Radius_error_vs_orbits.png",
     dpi=300
)
plt.close()

# Computing energy
energy_rk4=0.5*np.sum(velocity_rk4**2,axis=1)
energy_boris=0.5*np.sum(velocity_boris**2,axis=1)
E0_rk4 =energy_rk4[0]
E0_boris=energy_boris[0]

energy_error_rk4=(energy_rk4-E0_rk4)/E0_rk4
energy_error_boris =(energy_boris-E0_boris)/E0_boris


# plotting energy conservation error
plt.figure(figsize=(8,5))
plt.plot(orbit_number, energy_error_rk4, label="RK4")
plt.plot(orbit_number, energy_error_boris, label="Boris")
plt.xlabel("Orbit Number")
plt.ylabel(r"$\Delta E/E0$")
plt.title("Relative Energy Error vs. Orbit Number")
plt.yscale("symlog",linthresh=1e-15)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(
     "Relative_energy_error.png",
     dpi=300,
     bbox_inches="tight"
)
plt.close()
print("Relative energy error figure saved successfully")
# plotting kinetic energy Vs. time
plt.figure(figsize=(8,5))

plt.plot(time_array, energy_rk4, label="RK4")
plt.plot(time_array, energy_boris, label="Boris")

plt.xlabel("Time")
plt.ylabel("Kinetic Energy")
plt.title("Energy Conservation")
plt.grid(True)
plt.legend()

plt.savefig("energy_vs_time.png", dpi=300)
plt.close()




results_table=pd.DataFrame({
     "Metric":[
          "Max radius error",
          "Mean radius error",
          "Final radius error",
          "Final energy error",
          "Execution time (s)"
     ],
     "RK4 Method": [
          f"{np.max(radius_error_rk4):.3e}",
          f"{np.mean(radius_error_rk4):.3e}",
          f"{radius_error_rk4[-1]:.3e}",
          f"{energy_error_rk4[-1]:.3e}",
          f"{rk4_time:.4f}"
     ],
     "Boris Method":[
          f"{np.max(radius_error_boris):.3e}",
          f"{np.mean(radius_error_boris):.3e}",
          f"{radius_error_boris[-1]:.3e}",
          f"{energy_error_boris[-1]:.3e}",
          f"{boris_time:.4f}"
     ]
})

fig, ax=plt.subplots(figsize=(8,3))
ax.axis('off')

table=ax.table(
     cellText=results_table.values,
     colLabels=results_table.columns,
     loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2,1.5)

plt.savefig(
     "simulation_results_table.png",
     dpi=300,
     bbox_inches ="tight"
)
plt.close()