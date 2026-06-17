import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from orbit_functions import * # this imports all the functions that I have defined in the orbit_functions file, cool stuff that I learnt today!!!

v_initial = np.array([0.2, 1.0,1.0]) # initial velocity of particle, enables in the particle to curve since it is perpendicular to B. (0.2 V_(x) is used to give a radial kick )
s_0 = (1.0,3.0,5.0)
x_initial = np.array([1.0,0.0,0.0]) # initial distance of the particle from the wire

dt = 0.01 # time step
nsteps= 50000
particle_trajectory={} # stores the value of s and z for for each individual distance separately




# defining grad B drift:

def gradB_wire(x):  # gradB is basically the gradient of the B field, for a wire, the B field was 1/r, which means that gradB becomes -1/r**2
    X,Y,Z=x
    r=np.sqrt(X**2 + Y**2)
    r=max(r,1e-8)

    return np.array([
        -X/r**3,  # GradB expressed in cartesian coordinates
        -Y/r**3,
        0.0
    ])


for j in s_0:
    x_initial = np.array([j,0.0,0.0])
    B= wire_field(x_initial)
    x= x_initial.copy() # copies the values stored in x_initial so it doesnt destroy the original initial position conditions
    v= v_initial.copy()
    new_positions=[]
    new_velocities=[]


    # initializing empty array for storing Velocity drift due to gradB:
    v_gradB_list=[]
    B_values = [] # initializing an empty array to later store B field values
    for i in range(nsteps):
        new_positions.append(x.copy()) # append adds on to the list as the position changes
        new_velocities.append(v.copy())
        B= wire_field(x)
        Bmag = np.linalg.norm(B)
        b_hat = B/Bmag

        v_parallel=np.dot(v,b_hat)
        v_perp_vec=v-v_parallel*b_hat
        v_perp2=np.dot(v_perp_vec,v_perp_vec)

        gradB= gradB_wire(x)
        v_gradB=(v_perp2/(2*Bmag**3))*np.cross(B,gradB)
        v_gradB_list.append(v_gradB.copy())
        B_values.append(B.copy())
        x,v = boris_push(x,v, B, dt)
    new_positions= np.array(new_positions)
    new_velocities=np.array(new_velocities)
    v_grad_list = np.array(v_gradB_list)



    B_values=np.array(B_values)
    s = np.sqrt(new_positions[:,0]**2 + new_positions[:,1]**2)
    z = new_positions[:,2]
    vz = new_velocities[:,2]
    particle_trajectory[j] = {
        "s":s,
        "z":z,
        "v_gradB":v_grad_list
    }
    

## creating an animation simulation for the plot above:

fig, ax = plt.subplots()
ax.axvline(0, color='black', lw=3, label='wire')
lines={}
for j in s_0:
    line,= ax.plot([],[],lw=1,label=f"s0 ={j}")
    lines[j]=line



# setting up the update frame function

ax.set_xlabel("s (distance from wire)")
ax.set_ylabel("z")
ax.set_ylim(0,100)
ax.set_xlim(0,6)
ax.set_title("Particle oscillations due to different initial particle distance from wire")
ax.legend()
ax.grid(True)

skip = 100
frames = np.array(list(range(0,nsteps,skip)))

def update(frame):
    for j in s_0:
        s=particle_trajectory[j]["s"]
        z = particle_trajectory[j]["z"]
        lines[j].set_data(s[:frame], z[:frame])
    return list(lines.values())


# connecting the update frame to Matplotlib FuncAnimation:
ani = FuncAnimation(
    fig,update,frames=frames, interval= 5, blit=True, repeat=True, repeat_delay=0.02)


ani.save("particle_oscillations.mp4",writer ="ffmpeg",fps=45,dpi=100)
plt.show()



# To confirm the magnetic field afftects the particle trajectory as the distance of particle from the wire is increased.
# Looking down on wire from the top
x = new_positions[:,0] # position along x-axis
y = new_positions[:,1] # position along y-axis

plt.figure()
plt.plot(x,y, lw=0.5)

plt.plot(0,0, 'ko', label='wire') # plotting the wire at origin
plt.plot(x[0],y[0],'go', label ='start')

plt.xlabel("x (position of particle in x-axis)")
plt.ylabel("y (position of particle in y-axis)")
plt.title("top-view of particle motion")
plt.legend()
plt.grid(True)
plt.show()

# Another plot which comparing how the radius changes with time
t = np.arange(len(s)) * dt
plt.figure()
plt.plot(t,s, lw=1)
plt.axhline(x_initial[0], color='k', linestyle='--')
plt.xlabel("time")
plt.ylabel("distance from wire")
plt.title("Radius vs. time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# plotting velocity drift due to gradB in a magnetic wire field:
time=np.arange(nsteps)*dt

plt.figure()

for j in s_0:
    v_grad=particle_trajectory[j]["v_gradB"]
    plt.plot(time,v_grad[:,2],label=f"s0={j}")

plt.xlabel("Time")
plt.xlim(0,100)
plt.ylabel("grad-B drift velocity z")
plt.title("grad-B drift velocity in a wire field")
plt.legend()
plt.grid(True)
plt.show()