import Initializers
import Fields
import Visualizers
import Workers
import numpy as np
import matplotlib.pyplot as plt

initial = Initializers.InitialFromBody(1000, [[180,200],[180,200],[-10,10]], 1, 1, 0, 100)
print("Initialized")
output, steps = Workers.BetterBorisIterator(2*3.14*1000, 0.1, initial, Fields.AxisymmetricMagneticField, 6.5, 10)
print("Computed")
Visualizers.VisualizeTrajectories(steps, 10)


lost = output[np.abs(output['z']) > 9787]
kept = output[np.abs(output['z']) <= 9787]

def vstats(x, y, z):
    magPerp = np.sqrt(x**2 + y**2)
    return z / magPerp

lratios = []
for i in lost.iterrows():
    lratios.append(np.abs(vstats(*i[1][:3])))
    print(vstats(*i[1][:3]))

kratios = []
for i in kept.iterrows():
    kratios.append(np.abs(vstats(*i[1][:3])))

tratios = []
for i in output.iterrows():
    tratios.append(np.abs(vstats(*i[1][:3])))

print("Velocity ratios:")
print(np.average(lratios))
print(np.average(kratios))
print(np.average(tratios))
print("Ratio lost:")
print(len(lost)/len(output))

def EnergyChecker(initial, final):
    EInit = initial['vX']**2 + initial['vY']**2 + initial['vZ']**2
    Efin = final['vX']**2 + final['vY']**2 + final['vZ']**2
    
    print(EInit.mean, Efin.mean)
    
#EnergyChecker(initial, output)

plt.plot(steps[:,0,7], steps[:,0,3])
plt.show()
plt.plot(steps[:,0,7], steps[:,0,4])
plt.show()
plt.plot(steps[:,0,7], steps[:,0,5])
plt.show()