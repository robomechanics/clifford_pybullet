import sys
sys.path.append("..")
from simController import simController
import numpy as np
import pybullet as p

physicsClientId = p.connect(p.GUI)
sim = simController(timeStep=1./500.,stepsPerControlLoop=50,physicsClientId=physicsClientId)
sim.terrain.generate(cellHeightScale=0.0,perlinHeightScale=0.0)
sim.resetClifford()
for i in range(20):
	throttleAction = 10
	steerAction = 0.5
	data = sim.controlLoopStep([throttleAction,steerAction])
	#print(data)
p.disconnect(physicsClientId=physicsClientId)