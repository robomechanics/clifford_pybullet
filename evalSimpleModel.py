import pybullet as p
from simController import simController
from motionModel import simpleMotionModel
import matplotlib.pyplot as plt
import torch
import numpy as np

physicsClientId = p.connect(p.GUI)
sim = simController(physicsClientId=physicsClientId)
sim.terrain.generate()
#sim.terrain.generate(cellHeightScale=0,perlinHeightScale=0)
sim.resetClifford()
startState = sim.controlLoopStep([0,0])
predictedPose = [startState[3][0][0],startState[3][0][1],startState[4]]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
motionModel = simpleMotionModel().to(device)
#motionModel.load_state_dict(torch.load('motionModels/simple.pt'))
motionModel.eval()

actualX = [startState[3][0][0]]
actualY = [startState[3][0][1]]
predX = [startState[3][0][0]]
predY = [startState[3][0][1]]
for i in range(50):
	print(i)
	data = sim.controlLoopStep(sim.randomDriveAction())
	if data[2]:
		break
	inAction = torch.FloatTensor(data[0][2]).unsqueeze(0).to(device)
	prediction = motionModel([inAction])[0]
	print(prediction)
	newX = predictedPose[0] + prediction[0].item()*np.cos(predictedPose[2]) - prediction[1].item()*np.sin(predictedPose[2])
	newY = predictedPose[1] + prediction[0].item()*np.sin(predictedPose[2]) + prediction[1].item()*np.cos(predictedPose[2])
	print(newX)
	newHeading = predictedPose[2] + prediction[2].item()
	predictedPose = [newX,newY,newHeading]
	actualX.append(data[3][0][0])
	actualY.append(data[3][0][1])
	predX.append(predictedPose[0])
	predY.append(predictedPose[1])
p.disconnect(physicsClientId=physicsClientId)
plt.figure()
for i in range(len(actualX)):
	plt.clf()
	plt.plot(actualX[0:i],actualY[0:i])
	plt.plot(predX[0:i],predY[0:i])
	plt.xlim((-10,10))
	plt.ylim((-10,10))
	plt.pause(0.1)
plt.show()