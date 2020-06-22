import pybullet as p
from simController import simController
from motionModel import probabilisticMotionModel
import matplotlib.pyplot as plt
import torch

physicsClientId = p.connect(p.GUI)
sim = simController(physicsClientId=physicsClientId)
startState = sim.controlLoopStep([0,0])
predictedPose = startState[3][0:2]
predictedTwist = startState[1][0][7:13]
predictedJointState = startState[1][0][13:27]
inStateDim = len(startState[0][0])
inActionDim = len(startState[0][2])
inMapDim = startState[0][1].shape[1]
outStateDim = len(startState[1][0])

argDim = [inStateDim,inMapDim,inActionDim,outStateDim]
convSizes = [[32,5],[32,4],[32,3]]
fcSizes = [1024,512,256]#,128]
networkSizes = [convSizes,fcSizes]
dropout_ps = [0,0,0]
motionModelArgs = [argDim,networkSizes,dropout_ps]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
motionModel = probabilisticMotionModel(motionModelArgs).to(device)
motionModel.load_state_dict(torch.load('motionModels/probabilistic.pt'))
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
	inState = torch.FloatTensor(data[0][0]).unsqueeze(0).to(device)
	inAction = torch.FloatTensor(data[0][2]).unsqueeze(0).to(device)
	inMap = torch.from_numpy(data[0][1]).unsqueeze(0).float().to(device)
	prediction = motionModel([inState,inMap,inAction])[0]
	relativePos = prediction[0,0:3]
	relativeOrien = prediction[0,3:7]
	predictedTwist = prediction[0,7:13]
	predictedJointState = prediction[0,13:27]
	predictedPose = p.multiplyTransforms(predictedPose[0],predictedPose[1],relativePos,relativeOrien)
	actualX.append(data[3][0][0])
	actualY.append(data[3][0][1])
	predX.append(predictedPose[0][0])
	predY.append(predictedPose[0][1])
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