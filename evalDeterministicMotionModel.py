import pybullet as p
from simController import simController
from motionModel import deterministicMotionModel
import matplotlib.pyplot as plt
import torch
import numpy as np
#from standardizeData import standardizeData
from cliffordStateTransformation import cliffordStateTransformation

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
physicsClientId = p.connect(p.GUI)
sim = simController(physicsClientId=physicsClientId)
startState = sim.controlLoopStep([0,0])
cliffordStatePred = cliffordStateTransformation(torch.tensor(startState[3]).unsqueeze(0).to(device))

inStateDim = len(startState[0][0])+1
inActionDim = len(startState[0][2])
inMapDim = startState[0][1].shape[1]
outStateDim = len(startState[1][0])

argDim = [inStateDim,inMapDim,inActionDim,outStateDim]
convSizes = [[32,5],[32,4],[32,3]]
fcSizes = [1024,512,256]#,128]
networkSizes = [convSizes,fcSizes]
dropout_ps = [0,0,0]
motionModelArgs = [argDim,networkSizes,dropout_ps]
motionModel = deterministicMotionModel(motionModelArgs).to(device)
#motionModel.load_state_dict(torch.load('motionModels/deterministic.pt'))
motionModel.load_state_dict(torch.load('motionModels/sequentialDeterministic.pt'))
motionModel.eval()
#outputScale = standardizeData(device=device)
#outputScale.getDistribution(torch.load("simData/simOutputData0.pt"))

actualX = [startState[3][0]]
actualY = [startState[3][1]]
predX = [startState[3][0]]
predY = [startState[3][1]]
for i in range(50):
    print(i)
    data = sim.controlLoopStep(sim.randomDriveAction())
    if data[2]:
        break
    inState = cliffordStatePred.stateToNNInState()
    inAction = torch.FloatTensor(data[0][2]).unsqueeze(0).to(device)
    inMap = torch.from_numpy(data[0][1]).unsqueeze(0).float().to(device)
    prediction = motionModel([inState,inMap,inAction])
    #prediction = outputScale.raw(prediction)
    #prediction = torch.tensor(data[1]).to(prediction.device)
    predictedPose = cliffordStatePred.moveState(prediction)
    actualX.append(data[3][0])
    actualY.append(data[3][1])
    predX.append(predictedPose[0,0])
    predY.append(predictedPose[0,1])
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