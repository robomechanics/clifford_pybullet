import sys
sys.path.append("..")
import pybullet as p
from simController import simController
import matplotlib.pyplot as plt
import torch
import numpy as np
from cliffordStateTransformation import cliffordStateTransformation
from motionModel import deterministicMotionModel

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# start a sim to generate starting state and random terrain
physicsClientId = p.connect(p.DIRECT)
sim = simController(timeStep=1./500.,stepsPerControlLoop=50,numSolverIterations=300,physicsClientId=physicsClientId)
#sim.terrain.generate(AverageAreaPerCell = 1,cellPerlinScale=5,cellHeightScale=0.75)
sim.terrain.generate(AverageAreaPerCell = 1,cellPerlinScale=5,cellHeightScale=0.6,smoothing=1)
sim.resetClifford()
startState = sim.controlLoopStep([0,0])
p.disconnect(physicsClientId=physicsClientId)

#initialize motion model
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
motionModel.load_state_dict(torch.load('../motionModels/v2sequentialDeterministic.pt'))
motionModel.eval()

# initialize all particles
numParticles = 10
particleStates = cliffordStateTransformation(torch.tensor(startState[3]).unsqueeze(0).to(device),numParticles=numParticles)
predX = [[startState[3][0]] for i in range(numParticles)]
predY = [[startState[3][1]] for i in range(numParticles)]

# loop through time and take action
for t in range(20):
    # get current state of particles
    inStates = particleStates.stateToNNInState()

    # get terrain map based on particle position
    poses = particleStates.posHeading()
    poses = [poses[i,:].detach().cpu().numpy() for i in range(numParticles)]
    maps = [torch.from_numpy(sim.terrain.robotHeightMap(poses[i][0:3],poses[i][3],sim.rMapWidth,sim.rMapHeight,sim.rMapScale)).unsqueeze(0).unsqueeze(0).float().to(device) for i in range(len(poses))]
    inMaps = torch.cat(maps,dim=0)

    # Robot action (throttle,steering) steering between -0.5 and 0.5
    # choose throttle of 10 and different steering
    actionsTaken = torch.cat((torch.ones(numParticles,1)*50,torch.linspace(-0.5,0.5,numParticles).unsqueeze(1)),dim=1).to(device)

    # get prediction of relative robot movement
    prediction = motionModel([inStates,inMaps,actionsTaken])

    # move robot forward based on prediction
    predictedPose = particleStates.moveState(prediction)

    # record position
    for i in range(numParticles):
        predX[i].append(poses[i][0])
        predY[i].append(poses[i][1])
plt.figure()
for i in range(numParticles):
    plt.plot(predX[i],predY[i])
plt.xlim((-10,10))
plt.ylim((-10,10))
plt.show()
