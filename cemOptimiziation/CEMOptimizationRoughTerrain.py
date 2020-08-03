import sys
sys.path.append("..")
import pybullet as p
from simController import simController
import matplotlib.pyplot as plt
import torch
import numpy as np
from cliffordStateTransformation import cliffordStateTransformation
from learnLSTM import lstmMotionModel
from terrain import terrain

class CEMOptimization:
    def __init__(self):
        #initialize 
        return None
    def sampleFromDist(self):
        # sample particles from distribution
        return None
    def getCost(self):
        # calculate cost of samples
        return None
    def updateDist(self):
        # update distribution based on elite set
        return True # if stop criteria met
        return False # if stop criteria not met

if __name__=="__main__":
    #initialize clifford start state and terrain
    generateNewTerrainAndStartState = False
    if generateNewTerrainAndStartState:
        physicsClientId = p.connect(p.GUI)
        sim = simController(timeStep=1./500.,stepsPerControlLoop=50,numSolverIterations=300,physicsClientId=physicsClientId)
        sim.terrain.generate(AverageAreaPerCell = 1,cellPerlinScale=5,cellHeightScale=0.6,smoothing=1)
        sim.resetClifford()
        startState = sim.controlLoopStep([0,0])
        startState = torch.tensor(startState[3][0:-14])
        terrainParam = np.array([sim.terrain.mapWidth,sim.terrain.mapHeight,sim.terrain.meshScale[0],sim.terrain.meshScale[1],sim.rMapWidth,sim.rMapScale])
        np.save('terrainParam.npy',terrainParam)
        np.save('terrainHeight.npy',sim.terrain.gridZ)
        torch.save(startState,'startState.pt')
    #load start state and terrain
    terrainParam = np.load('terrainParam.npy')
    ter = terrain(int(terrainParam[0]),int(terrainParam[1]),terrainParam[2],terrainParam[3])
    ter.loadTerrain('terrainHeight.npy')
    inMapDim = int(terrainParam[4])
    inMapScale = terrainParam[5]
    startState = torch.load('startState.pt')

    # set up motion model
    print('initializing motion model')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    inStateDim = 9
    inActionDim = 2
    outStateDim = 13
    argDim = [inStateDim,inMapDim,inActionDim,outStateDim]
    convSizes = [[32,5],[32,4],[32,3]]
    lstmSize = [3,1024]#,128]
    networkSizes = [convSizes,lstmSize]
    motionModelArgs = [argDim,networkSizes]
    motionModelLSTM = lstmMotionModel(motionModelArgs).to(device)
    print('loading motion model')
    #if device == 'cpu':
    motionModelLSTM.load_state_dict(torch.load('lstmMotionModel.pt'))
    #else:
    #    motionModelLSTM.load_state_dict(torch.load('../motionModels/lstmMotionModelV2.pt'))
    #motionModelLSTM.eval()
    print('finished loading')

    # action sequence should be tensor of shape (number particles x action size(2) x sequenceLength)
    nParticles = 100
    sequenceLength = 50

    # as an example sequence, give same throttle, but different steering for all particles (for whole sequence)
    throttleCommands = torch.ones(nParticles,1,sequenceLength) * 30
    steeringCommands = torch.linspace(-1,1,nParticles).unsqueeze(1).unsqueeze(1).repeat_interleave(sequenceLength,dim=2)
    actionSequence = torch.cat((throttleCommands,steeringCommands),dim=1)
    # the (i,0,j) element of action sequence is the drive command of the ith particle at time j in the sequence
    # the (i,1,j) element of action sequence is the steer command of the ith particle at time j in the sequence

    # set up initial state of clifford for all particles
    prevLSTMStates = None
    cliffordStateLSTM = cliffordStateTransformation(startState.unsqueeze(0).to(device),nParticles)
    trajectories = torch.zeros(nParticles,2,0).to(device)

    # loop through action sequences and make predictions
    with torch.no_grad():
        for t in range(sequenceLength):
            print(t)
            inStates = cliffordStateLSTM.stateToNNInState()
            inActions  = actionSequence[:,:,t].to(device)
            poses = cliffordStateLSTM.posHeading().detach().cpu().numpy()
            maps = [torch.from_numpy(ter.robotHeightMap(poses[i,0:3],poses[i,3],inMapDim,inMapDim,inMapScale)).unsqueeze(0).unsqueeze(0).float() for i in range(nParticles)]
            inMaps = torch.cat(maps,dim=0).to(device)
            prediction,prevLSTMStates = motionModelLSTM([inStates,inMaps,inActions],1,nParticles,prevLSTMStates)
            predictedPoses = cliffordStateLSTM.moveState(prediction)
            trajectories = torch.cat((trajectories,predictedPoses[:,0:2].unsqueeze(2)),dim=2)

    # plot trajectories
    # trajectories[i,0,t] gives xth position of particle i at time t
    # trajectories[i,1,t] gives yth position of particle i at time t
    plt.figure()
    for i in range(nParticles):
        plt.plot(trajectories[i,0,:].cpu(),trajectories[i,1,:].cpu())
    plt.show()




