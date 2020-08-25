#!/usr/bin/env python3

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
    def __init__(self, model, targEnd, mu_0, covar_0, sample, rarParam, a):
        #initialize 
        self.motionModel = model
        self.target = targEnd
        self.N = sample
        Ne = int(N * rarParam)
        self.rho = N - Ne + 1
        self.alpha = a
        t = 0

        self.vX_t = mu_0
        self.vY_t = mu_0
        self.covarX = covar_0
        self.covarY = covar_0
        targX = targEnd[0]
        targY = targEnd[1]
        gamma_t = 0
        optDist = (targX**2 + targY**2)**0.5

        self.numSteps = len(mu_0)
        self.iterate = True

        while self.iterate:
            t += 1

            # generate N samples from the distribution
            pathsX = self.sampleFromDist(self.vX_t, self.covarX)
            pathsY = self.sampleFromDist(self.vY_t, self.covarY)

            # simulate paths
            endPts = self.simulate(pathsX, pathsY)

            # calculate costs
            costs = self.getCost(endPts)

            # update gamma
            gamma_0 = gamma_t
            gamma_t = self.updateGamma(costs)
            if gamma_t-gamma_0 < 0.0005:
                self.iterate = False

            # update distributions
            self.updateDist(gamma_t)



        return None

    def sampleFromDist(self, mean, covar):
        # sample particles from distribution
        paths = np.zeros((self.N, self.numSteps))

        L = np.linalg.cholesky(covar)
        for i in range(self.N):
            r = np.random.uniform(-1, 1, numSteps)
            paths[i] = mean + L@r

        return paths

    def getCost(self):
        # calculate cost of samples
        costs = np.empty(self.N)

        for i in range(self.N):
            error = ((targX - endPts[i][0])**2 + (targY - endPts[i][1])**2)**(0.5)
            dist = endPts[i][2]
            costs[i] = error + dist*0.5
        return costs

    def updateGamma(self, costs):
        # update gamma based on calculated costs
        costsSort = np.sort(costs)
        return costsSort[self.rho]

    def updateDist(self, gamma_t):
        # update distribution based on elite set

        alpha = self.alpha
        numSteps = self.numSteps
        N = self.N
        # update mu
        eliteNumX = 0
        eliteNumY = 0
        eliteSumX = np.zeros(self.numSteps)
        eliteSumY = np.zeros(self.numSteps)
        for k in range(self.N):
            if costs[k] < gamma_t:
                eliteNumX += 1
                eliteSumX += pathsX[k]
                eliteNumY += 1
                eliteSumY += pathsY[k]
                 
        self.vX_t = alpha * vX_t + (1-alpha) * eliteSumX/eliteNumX
        self.vY_t = alpha * vY_t + (1-alpha) * eliteSumY/eliteNumY

        # update sigma
        eliteArrayX = np.zeros((numSteps, numSteps))
        eliteArrayY = np.zeros((numSteps, numSteps))
        for k in range(N):
            if costs[k] < gamma_t:
                vector = pathsX[k] - vX_t
                eliteArrayX += np.outer(vector, np.transpose(vector))
                vector = pathsY[k] - vY_t
                eliteArrayY += np.outer(vector, np.transpose(vector))
                
        self.covarX = alpha * covarX + (1-alpha) * eliteArrayX/(eliteNumX-1)
        self.covarY = alpha * covarY + (1-alpha) * eliteArrayY/(eliteNumY-1)

        return None

    def simulate(self, pathsX, pathsY):
        numSteps = self.numSteps
        N = self.N
        throttle = torch.ones(N, 1)
        steering = torch.ones(N, 1)
        Poses = [torch.zeros(N, 3)]
        dist = 0
        
        for i in range(numSteps):
            for j in range(N):
                throttle[j][0] = np.clip(pathsX[j][i], -1, 1) * 50
                steering[j][0] = np.clip(pathsY[j][i], -0.5, 0.5)

            actionTaken = torch.cat((throttle, steering), dim=1)
            prediction = self.motionModel([actionTaken])

            newX = Poses[-1][:,0] + prediction[:,0]*torch.cos(Poses[-1][:,2]) - prediction[:,1]*torch.sin(Poses[-1][:,2])
            newY = Poses[-1][:,1] + prediction[:,0]*torch.sin(Poses[-1][:,2]) + prediction[:,1]*torch.cos(Poses[-1][:,2])
            newHeading = Poses[-1][:,2] + prediction[:,2]

            # calculate distance travelled
            oldX = Poses[-1][:,0]
            oldY = Poses[-1][:,1]
            dist += ((newX-oldX)*(newX-oldX) + (newY-oldY)*(newY-oldY))**(0.5)

            # append new pose to end of list
            Poses.append(torch.cat((newX.unsqueeze(1),newY.unsqueeze(1),newHeading.unsqueeze(1)),dim=1))


        endPt = torch.zeros(N, 2)
        plt.figure()
        for i in range(N):
            pathX = []
            pathY = []
            for j in range(len(Poses)):
                pathX.append(Poses[j][i,0].item())
                pathY.append(Poses[j][i,1].item())
            plt.plot(pathX,pathY)
            endPt[i][0] = pathX[-1]
            endPt[i][1] = pathY[-1]
        plt.plot(self.target[0], self.target[1], 'r+')
        plt.xlim((-10,10))
        plt.ylim((-10,10))
        plt.show()

        return torch.cat((endPt, dist.unsqueeze(1)), dim=1)



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




