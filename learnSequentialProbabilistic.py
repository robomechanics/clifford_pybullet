import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from motionModel import probabilisticMotionModel
from replayBuffer import sequentialReplayBuffer
from cliffordStateTransformation import cliffordStateTransformation
from torch.utils.tensorboard import SummaryWriter

class learnSequentialProbabilistic(object):
    def __init__(self,learningArgs,motionModelArgs):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # set up training
        lr = learningArgs[0]
        lrDecay_stepSize = learningArgs[1]
        lrDecay_gamma = learningArgs[2]
        weight_decay = learningArgs[3]
        self.numParticles = learningArgs[4]
        self.MotionModel = probabilisticMotionModel(motionModelArgs).to(self.device)
        self.optimizer = Adam(self.MotionModel.parameters(),lr = lr,weight_decay=weight_decay)
        self.lrScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lrDecay_stepSize, gamma=lrDecay_gamma) # for lowering step size as training progresses
        self.criterion = torch.nn.MSELoss()
    def updateMotionModel(self,dataBatch):
        # initialize update
        self.MotionModel.train()
        self.optimizer.zero_grad()
        loss = torch.zeros(1,requires_grad=True).to(self.device)
        lossCount = 0
        # start at ground truth state. Initialize all particles
        cliffordPredState = cliffordStateTransformation(dataBatch[0][0][3],numParticles = self.numParticles)
        # loop through sequence predicting robot pose with motion model
        for stepNum in range(0,len(dataBatch)-1):
            # use robot state from prediction and map/action from data
            mmInput = [cliffordPredState.stateToNNInState()]+[dataBatch[stepNum][0][i].repeat_interleave(self.numParticles,dim=0) for i in range(1,len(dataBatch[stepNum][0]))]
            predDists = self.MotionModel(mmInput)
            relativeGroundTruth = cliffordPredState.getRelativeState(dataBatch[stepNum+1][0][3].repeat_interleave(self.numParticles,dim=0))
            loss = loss - self.MotionModel.logMeanLikelihoods(relativeGroundTruth,predDists,self.numParticles).mean()
            lossCount +=1
            predSamples = self.MotionModel.sampleFromDist(predDists)
            cliffordPredState.moveState(predSamples)
        loss = loss/lossCount
        loss.backward()
        self.optimizer.step()
        return loss.item()

if __name__ == '__main__':
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()

    # load replay buffer
    cpuReplayBuffer = sequentialReplayBuffer(loadDataPrefix='simData/v2')
    cpuReplayBuffer.loadData()
    data = cpuReplayBuffer.getRandSequence()
    inStateDim = data[0][0].shape[1]+1
    inMapDim = data[0][1].shape[2]
    inActionDim = data[0][2].shape[1]
    outStateDim = data[1][0].shape[1]
    outStateDim = 27 # (7 relative position) + (6 twist) + (14 joint state)

    # training/ neural network parameters
    learningRate = 0.000025
    lrDecay_stepSize = 2000
    lrDecay_gamma = 1#0.9
    weight_decay=0
    finalNumParticles = 64
    numParticles = 1
    learningArgs = [learningRate,lrDecay_stepSize,lrDecay_gamma,weight_decay,numParticles]
    argDim = [inStateDim,inMapDim,inActionDim,outStateDim]
    convSizes = [[32,5],[32,4],[32,3]]
    fcSizes = [1024,512,256]#,128]
    networkSizes = [convSizes,fcSizes]
    dropout_ps = [0,0,0]
    motionModelArgs = [argDim,networkSizes,dropout_ps]
    Learn = learnSequentialProbabilistic(learningArgs,motionModelArgs)
    finalTrainBatchSize = 16
    trainBatchSize = 64
    sequenceLength = 2
    maxSequenceLength = 50
    trainingSet = [0,0.8]
    testBatchSize = 256
    testSet = [trainingSet[1],1.0]
    smoothing = 0.9
    smoothedLoss = 1
    switchValue = -50
    finalSwitchValue = -50

    numUpdates = 500000000
    for updateCount in range(numUpdates):
        if smoothedLoss < switchValue and sequenceLength<maxSequenceLength:
            sequenceLength+=1
            numParticles = finalNumParticles
            switchValue = finalSwitchValue
            smoothedLoss = switchValue+np.abs(switchValue*0.1)
            trainBatchSize = finalTrainBatchSize
        dataBatch = cpuReplayBuffer.getRandSequenceFixedLength(trainBatchSize,sequenceLength,device=device,percentageRange=trainingSet)
        trainLoss = Learn.updateMotionModel(dataBatch)
        smoothedLoss = smoothing*smoothedLoss + (1-smoothing)*trainLoss
        if updateCount%10==0:
            writer.add_scalar('train/mse_loss',trainLoss,updateCount)
            writer.add_scalar('train/log_loss',smoothedLoss,updateCount)
            print("updateCount: " + str(updateCount) + " smoothedLoss: " +str(smoothedLoss) + " trainLoss: " + str(trainLoss) + " sequenceLength: " + str(sequenceLength) + " lr: " + str(Learn.optimizer.state_dict()['param_groups'][-1]['lr']))
        if updateCount%1000==0:
            torch.save(Learn.MotionModel.state_dict(), 'motionModels/sequentialProbailistic.pt')











