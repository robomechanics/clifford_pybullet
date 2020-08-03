import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from replayBuffer import sequentialReplayBuffer
from cliffordStateTransformation import cliffordStateTransformation
from torch.utils.tensorboard import SummaryWriter

class lstmMotionModel(nn.Module):
    def __init__(self, motionModelArgs):
        super(lstmMotionModel, self).__init__()
        # set size of neural network input
        argDim = motionModelArgs[0]
        self.inStateDim = argDim[0]
        self.inMapDim = argDim[1]
        self.inActionDim = argDim[2]
        self.outStateDim = argDim[3]
        # set sizes of neural networks
        nnSize = motionModelArgs[1]
        convSize = nnSize[0]
        lstmSize = nnSize[1]
        #set up conv networks
        self.convs = nn.ModuleList([])
        lastDim = [1,self.inMapDim]
        for i in range(len(convSize)):
            self.convs.append(nn.Conv2d(lastDim[0],convSize[i][0],convSize[i][1]))
            lastDim = [convSize[i][0],lastDim[1]-convSize[i][1]+1]
        self.convOutputDim = lastDim[0]*lastDim[1]*lastDim[1]
        #set up LSTM networks
        lastDim = self.convOutputDim+self.inStateDim+self.inActionDim # size of input to first LSTM
        self.lstm = nn.LSTM(input_size=lastDim,num_layers=lstmSize[0],hidden_size=lstmSize[1],batch_first = True)
        #self.LSTMs = nn.ModuleList([])
        #for i in range(len(lstmSize)):
        #    self.LSTMs.append(nn.LSTM(inputSize = lastDim, hidden_size = ,))
        #    lastDim = lstmSize[i]
        self.fcOutput = nn.Linear(lstmSize[-1],self.outStateDim)
    def forward(self,data,sequenceLength,batchSize,prevLSTMStates=None):
        rState = data[0]
        rMap = data[1]
        rAction = data[2]
        for i in range(len(self.convs)):
            rMap = self.convs[i](rMap)
            rMap = F.leaky_relu(rMap)
        rMap = rMap.view(-1,self.convOutputDim)
        connected = torch.cat((rMap,rState,rAction),axis=1)
        connected = connected.view(batchSize,sequenceLength,-1)
        if prevLSTMStates == None:
            lstmOutput,lstmStates = self.lstm(connected)
        else:
            lstmOutput,lstmStates = self.lstm(connected,prevLSTMStates)
        lstmOutput = lstmOutput.reshape(batchSize*sequenceLength,-1)
        output = self.fcOutput(lstmOutput)
        return (output,lstmStates)

class learnLSTM(object):
    def __init__(self,learningArgs,motionModelArgs):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # set up training
        lr = learningArgs[0]
        lrDecay_stepSize = learningArgs[1]
        lrDecay_gamma = learningArgs[2]
        weight_decay = learningArgs[3]
        self.MotionModel = lstmMotionModel(motionModelArgs).to(self.device)
        self.optimizer = Adam(self.MotionModel.parameters(),lr = lr,weight_decay=weight_decay)
        self.lrScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lrDecay_stepSize, gamma=lrDecay_gamma) # for lowering step size as training progresses
        self.criterion = torch.nn.MSELoss()
    def updateMotionModel(self,dataBatch,sequenceLength,batchSize):
        # initialize update
        self.MotionModel.train()
        self.optimizer.zero_grad()
        # currently not running multi-step
        cliffordPredState = cliffordStateTransformation(dataBatch[0][3])
        mmInput = [cliffordPredState.stateToNNInState()]+[dataBatch[0][i] for i in range(1,len(dataBatch[0]))]
        prediction = self.MotionModel(mmInput,sequenceLength,batchSize)[0]
        #relativeGroundTruth = cliffordPredState.getRelativeState(dataBatch)
        relativeGroundTruth = dataBatch[1][0]
        loss = self.criterion(prediction,relativeGroundTruth)
        loss.backward()
        self.optimizer.step()
        self.lrScheduler.step()
        return loss.item()
        """
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
        return loss.item()"""

if __name__ == '__main__':
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()

    # load replay buffer
    cpuReplayBuffer = sequentialReplayBuffer(loadDataPrefix='simData/v2')
    cpuReplayBuffer.loadData()
    # remove all joint states
    cpuReplayBuffer.inputData[0] = cpuReplayBuffer.inputData[0][:,0:-14]
    cpuReplayBuffer.inputData[3] = cpuReplayBuffer.inputData[3][:,0:-14]
    cpuReplayBuffer.outputData[0] = cpuReplayBuffer.outputData[0][:,0:-14]
    inStateDim = 9 # (3 gravity vector) + (6 twist)
    inMapDim = cpuReplayBuffer.inputData[1].shape[2]
    inActionDim = cpuReplayBuffer.inputData[2].shape[1]
    outStateDim = 13 # (7 relative position) + (6 twist)

    # training/ neural network parameters
    learningRate = 0.001
    lrDecay_stepSize = 10000
    lrDecay_gamma = 0.9
    weight_decay=0
    finalNumParticles = 64
    numParticles = 1
    learningArgs = [learningRate,lrDecay_stepSize,lrDecay_gamma,weight_decay,numParticles]
    argDim = [inStateDim,inMapDim,inActionDim,outStateDim]
    convSizes = [[32,5],[32,4],[32,3]]
    lstmSize = [3,1024]#,128]
    networkSizes = [convSizes,lstmSize]
    motionModelArgs = [argDim,networkSizes]
    Learn = learnLSTM(learningArgs,motionModelArgs)

    trainBatchSize = 2
    sequenceLength = 64
    trainingSet = [0,0.8]

    numUpdates = 500000000
    for updateCount in range(numUpdates):
        dataBatch = cpuReplayBuffer.getRandSequenceFixedLengthInterleaved(sequenceLength,trainBatchSize,device=device,percentageRange=trainingSet)
        trainLoss = Learn.updateMotionModel(dataBatch,sequenceLength,trainBatchSize)
        if updateCount%10==0:
            writer.add_scalar('train/mse_loss',trainLoss,updateCount)
            print("updateCount: " + str(updateCount) + " trainLoss: " + str(trainLoss) + " sequenceLength: " + str(sequenceLength) + " lr: " + str(Learn.optimizer.state_dict()['param_groups'][-1]['lr']))
        if updateCount%1000==0:
            torch.save(Learn.MotionModel.state_dict(), 'motionModels/lstmMotionModelV2.pt')











