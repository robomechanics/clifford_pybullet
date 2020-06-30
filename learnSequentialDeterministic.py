import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from motionModel import deterministicMotionModel
from replayBuffer import sequentialReplayBuffer
from cliffordStateTransformation import cliffordStateTransformation
from torch.utils.tensorboard import SummaryWriter

class learnSequentialDeterministic(object):
    def __init__(self,learningArgs,motionModelArgs):
        lr = learningArgs[0]
        lrDecay_stepSize = learningArgs[1]
        lrDecay_gamma = learningArgs[2]
        weight_decay = learningArgs[3]
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.MotionModel = deterministicMotionModel(motionModelArgs,batchNormsOn = False).to(self.device)
        self.optimizer = Adam(self.MotionModel.parameters(),lr = lr,weight_decay=weight_decay)
        self.lrScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lrDecay_stepSize, gamma=lrDecay_gamma)
        self.criterion = torch.nn.MSELoss()
    def updateMotionModel(self,dataBatch):
        self.MotionModel.train()
        self.optimizer.zero_grad()
        loss = torch.zeros(1,requires_grad=True).to(self.device)
        lossCount = 0
        for i in range(dataBatch[2].shape[0]-1):
            cliffordPredState = cliffordStateTransformation(dataBatch[0][3][dataBatch[2][i],:].repeat(1,1))
            for j in range(dataBatch[2][i],dataBatch[2][i+1]-1):
                mmInput = [dataBatch[0][k][j:j+1,:] for k in range(len(dataBatch[0])-1)]
                relativePrediction = self.MotionModel(mmInput)
                absolutePrediction = cliffordPredState.moveState(relativePrediction)
                actualNextState = dataBatch[0][3][j+1:j+2,:]
                loss = loss+self.criterion(absolutePrediction,actualNextState)
                lossCount += 1
        loss = loss/lossCount
        loss.backward()
        self.optimizer.step()
        self.lrScheduler.step()
        return loss.item()
    def evalMotionModel(self,dataBatch):
        self.MotionModel.eval()
        loss = torch.zeros(1).to(self.device)
        for i in range(dataBatch[2].shape[0]-1):
            cliffordPredState = cliffordStateTransformation(dataBatch[0][3][dataBatch[2][i],:].repeat(1,1))
            for j in range(dataBatch[2][i],dataBatch[2][i+1]-1):
                mmInput = [dataBatch[0][k][j:j+1,:] for k in range(len(dataBatch[0])-1)]
                relativePrediction = self.MotionModel(mmInput)
                absolutePrediction = cliffordPredState.moveState(relativePrediction)
                actualNextState = dataBatch[0][3][j+1:j+2,:]
                loss = loss+self.criterion(absolutePrediction,actualNextState)
        return loss.item()
    def updateMotionModelFixedLength(self,dataBatch):
        self.MotionModel.train()
        self.optimizer.zero_grad()
        loss = torch.zeros(1,requires_grad=True).to(self.device)
        lossCount = 0
        cliffordPredState = cliffordStateTransformation(dataBatch[0][0][3])
        for stepNum in range(0,len(dataBatch)-1):
            if torch.isnan(torch.sum(cliffordPredState.currentState)):
                print("bad absolute state")
            mmInput = [cliffordPredState.stateToNNInState()]+dataBatch[stepNum][0][1:]
            if torch.isnan(torch.sum(mmInput[0])).item():
                print("bad mm input")
            relativePrediction = self.MotionModel(mmInput)
            if torch.isnan(torch.sum(relativePrediction)).item():
                print("bad relative prediction")
            absolutePrediction = cliffordPredState.moveState(relativePrediction)
            actualNextState = dataBatch[stepNum+1][0][3]
            lossCount+=1
            loss = loss + self.criterion(absolutePrediction,actualNextState)
        loss = loss/lossCount
        #loss = self.criterion(absolutePrediction,actualNextState)
        if torch.isnan(torch.sum(torch.autograd.grad(loss, absolutePrediction,retain_graph=True)[0])):
            print("bad absolutePrediction prediction")
        if torch.isnan(torch.sum(torch.autograd.grad(loss, relativePrediction,retain_graph=True)[0])):
            print("bad relativePrediction prediction")
        loss.backward()
        if torch.isnan(loss).item():
            print("bad loss value")
        parameterSum = 0
        parameterGradientSum = 0
        for param in self.MotionModel.parameters():
          if param.grad is not None:
            parameterGradientSum+=torch.sum(param.grad).item()
        if np.isnan(parameterGradientSum):
            print('nan parameter gradient')
        self.optimizer.step()
        for param in self.MotionModel.parameters():
          if param.grad is not None:
            parameterSum+=torch.sum(param).item()
        if np.isnan(parameterSum):
            print('nan parameter')
        self.lrScheduler.step()
        return loss.item()
    def updateMotionModelNonSequential(self,dataBatch):
        self.MotionModel.train()
        self.optimizer.zero_grad()
        actualNextStates = dataBatch[1][0]
        predictedNextStates = self.MotionModel(dataBatch[0])
        loss = self.criterion(actualNextStates,predictedNextStates)
        loss.backward()
        self.optimizer.step()
        self.lrScheduler.step()
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
    learningRate = 0.00005
    lrDecay_stepSize = 2000
    lrDecay_gamma = 1#0.9
    weight_decay=0
    learningArgs = [learningRate,lrDecay_stepSize,lrDecay_gamma,weight_decay]
    argDim = [inStateDim,inMapDim,inActionDim,outStateDim]
    convSizes = [[32,5],[32,4],[32,3]]
    fcSizes = [1024,512,256]#,128]
    networkSizes = [convSizes,fcSizes]
    dropout_ps = [0,0,0]
    motionModelArgs = [argDim,networkSizes,dropout_ps]
    Learn = learnSequentialDeterministic(learningArgs,motionModelArgs)
    trainBatchSize = 64
    sequenceLength = 2
    maxSequenceLength = 50
    trainingSet = [0,0.8]
    testBatchSize = 512
    testSet = [trainingSet[1],1.0]
    smoothing = 0.9
    smoothedLoss = 1
    switchValue = 0.03

    numUpdates = 500000
    for updateCount in range(numUpdates):
        #if updateCount%2000==0 and sequenceLength<maxSequenceLength:
        if smoothedLoss < switchValue and sequenceLength<maxSequenceLength:
            sequenceLength+=1
            smoothedLoss = switchValue*1.1
        #dataBatch = cpuReplayBuffer.getRandSequence(trainBatchSize,device=device,percentageRange=trainingSet)
        #trainLoss = Learn.updateMotionModel(dataBatch)
        dataBatch = cpuReplayBuffer.getRandSequenceFixedLength(trainBatchSize,sequenceLength,device=device,percentageRange=trainingSet)
        trainLoss = Learn.updateMotionModelFixedLength(dataBatch)
        smoothedLoss = smoothing*smoothedLoss + (1-smoothing)*trainLoss
        if updateCount%10==0:
            writer.add_scalar('train/mse_loss',trainLoss,updateCount)
            writer.add_scalar('train/log_loss',smoothedLoss,updateCount)
            print("updateCount: " + str(updateCount) + " smoothedLoss: " +str(smoothedLoss) + " trainLoss: " + str(trainLoss) + " sequenceLength: " + str(sequenceLength) + " lr: " + str(Learn.optimizer.state_dict()['param_groups'][-1]['lr']))
        if updateCount%1000==0:
            torch.save(Learn.MotionModel.state_dict(), 'motionModels/v2sequentialDeterministic.pt')











