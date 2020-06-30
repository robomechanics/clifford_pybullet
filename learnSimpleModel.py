import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from motionModel import simpleMotionModel
from replayBuffer import ReplayBuffer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from torch.utils.tensorboard import SummaryWriter

class learnSimpleMotionModel(object):
    def __init__(self,learningArgs):
        lr = learningArgs[0]
        lrDecay_stepSize = learningArgs[1]
        lrDecay_gamma = learningArgs[2]
        weight_decay = learningArgs[3]
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.MotionModel = simpleMotionModel().to(self.device)
        self.optimizer = Adam(self.MotionModel.parameters(),lr = lr,weight_decay=weight_decay)
        self.lrScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lrDecay_stepSize, gamma=lrDecay_gamma)
        self.criterion = torch.nn.MSELoss()
    def updateMotionModel(self,dataBatch):
        self.MotionModel.train()
        self.optimizer.zero_grad()
        actualNextStates = dataBatch[1][0]
        predictedNextState = self.MotionModel(dataBatch[0])
        loss = self.criterion(actualNextStates,predictedNextState)
        loss.backward()
        self.optimizer.step()
        self.lrScheduler.step()
        return loss.item()
    def evalMotionModel(self,dataBatch):
        self.MotionModel.eval()
        actualNextStates = dataBatch[1][0]
        predictedNextState = self.MotionModel(dataBatch[0])
        loss = self.criterion(actualNextStates,predictedNextState)
        print("max prediction")
        print(torch.max(predictedNextState))
        return loss.item()

if __name__ == '__main__':
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()
    cpuReplayBuffer = ReplayBuffer(loadDataPrefix='simData/simple',saveDataPrefix='simData/simple',chooseCPU = True)
    cpuReplayBuffer.loadData(matchLoadSize=True)

    # training/ neural network parameters
    learningRate = 0.01
    lrDecay_stepSize = 1000
    lrDecay_gamma = 0.9
    weight_decay=0.0
    learningArgs = [learningRate,lrDecay_stepSize,lrDecay_gamma,weight_decay]
    Learn = learnSimpleMotionModel(learningArgs)

    trainBatchSize = 128
    trainingSet = [0,0.8]
    testBatchSize = 512
    testSet = [trainingSet[1],1.0]
    numUpdates = 5000000
    for updateCount in range(numUpdates):
        dataBatch = cpuReplayBuffer.getRandBatch(trainBatchSize,device=device,percentageRange=trainingSet)
        trainLoss = Learn.updateMotionModel(dataBatch)
        if updateCount%100==0:
            trainLoss = Learn.evalMotionModel(dataBatch)
            dataBatch = cpuReplayBuffer.getRandBatch(testBatchSize,device=device,percentageRange=testSet)
            testLoss = Learn.evalMotionModel(dataBatch)
            writer.add_scalar('train/mse_loss',trainLoss,updateCount)
            writer.add_scalar('test/mse_loss',testLoss,updateCount)
            print("updateCount: " + str(updateCount) + " testLoss: " +str(testLoss) + " trainLoss: " + str(trainLoss) + " lr: " + str(Learn.optimizer.state_dict()['param_groups'][-1]['lr']))
            print(Learn.MotionModel.wheelBase)
            print(Learn.MotionModel.driveScale)
        if updateCount%50000==0:
            torch.save(Learn.MotionModel.state_dict(), 'motionModels/simple.pt')






