import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from replayBuffer import sequentialReplayBuffer
from cliffordStateTransformation import cliffordStateTransformation
from torch.utils.tensorboard import SummaryWriter

class inputVAE(nn.Module):
    def __init__(self, dimensionArgs,nnSize):
        super(inputVAE, self).__init__()
        # set size of neural network input
        self.inStateDim = dimensionArgs[0]
        self.inMapDim = dimensionArgs[1]
        self.inActionDim = dimensionArgs[2]
        self.inputLatentDim = dimensionArgs[3]

        # set sizes of neural networks
        encoderConvSize = nnSize[0]
        encoderFCSize = nnSize[1]

        # setup input encoder & decoder conv layers
        self.inputEncoderConvs = nn.ModuleList([])
        self.inputDecoderConvs = nn.ModuleList([])
        lastDim = [1,self.inMapDim] # num layers of channels of input, dimension of input
        for i in range(len(encoderConvSize)):
            self.inputEncoderConvs.append(nn.Conv2d(lastDim[0],encoderConvSize[i][0],encoderConvSize[i][1]))
            self.inputDecoderConvs.insert(0,nn.ConvTranspose2d(encoderConvSize[i][0],lastDim[0],encoderConvSize[i][1]))
            lastDim = [encoderConvSize[i][0],lastDim[1]-encoderConvSize[i][1]+1]
        self.inputEncoderConvOutputDim = lastDim[0]*lastDim[1]*lastDim[1]
        self.inputDecoderConvInputDim = [lastDim[0],lastDim[1]]
        
        #set up input encoder FC layers
        self.inputEncoderFCs = nn.ModuleList([])
        lastDim = self.inputEncoderConvOutputDim+self.inStateDim+self.inActionDim # size of input to first LSTM
        for i in range(len(encoderFCSize)):
            self.inputEncoderFCs.append(nn.Linear(lastDim,encoderFCSize[i]))
            lastDim = encoderFCSize[i]
        self.inputEncoderMeanFC = nn.Linear(lastDim,self.inputLatentDim)
        self.inputEncoderLogVarFC = nn.Linear(lastDim,self.inputLatentDim)

        #set up input decoder FC layers
        self.inputDecoderFCs = nn.ModuleList([])
        lastDim = self.inputLatentDim
        for i in reversed(range(len(encoderFCSize))):
            self.inputDecoderFCs.append(nn.Linear(lastDim,encoderFCSize[i]))
            lastDim = encoderFCSize[i]
        self.inputDecoderInStateFC = nn.Linear(lastDim,self.inStateDim)
        self.inputDecoderInActionFC = nn.Linear(lastDim,self.inActionDim)
        self.inputDecoderMapFC = nn.Linear(lastDim,self.inputEncoderConvOutputDim)
    def inputEncoder(self,data):
        # takes in robot state, map, action and predicts distribution in latent space
        rState = data[0]
        rMap = data[1]
        rAction = data[2]
        for i in range(len(self.inputEncoderConvs)):
            rMap = self.inputEncoderConvs[i](rMap)
            rMap = F.relu(rMap)
            #rMap = F.sigmoid(rMap)
            #rMap = F.tanh(rMap)
        rMap = rMap.view(-1,self.inputEncoderConvOutputDim)
        connected = torch.cat((rMap,rState,rAction),axis=1)
        for i in range(len(self.inputEncoderFCs)):
            connected = self.inputEncoderFCs[i](connected)
            connected = F.relu(connected)
            #connected = F.sigmoid(connected)
            #connected = F.tanh(connected)
        #return F.tanh(self.inputEncoderMeanFC(connected)),F.tanh(self.inputEncoderLogVarFC(connected))
        return self.inputEncoderMeanFC(connected),self.inputEncoderLogVarFC(connected)
    def inputDecoder(self,latentVector):
        connected = latentVector
        for i in range(len(self.inputDecoderFCs)):
            connected = self.inputDecoderFCs[i](connected)
            connected = F.relu(connected)
            #connected = F.sigmoid(connected)
            #connected = F.tanh(connected)
        rMap = self.inputDecoderMapFC(connected).reshape(-1,self.inputDecoderConvInputDim[0],self.inputDecoderConvInputDim[1],self.inputDecoderConvInputDim[1])
        for i in range(len(self.inputDecoderConvs)):
            rMap = self.inputDecoderConvs[i](rMap)
            if i < len(self.inputDecoderConvs)-1:
                rMap = F.relu(rMap)
                #rMap = F.sigmoid(rMap)
                #rMap = F.tanh(rMap)
        return self.inputDecoderInStateFC(connected),rMap,self.inputDecoderInActionFC(connected)
    def sampleLatent(self, mu, logvar):
        # samples latent vector from latent vector distribution
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self,inputData):
        latentMu,latentLogVar = self.inputEncoder(inputData)
        latentVecs = self.sampleLatent(latentMu,latentLogVar)
        reconstruction = self.inputDecoder(latentVecs)
        return latentVecs,latentMu,latentLogVar,reconstruction

class latentMotionModel(nn.Module):
    def __init__(self, dimensionArgs,motionModelLSTMSize):
        super(latentMotionModel, self).__init__()
        # set size of neural network input
        self.inputLatentDim = dimensionArgs[3]
        self.outStateDim = dimensionArgs[4]

        #set up motion model layers
        self.latentMotionModelLSTM = nn.LSTM(input_size=self.inputLatentDim,num_layers=motionModelLSTMSize[0],hidden_size=motionModelLSTMSize[1],batch_first = True)
        self.latentMotionModelFC = nn.Linear(motionModelLSTMSize[1],self.outStateDim)
    def forward(self,latentData,sequenceLength,batchSize,prevLSTMStates=None):
        latentData = latentData.view(batchSize,sequenceLength,-1)
        if prevLSTMStates == None:
            lstmOutput,lstmStates = self.latentMotionModelLSTM(latentData)
        else:
            lstmOutput,lstmStates = self.latentMotionModelLSTM(latentData,prevLSTMStates)
        lstmOutput = lstmOutput.reshape(batchSize*sequenceLength,-1)
        output = self.latentMotionModelFC(lstmOutput)
        return (output,lstmStates)

class trainSeparate(object):
    def __init__(self,learningArgs,dimensionArgs,VAENNSize,motionModelNNSize,lossArgs):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # set up training
        lr = learningArgs[0]
        lrDecay_stepSize = learningArgs[1]
        lrDecay_gamma = learningArgs[2]
        weight_decay = learningArgs[3]
        self.inputVAE = inputVAE(dimensionArgs,VAENNSize).to(self.device)
        self.latentMotionModel = latentMotionModel(dimensionArgs,motionModelNNSize).to(self.device)
        self.predictionLossWeight = lossArgs[0]
        self.reconstructionLossWeight = lossArgs[1]
        self.KLDWeight = lossArgs[2]
        self.inputVAEOptimizer = Adam(self.inputVAE.parameters(),lr = lr,weight_decay=weight_decay)
        self.motionModelOptimizer = Adam(self.latentMotionModel.parameters(),lr = lr,weight_decay=weight_decay)
        self.allOptimizer = Adam(list(self.inputVAE.parameters()) + list(self.latentMotionModel.parameters()),lr = lr,weight_decay=weight_decay)
        #self.lrScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lrDecay_stepSize, gamma=lrDecay_gamma) # for lowering step size as training progresses
    def updateInputVAE(self,dataBatch):
        # set up update
        self.inputVAE.train()
        self.inputVAEOptimizer.zero_grad()
        # VAE computations
        cliffordPredState = cliffordStateTransformation(dataBatch[0][3])
        mmInput = [cliffordPredState.stateToNNInState()]+[dataBatch[0][i] for i in range(1,len(dataBatch[0]))]
        latentVecs, latentMu, latentLogVar, reconstruction = self.inputVAE(mmInput)
        # calculate loss and optimize
        vaeLoss, reconstructionLoss, KLD = self.vaeLoss(latentMu,latentLogVar,reconstruction,mmInput)
        vaeLoss.backward()
        self.inputVAEOptimizer.step()
        return vaeLoss.item(), reconstructionLoss.item(), KLD.item()
    def updateMotionModel(self,dataBatch,sequenceLength,batchSize,prevLSTMStates=None):
        self.latentMotionModel.train()
        self.inputVAE.eval()
        self.motionModelOptimizer.zero_grad()
        # compute VAE and predicted motion
        cliffordPredState = cliffordStateTransformation(dataBatch[0][3])
        mmInput = [cliffordPredState.stateToNNInState()]+[dataBatch[0][i] for i in range(1,len(dataBatch[0]))]
        latentVecs, latentMu, latentLogVar, reconstruction = self.inputVAE(mmInput)
        prediction, lstmStates = self.latentMotionModel(latentMu, sequenceLength, batchSize, prevLSTMStates)
        relativeGroundTruth = dataBatch[1][0]
        # calculate loss and optimize
        predictionLoss = F.mse_loss(prediction,relativeGroundTruth)
        predictionLoss.backward()
        self.motionModelOptimizer.step()
        return predictionLoss.item()
    def updateAll(self,dataBatch,sequenceLength,batchSize,prevLSTMStates=None):
        self.latentMotionModel.train()
        self.inputVAE.train()
        self.allOptimizer.zero_grad()
        # compute VAE and predicted motion
        cliffordPredState = cliffordStateTransformation(dataBatch[0][3])
        mmInput = [cliffordPredState.stateToNNInState()]+[dataBatch[0][i] for i in range(1,len(dataBatch[0]))]
        latentVecs, latentMu, latentLogVar, reconstruction = self.inputVAE(mmInput)
        prediction, lstmStates = self.latentMotionModel(latentMu, sequenceLength, batchSize, prevLSTMStates)
        relativeGroundTruth = dataBatch[1][0]
        # calculate loss and optimize
        vaeLoss, reconstructionLoss, KLD = self.vaeLoss(latentMu,latentLogVar,reconstruction,mmInput)
        predictionLoss = F.mse_loss(prediction,relativeGroundTruth)
        totalLoss = vaeLoss + self.predictionLossWeight*predictionLoss
        totalLoss.backward()
        self.allOptimizer.step()
        return totalLoss.item(), predictionLoss.item(), reconstructionLoss.item(), KLD.item()

    def vaeLoss(self,latentMu, latentLogVar, reconstruction, input):
        reconstructionLoss = F.mse_loss(reconstruction[0],input[0]) + F.mse_loss(reconstruction[1],input[1]) + F.mse_loss(reconstruction[2],input[2])
        KLD = -0.5 * torch.sum(1 + latentLogVar - latentMu.pow(2) - latentLogVar.exp())
        totalLoss = self.reconstructionLossWeight*reconstructionLoss + self.KLDWeight * KLD
        return totalLoss, reconstructionLoss, KLD

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
    latentDim = 32

    # training/ neural network parameters
    learningRate = 0.00025
    lrDecay_stepSize = 10000
    lrDecay_gamma = 0.9
    weight_decay=0
    learningArgs = [learningRate,lrDecay_stepSize,lrDecay_gamma,weight_decay]
    dimensionArgs = [inStateDim,inMapDim,inActionDim,latentDim,outStateDim]
    encoderConvSize = [[32,5],[32,4],[32,3]]
    encoderFCSize = [256,128,64]#[1024,512,256]
    inputVAEArgs = [encoderConvSize,encoderFCSize]
    motionModelLSTMSize = [3,1024]
    predictionWeight = 1
    reconstructionLossWeight = 0.00
    KLDWeight = 0.0000
    lossArgs = [predictionWeight,reconstructionLossWeight,KLDWeight]
    Learn = trainSeparate(learningArgs,dimensionArgs,inputVAEArgs,motionModelLSTMSize,lossArgs)
    #Learn.inputVAE.load_state_dict(torch.load('motionModels/inputVAE.pt'))
    #Learn.latentMotionModel.load_state_dict(torch.load('motionModels/latentMotionModel.pt'))

    trainBatchSize = 4
    sequenceLength = 64
    trainingSet = [0,0.8]

    numUpdates = 500000000
    for updateCount in range(numUpdates):
        dataBatch = cpuReplayBuffer.getRandSequenceFixedLengthInterleaved(sequenceLength,trainBatchSize,device=device,percentageRange=trainingSet)
        if updateCount < 00000:
            totalLoss, reconstructionLoss, KLD = Learn.updateInputVAE(dataBatch)
            if updateCount%10==0:
                writer.add_scalar('train/mse_loss',totalLoss,updateCount)
                print("updateCount: " + str(updateCount) + " totalLoss: " + str(totalLoss) + " reconstructionLoss: " + str(reconstructionLoss) + " KLD: " + str(KLD))
            if updateCount%1000 == 0:
                torch.save(Learn.inputVAE.state_dict(), 'motionModels/inputVAE.pt')
        elif updateCount < 0:
            predictionLoss = Learn.updateMotionModel(dataBatch,sequenceLength,trainBatchSize)
            if updateCount%10==0:
                writer.add_scalar('train/mse_loss',predictionLoss,updateCount)
                print("updateCount: " + str(updateCount) + " predictionLoss: " + str(predictionLoss))
            if updateCount%1000 == 0:
                torch.save(Learn.latentMotionModel.state_dict(), 'motionModels/latentMotionModel.pt')
        else:
            totalLoss,predictionLoss,reconstructionLoss,KLD = Learn.updateAll(dataBatch,sequenceLength,trainBatchSize)
            if updateCount%10==0:
                writer.add_scalar('train/mse_loss',totalLoss,updateCount)
                print("updateCount: " + str(updateCount) + " totalLoss: " + str(totalLoss) + " predictionLoss: " + str(predictionLoss) + " reconstructionLoss: " + str(reconstructionLoss) + " KLD: " + str(KLD))
            if updateCount%1000 == 0:
                torch.save(Learn.inputVAE.state_dict(), 'motionModels/inputVAE.pt')
                torch.save(Learn.latentMotionModel.state_dict(), 'motionModels/latentMotionModel.pt')

















