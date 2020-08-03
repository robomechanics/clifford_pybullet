import pybullet as p
from simController import simController
from motionModel import simpleMotionModel
from motionModel import probabilisticMotionModel
import matplotlib.pyplot as plt
import torch
import numpy as np
from cliffordStateTransformation import cliffordStateTransformation
from motionModel import deterministicMotionModel
from motionModel import probabilisticMotionModel
from learnLSTM import lstmMotionModel
from learnVAELSTM import inputVAE, latentMotionModel


evalSequential = False 
evalSimple = False
evalProbabilistic = False
evalLSTM = True
evalVAE = True
physicsClientId = p.connect(p.GUI)
sim = simController(timeStep=1./500.,stepsPerControlLoop=50,numSolverIterations=300,physicsClientId=physicsClientId)
sim.terrain.generate(AverageAreaPerCell = 1,cellPerlinScale=5,cellHeightScale=0.6,smoothing=1)
sim.resetClifford()
startState = sim.controlLoopStep([0,0])
actualX = [startState[3][0]]
actualY = [startState[3][1]]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if evalSimple:
    predictedSimplePose = [startState[3][0],startState[3][1],startState[4]]
    smm = simpleMotionModel().to(device)
    smm.eval()
    predSimpleX = [startState[3][0]]
    predSimpleY = [startState[3][1]]
if evalSequential:
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
    motionModelDet = deterministicMotionModel(motionModelArgs).to(device)
    motionModelDet.load_state_dict(torch.load('motionModels/v2sequentialDeterministic.pt'))
    motionModelDet.eval()
    cliffordStatePredDet = cliffordStateTransformation(torch.tensor(startState[3]).unsqueeze(0).to(device))
    predSequentialX = [startState[3][0]]
    predSequentialY = [startState[3][1]]
if evalProbabilistic:
    numParticles = 10
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
    motionModelProb = probabilisticMotionModel(motionModelArgs).to(device)
    motionModelProb.load_state_dict(torch.load('motionModels/sequentialProbailistic.pt'))
    motionModelProb.eval()
    cliffordStatePredProb = cliffordStateTransformation(torch.tensor(startState[3]).unsqueeze(0).to(device),numParticles=numParticles)
    predProbabilisticX = [[startState[3][0]] for i in range(numParticles)]
    predProbabilisticY = [[startState[3][1]] for i in range(numParticles)]
if evalLSTM:
    inStateDim = 9
    inActionDim = len(startState[0][2])
    inMapDim = startState[0][1].shape[1]
    outStateDim = 13
    argDim = [inStateDim,inMapDim,inActionDim,outStateDim]
    convSizes = [[32,5],[32,4],[32,3]]
    lstmSize = [3,1024]#,128]
    networkSizes = [convSizes,lstmSize]
    motionModelArgs = [argDim,networkSizes]
    motionModelLSTM = lstmMotionModel(motionModelArgs).to(device)
    motionModelLSTM.load_state_dict(torch.load('motionModels/lstmMotionModelV2.pt'))
    motionModelLSTM.eval()
    prevLSTMStates = None
    cliffordStateLSTM = cliffordStateTransformation(torch.tensor(startState[3][0:-14]).unsqueeze(0).to(device))
    predLSTMX = [startState[3][0]]
    predLSTMY = [startState[3][1]]
if evalVAE:
    inStateDim = 9
    inActionDim = len(startState[0][2])
    inMapDim = startState[0][1].shape[1]
    outStateDim = 13
    latentDim = 32
    dimensionArgs = [inStateDim,inMapDim,inActionDim,latentDim,outStateDim]
    encoderConvSize = [[32,5],[32,4],[32,3]]
    encoderFCSize = [256,128,64]#[1024,512,256]
    inputVAEArgs = [encoderConvSize,encoderFCSize]
    motionModelLSTMSize = [3,1024]
    vaeEncoder = inputVAE(dimensionArgs,inputVAEArgs).to(device)
    vaeEncoder.load_state_dict(torch.load('motionModels/inputVAE.pt'))
    vaeEncoder.eval()
    vaeMotionModel = latentMotionModel(dimensionArgs,motionModelLSTMSize).to(device)
    vaeMotionModel.load_state_dict(torch.load('motionModels/latentMotionModel.pt'))
    vaeMotionModel.eval()
    prevVAELSTMStates = None
    cliffordStateVAE = cliffordStateTransformation(torch.tensor(startState[3][0:-14]).unsqueeze(0).to(device))
    predVAEX = [startState[3][0]]
    predVAEY = [startState[3][1]]


for t in range(50):
    print(t)
    data = sim.controlLoopStep(sim.randomDriveAction())
    if data[2]:
        break
    actualX.append(data[3][0])
    actualY.append(data[3][1])
    inAction = torch.FloatTensor(data[0][2]).unsqueeze(0).to(device)
    if evalSimple:
        prediction = smm([inAction])[0]
        newX = predictedSimplePose[0] + prediction[0].item()*np.cos(predictedSimplePose[2]) - prediction[1].item()*np.sin(predictedSimplePose[2])
        newY = predictedSimplePose[1] + prediction[0].item()*np.sin(predictedSimplePose[2]) + prediction[1].item()*np.cos(predictedSimplePose[2])
        newHeading = predictedSimplePose[2] + prediction[2].item()
        predictedSimplePose = [newX,newY,newHeading]
        predSimpleX.append(predictedSimplePose[0])
        predSimpleY.append(predictedSimplePose[1])
    if evalSequential:
        inState = cliffordStatePredDet.stateToNNInState()
        inAction = torch.FloatTensor(data[0][2]).unsqueeze(0).to(device)
        #inMap = torch.from_numpy(data[0][1]).unsqueeze(0).float().to(device)
        pose = cliffordStatePredDet.posHeading()[0,:].detach().cpu().numpy()
        inMap = torch.from_numpy(sim.terrain.robotHeightMap(pose[0:3],pose[3],sim.rMapWidth,sim.rMapHeight,sim.rMapScale)).unsqueeze(0).unsqueeze(0).float().to(device)
        prediction = motionModelDet([inState,inMap,inAction])
        predictedPose = cliffordStatePredDet.moveState(prediction)
        predSequentialX.append(predictedPose[0,0])
        predSequentialY.append(predictedPose[0,1])
    if evalProbabilistic:
        inStates = cliffordStatePredProb.stateToNNInState()
        inAction = torch.FloatTensor(data[0][2]).unsqueeze(0).to(device).repeat_interleave(numParticles,dim=0)
        inMaps = torch.from_numpy(data[0][1]).unsqueeze(0).float().to(device).repeat_interleave(numParticles,dim=0)
        poses = cliffordStatePredProb.posHeading().detach().cpu().numpy()
        if poses.max() > 7.5 or poses.min() < -7.5:
            break
        #print(poses)
        maps = [torch.from_numpy(sim.terrain.robotHeightMap(poses[i,0:3],poses[i,3],sim.rMapWidth,sim.rMapHeight,sim.rMapScale)).unsqueeze(0).unsqueeze(0).float() for i in range(numParticles)]
        inMaps = torch.cat(maps,dim=0).to(device)
        predDists = motionModelProb([inStates,inMaps,inAction])
        print(predDists)
        predSamples = motionModelProb.sampleFromDist(predDists)
        predictedPoses = cliffordStatePredProb.moveState(predSamples)
        for i in range(numParticles):
            predProbabilisticX[i].append(predictedPoses[i,0])
            predProbabilisticY[i].append(predictedPoses[i,1])
    if evalLSTM:
        inState = cliffordStateLSTM.stateToNNInState()
        inAction = torch.FloatTensor(data[0][2]).unsqueeze(0).to(device)
        pose = cliffordStateLSTM.posHeading()[0,:].detach().cpu().numpy()
        inMap = torch.from_numpy(sim.terrain.robotHeightMap(pose[0:3],pose[3],sim.rMapWidth,sim.rMapHeight,sim.rMapScale)).unsqueeze(0).unsqueeze(0).float().to(device)
        prediction,prevLSTMStates = motionModelLSTM([inState,inMap,inAction],1,1,prevLSTMStates)
        predictedPose = cliffordStateLSTM.moveState(prediction)
        predLSTMX.append(predictedPose[0,0])
        predLSTMY.append(predictedPose[0,1])
    if evalVAE:
        inState = cliffordStateVAE.stateToNNInState()
        inAction = torch.FloatTensor(data[0][2]).unsqueeze(0).to(device)
        pose = cliffordStateVAE.posHeading()[0,:].detach().cpu().numpy()
        inMap = torch.from_numpy(sim.terrain.robotHeightMap(pose[0:3],pose[3],sim.rMapWidth,sim.rMapHeight,sim.rMapScale)).unsqueeze(0).unsqueeze(0).float().to(device)
        Encoding = vaeEncoder([inState,inMap,inAction])[1]
        prediction, prevVAELSTMStates = vaeMotionModel(Encoding,1,1,prevVAELSTMStates)
        predictedPose = cliffordStateVAE.moveState(prediction)
        predVAEX.append(predictedPose[0,0])
        predVAEY.append(predictedPose[0,1])

p.disconnect(physicsClientId=physicsClientId)
plt.figure()
for i in range(len(actualX)):
    plt.clf()
    plt.plot(actualX[0:i],actualY[0:i],label="Actual Robot Motion")
    if evalSequential:
        plt.plot(predSequentialX[0:i],predSequentialY[0:i],label="Learned Motion Model")
    if evalSimple:
        plt.plot(predSimpleX[0:i],predSimpleY[0:i],label="Handcrafted Motion Model")
    if evalProbabilistic:
        for j in range(numParticles):
            plt.plot(predProbabilisticX[j][0:i],predProbabilisticY[j][0:i])
    if evalLSTM:
        plt.plot(predLSTMX[0:i],predLSTMY[0:i],label="LSTM Motion Model")
    if evalVAE:
        plt.plot(predVAEX[0:i],predVAEY[0:i],label="VAE Motion Model")
    plt.xlim((-10,10))
    plt.ylim((-10,10))
    plt.legend()
    plt.pause(0.1)
plt.show()