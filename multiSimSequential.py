import pybullet as p
import time
import pybullet_data
from RandomRockyTerrain import RandomRockyTerrain
from cliffordFixed import Clifford
import concurrent.futures
import threading
from simController import simController
from replayBuffer import sequentialReplayBuffer
import matplotlib.pyplot as plt

def runSim(input):
    sim = input[0]
    replayBuffer = input[1]
    print("starting " + str(sim.physicsClientId))
    sim.lastStateRecordFlag = False
    sim.terrain.generate(AverageAreaPerCell = 1,cellPerlinScale=5,cellHeightScale=1)
    sTime = time.time()
    maxSequenceLength = 150
    sequenceLengthCount = 0
    while not replayBuffer.bufferFilled:
        output = sim.controlLoopStep(sim.randomDriveAction())
        replayBuffer.addData(output[0],output[1],output[2])
        sequenceLengthCount+=1
        if output[2] or sequenceLengthCount==maxSequenceLength:
            sim.terrain.generate(AverageAreaPerCell = 1,cellPerlinScale=5,cellHeightScale=1)
            sim.resetClifford()
            sequenceLengthCount=0
        if replayBuffer.bufferIndex%100==0:
            currentTime = time.time()-sTime
            print("simIndex: " + str(sim.physicsClientId) + " bufferIndex: " + str(replayBuffer.bufferIndex))
            print("rtf: " + str(replayBuffer.bufferIndex*0.1/currentTime) + " estimated time left: " + str((replayBuffer.bufferLength-replayBuffer.bufferIndex)/(replayBuffer.bufferIndex)*currentTime/60./60.) + "hours")  
    return replayBuffer

if __name__=="__main__":
    numParallelSims = 16
    replayBufferLength = int(500000/numParallelSims)
    sims = []
    replayBuffers = []
    # set up simulations
    for i in range(numParallelSims):
        if i == -1:
            physicsClientId = p.connect(p.GUI)
        else:
            physicsClientId = p.connect(p.DIRECT)
        sims.append(simController(timeStep=1./500.,stepsPerControlLoop=50,numSolverIterations=300,physicsClientId=physicsClientId))
        data = sims[-1].controlLoopStep([0,0])
        replayBuffers.append(sequentialReplayBuffer(replayBufferLength,data[0],data[1]))
    allDataReplayBuffer = sequentialReplayBuffer(replayBufferLength*numParallelSims,data[0],data[1],saveDataPrefix='simData/v2')
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(runSim,[[sims[i],replayBuffers[i]] for i in range(numParallelSims)])
    for otherBuffer in results:
        print(otherBuffer.bufferIndex)
        if otherBuffer.bufferFilled:
            allDataReplayBuffer.inheritData(otherBuffer)
        else:
            print("one buffer not filled")
    allDataReplayBuffer.saveData()
    executor.shutdown()
    for i in range(numParallelSims):
        p.disconnect(physicsClientId=sims[i].physicsClientId)