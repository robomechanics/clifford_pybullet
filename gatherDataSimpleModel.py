import pybullet as p
import time
import pybullet_data
from RandomRockyTerrain import RandomRockyTerrain
from cliffordFixed import Clifford
import concurrent.futures
import threading
from simController import simController
from replayBuffer import ReplayBuffer
import matplotlib.pyplot as plt
def simpleData(rawData):
    simpleModelInput = [rawData[0][2]]
    simpleModelOutput = [[rawData[1][0][0],rawData[1][0][1],rawData[4]]]
    processedData = [simpleModelInput,simpleModelOutput,rawData[2]]
    return processedData

def runSim(sim):
    print("starting " + str(sim.physicsClientId))
    sim.terrain.generate(cellHeightScale=0,perlinHeightScale=0)
    sim.resetClifford()
    collectedData = []
    startTime = time.time()
    while time.time()-startTime<30.:
        collectedData.append(simpleData(sim.controlLoopStep(sim.randomDriveAction())))
        if collectedData[-1][2]:
            sim.terrain.generate(cellHeightScale=0,perlinHeightScale=0)
            sim.resetClifford()
    return collectedData

replayBufferLength = 100000
numParallelSims = 16
sims = []
# set up simulations
for i in range(numParallelSims):
    if i == -1:
        physicsClientId = p.connect(p.GUI)
    else:
        physicsClientId = p.connect(p.DIRECT)
    sims.append(simController(physicsClientId=physicsClientId))

data = simpleData(sims[0].controlLoopStep([0,0]))
replayBuffer = ReplayBuffer(replayBufferLength,data[0],data[1],saveDataPrefix='simData/simple',chooseCPU=True)

sTime = time.time()
executor = concurrent.futures.ProcessPoolExecutor()
while not replayBuffer.bufferFilled:
    results = executor.map(runSim,sims)
    for result in results:
        for data in result:
            replayBuffer.addData(data[0],data[1])
    print("replay buffer index: " + str(replayBuffer.bufferIndex) + ", rtf: " + str(replayBuffer.bufferIndex*0.25/(time.time()-sTime)))
    print("estimated time left: " + str((replayBufferLength-replayBuffer.bufferIndex)/(replayBuffer.bufferIndex)*(time.time()-sTime)/60./60.) + "hours")  
replayBuffer.saveData()
executor.shutdown()
for i in range(numParallelSims):
    p.disconnect(physicsClientId=sims[i].physicsClientId)