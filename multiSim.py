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
def runSim(sim):
    print("starting " + str(sim.physicsClientId))
    collectedData = []
    startTime = time.time()
    sim.lastStateRecordFlag
    while time.time()-startTime<10.:
        collectedData.append(sim.controlLoopStep(sim.randomDriveAction()))
        if collectedData[-1][2]:
            sim.terrain.generate()
            sim.resetClifford()
    return collectedData

if __name__=="__main__":
    replayBufferLength = 500000
    numParallelSims = 16
    sims = []
    # set up simulations
    for i in range(numParallelSims):
        if i == -1:
            physicsClientId = p.connect(p.GUI)
        else:
            physicsClientId = p.connect(p.DIRECT)
        sims.append(simController(physicsClientId=physicsClientId))

    data = sims[0].controlLoopStep([0,0])
    replayBuffer = ReplayBuffer(replayBufferLength,data[0],data[1],saveDataPrefix='simData/',chooseCPU=True)

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