import pybullet as p
import time
import pybullet_data
from RandomRockyTerrain import RandomRockyTerrain
import numpy as np
from cliffordFixed import Clifford
from replayBuffer import ReplayBuffer
from noise import pnoise1
from ouNoise import ouNoise

class simController:
    def __init__(self,sdfRootPath='',physicsClientId=0,timeStep=1./240.,stepsPerControlLoop=60,gravity=-10,numSolverIterations=300,
                rMapWidth=50,rMapHeight=50,rMapScale=0.02,tMapWidth = 300,tMapHeight=300,tMapScale =0.1):
        self.stepsPerControlLoop=stepsPerControlLoop
        self.physicsClientId=physicsClientId
        self.lastStateRecordFlag = False
        self.rMapWidth = rMapWidth
        self.rMapHeight = rMapHeight
        self.rMapScale = rMapScale
        self.timeStep = timeStep
        p.setPhysicsEngineParameter(numSolverIterations=numSolverIterations,physicsClientId=self.physicsClientId)
        p.setGravity(0,0,gravity,physicsClientId=self.physicsClientId)
        p.setTimeStep(timeStep,physicsClientId=self.physicsClientId)
        self.terrain = RandomRockyTerrain(tMapWidth,tMapHeight,tMapScale,tMapScale,physicsClientId=self.physicsClientId)
        self.terrain.generate()#cellHeightScale=0,perlinHeightScale=0.0)
        self.clifford = Clifford(physicsClientId=physicsClientId)
        self.resetClifford()
        self.randDrive = ouNoise()
    def resetClifford(self,doFall=True):
        safeFallHeight = self.terrain.safeFallHeight([0,0])
        self.clifford.reset([[0,0,safeFallHeight],[0,0,0,1]])
        if doFall:
            self.cliffordFall()
    def cliffordFall(self,fallTime=0.5):
        fallSteps = int(np.ceil(fallTime/self.timeStep))
        for i in range(fallSteps):
            self.stepSim()
    def stepSim(self):
        self.clifford.updateSpringForce()
        p.stepSimulation(physicsClientId=self.physicsClientId)
        self.lastStateRecordFlag = False
    def controlLoopStep(self,driveCommand):
        throttle = driveCommand[0]
        steering = driveCommand[1]
        # check if last pose of clifford has been recorded
        if not self.lastStateRecordFlag:
            self.lastPose = self.clifford.getPositionOrientation()
            self.lastVel = self.clifford.getBaseVelocity_body()
            self.lastJointState = self.clifford.measureJoints()
        heightMap = self.terrain.robotHeightMap(self.lastPose[0],self.lastPose[2],self.rMapWidth,self.rMapHeight,self.rMapScale)
        heightMap = np.expand_dims(heightMap,axis=0)
        # Input to NN motion predictor
        # tilt, body twist, joint position & velocity, ground height map, robot action
        nnInput = [self.lastPose[3]+self.lastVel[:]+self.lastJointState[:],heightMap,[driveCommand[0],driveCommand[1]]]
        # command clifford throttle & steering
        self.clifford.drive(throttle)
        self.clifford.steer(steering)
        # Step sim until reach next control loop
        for i in range(self.stepsPerControlLoop):
            self.stepSim()
        # Record outcome state
        newPose = self.clifford.getPositionOrientation()
        invertedLastPose = p.invertTransform(self.lastPose[0],self.lastPose[1])
        relativePose = p.multiplyTransforms(invertedLastPose[0],invertedLastPose[1],newPose[0],newPose[1])
        relativeHeading = newPose[2]-self.lastPose[2]
        baseVel = self.clifford.getBaseVelocity_body()
        jointState = self.clifford.measureJoints()
        # relative position, body twist, joint position and velocity
        nnOutputGroundTruth = [list(relativePose[0])[:]+list(relativePose[1])[:]+baseVel[:]+jointState[:]]
        self.lastPose = newPose
        self.lastVel = baseVel
        self.lastJointState = jointState
        self.lastStateRecordFlag = True
        return nnInput,nnOutputGroundTruth,self.simTerminateCheck(newPose,jointState),newPose,relativeHeading
    def simTerminateCheck(self,cliffordPose,jointState):
        cliffordStuck = False
        # check if suspension is about to invert
        if np.max(jointState[0:4]) > 0.015:
            cliffordStuck = True
        # check if clifford is about to flip
        if cliffordPose[3][0] > 3.14/2.:
            cliffordStuck = True
        # check if clifford is out of bound
        maxZ = np.max(np.abs(self.terrain.gridZ)) + 1.
        maxX = np.max(self.terrain.gridX) - 1.
        maxY = np.max(self.terrain.gridY) - 1.
        if np.abs(cliffordPose[0][0])>maxX or np.abs(cliffordPose[0][1])>maxY or np.abs(cliffordPose[0][2])>maxZ:
            cliffordStuck = True
        return cliffordStuck
    def randomDriveAction(self):
        return self.randDrive.multiGenNoise(50)*np.array([50.,0.5])

if __name__=="__main__":
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    replayBufferLength = 200
    sim = simController()
    data = sim.controlLoopStep([0,0])
    replayBuffer = ReplayBuffer(replayBufferLength,data[0],data[1])
    while not replayBuffer.bufferFilled:
        simData = sim.controlLoopStep(sim.randomDriveAction())
        replayBuffer.addData(simData[0],simData[1])
        print(replayBuffer.getRandBatch()[0][2])
        if simData[2]:
            sim.terrain.generate()
            sim.resetClifford()
    #replayBuffer.saveData()
    print("DONE")
    p.disconnect()
