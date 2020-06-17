import pybullet as p
import time
import pybullet_data
import numpy as np
from scipy.ndimage import gaussian_filter
from noise import pnoise1,pnoise2
from scipy.interpolate import interp2d
from scipy.interpolate import griddata

class RandomRockyTerrain:
  def __init__(self,mapWidth,mapHeight,widthScale,heightScale,physicsClientId=0):
    self.mapWidth = mapWidth
    self.mapHeight = mapHeight
    self.meshScale = [widthScale,heightScale,1]
    self.mapSize = [(mapWidth-1)*widthScale,(mapHeight-1)*heightScale]
    self.gridX = np.linspace(-(mapWidth-1)/2,(mapWidth-1)/2,mapWidth)*widthScale
    self.gridY = np.linspace(-(mapHeight-1)/2,(mapHeight-1)/2,mapHeight)*heightScale
    self.gridX,self.gridY = np.meshgrid(self.gridX,self.gridY)
    self.replaceID = []
    self.physicsClientId = physicsClientId
    self.terrainBody = []
  def generate(self,AverageAreaPerCell = 2.5,cellPerlinScale=0.5,cellHeightScale=0.75,smoothing=1,perlinScale=5,perlinHeightScale=0.05):
    # generate random blocks
    numCells = int(float(self.mapSize[0])*float(self.mapSize[1])/float(AverageAreaPerCell))
    blockHeights = self.randomSteps(self.gridX.reshape(-1),self.gridY.reshape(-1),numCells,cellPerlinScale,cellHeightScale)
    blockHeights = gaussian_filter(blockHeights.reshape(self.gridX.shape), sigma=smoothing)
    # add more small noise
    smallNoise = self.perlinNoise(self.gridX.reshape(-1),self.gridY.reshape(-1),perlinScale,perlinHeightScale)
    smallNoise = smallNoise.reshape(self.gridX.shape)
    self.gridZ = blockHeights+smallNoise
    self.gridZ = self.gridZ-np.min(self.gridZ)
    if isinstance(self.replaceID,int):
      terrainShape = p.createCollisionShape(heightfieldTextureScaling=0.001,shapeType = p.GEOM_HEIGHTFIELD,meshScale = self.meshScale, heightfieldData=self.gridZ.reshape(-1), numHeightfieldRows=self.mapWidth, numHeightfieldColumns=self.mapHeight,replaceHeightfieldIndex=self.replaceID,physicsClientId=self.physicsClientId)
    else:
      terrainShape = p.createCollisionShape(heightfieldTextureScaling=0.001,shapeType = p.GEOM_HEIGHTFIELD,meshScale = self.meshScale, heightfieldData=self.gridZ.reshape(-1), numHeightfieldRows=self.mapWidth, numHeightfieldColumns=self.mapHeight,physicsClientId=self.physicsClientId)
      self.terrainBody  = p.createMultiBody(0, terrainShape,physicsClientId=self.physicsClientId)
      p.changeVisualShape(self.terrainBody, -1, rgbaColor=[0.82,0.71,0.55,1],physicsClientId=self.physicsClientId)
    self.replaceID = terrainShape
    offset = np.max(self.gridZ)/2.
    p.resetBasePositionAndOrientation(self.terrainBody,[0,0,offset], [0,0,0,1],physicsClientId=self.physicsClientId)
  def randomSteps(self,xPoints,yPoints,numCells,cellPerlinScale,cellHeightScale):
    centersX = np.random.uniform(size=numCells,low=np.min(xPoints),high=np.max(xPoints))
    centersY = np.random.uniform(size=numCells,low=np.min(yPoints),high=np.max(yPoints))
    centersZ = self.perlinNoise(centersX,centersY,cellPerlinScale,cellHeightScale)
    xPointsMatrix = np.matmul(np.matrix(xPoints).transpose(),np.ones((1,numCells)))
    yPointsMatrix = np.matmul(np.matrix(yPoints).transpose(),np.ones((1,numCells)))
    centersXMatrix = np.matmul(np.matrix(centersX).transpose(),np.ones((1,len(xPoints)))).transpose()
    centersYMatrix = np.matmul(np.matrix(centersY).transpose(),np.ones((1,len(yPoints)))).transpose()
    xDiff = xPointsMatrix - centersXMatrix
    yDiff = yPointsMatrix - centersYMatrix
    distMatrix = np.multiply(xDiff,xDiff)+np.multiply(yDiff,yDiff)
    correspondingCell = np.argmin(distMatrix,axis=1)
    return centersZ[correspondingCell]
  def perlinNoise(self,xPoints,yPoints,perlinScale,heightScale):
    randomSeed = np.random.rand(2)*1000
    return np.array([pnoise2(randomSeed[0]+xPoints[i]*perlinScale,randomSeed[1]+yPoints[i]*perlinScale) for i in range(len(xPoints))])*heightScale
  def robotHeightMap(self,position,heading,mapWidth,mapHeight,mapScale):
    maxRadius = np.sqrt((mapWidth-1)**2+(mapHeight-1)**2)*mapScale/2.
    vecX = self.gridX.reshape(-1)-position[0]
    vecY = self.gridY.reshape(-1)-position[1]
    indices = np.all(np.stack((np.abs(vecX)<=(maxRadius+self.meshScale[0]),np.abs(vecY)<=(maxRadius+self.meshScale[1]))),axis=0)
    vecX = vecX[indices]
    vecY = vecY[indices]
    vecZ = self.gridZ.reshape(-1)[indices]
    relativeX = vecX*np.cos(heading)+vecY*np.sin(heading)
    relativeY = -vecX*np.sin(heading)+vecY*np.cos(heading)
    rMapX = np.linspace(-(mapWidth-1)/2.,(mapWidth-1)/2.,mapWidth)*mapScale
    rMapY = np.linspace((mapHeight-1)/2.,-(mapHeight-1)/2.,mapHeight)*mapScale
    points = np.stack((relativeX,relativeY)).transpose()
    rMapX,rMapY = np.meshgrid(rMapX,rMapY)
    return griddata(points, vecZ, (rMapX,rMapY))-position[2]
  def safeFallHeight(self,position):
    vecX = self.gridX.reshape(-1)-position[0]
    vecY = self.gridY.reshape(-1)-position[1]
    indices = vecX*vecX+vecY*vecY<1
    vecZ = self.gridZ.reshape(-1)[indices]
    return np.expand_dims(np.max(vecZ)+0.18, axis=0)

if __name__=="__main__":
  physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
  p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
  p.setGravity(0,0,-10)
  mapWidth = 300
  mapHeight = 300
  terrain = RandomRockyTerrain(mapWidth,mapHeight,0.1,0.1)
  terrain.generate()
  time.sleep(10)
  terrain.generate()
  for i in range (10000):
      p.stepSimulation()
      time.sleep(1./240.)
  time.sleep(10)
  print("done")
  p.disconnect()
