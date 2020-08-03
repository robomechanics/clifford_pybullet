import numpy as np
from scipy.ndimage import gaussian_filter
from noise import pnoise1,pnoise2
from scipy.interpolate import griddata

class terrain:
  def __init__(self,mapWidth,mapHeight,widthScale,heightScale):
    self.mapWidth = mapWidth
    self.mapHeight = mapHeight
    self.meshScale = [widthScale,heightScale,1]
    self.gridX = np.linspace(-(mapWidth-1)/2,(mapWidth-1)/2,mapWidth)*widthScale
    self.gridY = np.linspace(-(mapHeight-1)/2,(mapHeight-1)/2,mapHeight)*heightScale
    self.gridX,self.gridY = np.meshgrid(self.gridX,self.gridY)
  def loadTerrain(self,gridZFile):
    self.gridZ = np.load(gridZFile)
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

if __name__=="__main__":
  ter = terrain(1,2,3,4)