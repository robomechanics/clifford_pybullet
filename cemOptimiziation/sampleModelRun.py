import sys
sys.path.append("..") 
from motionModel import simpleMotionModel
import matplotlib.pyplot as plt
import torch
import numpy as np

numParticles = 10
Poses = [torch.zeros(numParticles,3)] # 3 states (x,y,heading angle)

# set up motion model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
motionModel = simpleMotionModel().to(device)
motionModel.eval()
for i in range(20):
	# Robot action (throttle,steering) steering between -0.5 and 0.5
	# choose throttle of 10 and different steering
	actionTaken = torch.cat((torch.ones(numParticles,1)*50,0.5*torch.linspace(-0.5,0.5,numParticles).unsqueeze(1)),dim=1)
	prediction = motionModel([actionTaken]) # predicts pose of robot relative to last pose
	# translate last pose to new pose
	newX = Poses[-1][:,0] + prediction[:,0]*torch.cos(Poses[-1][:,2]) - prediction[:,1]*torch.sin(Poses[-1][:,2])
	newY = Poses[-1][:,1] + prediction[:,0]*torch.sin(Poses[-1][:,2]) + prediction[:,1]*torch.cos(Poses[-1][:,2])
	newHeading = Poses[-1][:,2] + prediction[:,2]
	# append new pose to end of list
	Poses.append(torch.cat((newX.unsqueeze(1),newY.unsqueeze(1),newHeading.unsqueeze(1)),dim=1))
plt.figure()
for i in range(numParticles):
	pathX = []
	pathY = []
	for j in range(len(Poses)):
		pathX.append(Poses[j][i,0].item())
		pathY.append(Poses[j][i,1].item())
	plt.plot(pathX,pathY)
plt.xlim((-10,10))
plt.ylim((-10,10))
plt.show()