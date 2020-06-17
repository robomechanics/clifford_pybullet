import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R

class deterministicMotionModel(nn.Module):
	def __init__(self, motionModelArgs):
		super(deterministicMotionModel, self).__init__()
		# set size of neural network input
		argDim = motionModelArgs[0]
		self.inStateDim = argDim[0]
		self.inMapDim = argDim[1]
		self.inActionDim = argDim[2]
		self.outStateDim = argDim[3]
		# set sizes of neural networks
		nnSize = motionModelArgs[1]
		convSize = nnSize[0]
		fcSize = nnSize[1]
		#set up conv networks
		self.convs = nn.ModuleList([])
		lastDim = [1,self.inMapDim]
		for i in range(len(convSize)):
			self.convs.append(nn.Conv2d(lastDim[0],convSize[i][0],convSize[i][1]))
			lastDim = [convSize[i][0],lastDim[1]-convSize[i][1]+1]
		self.convOutputDim = lastDim[0]*lastDim[1]*lastDim[1]
		#set up FC networks
		lastDim = self.convOutputDim+self.inStateDim+self.inActionDim # size of input to first fully connected neural network
		self.fcs = nn.ModuleList([])
		self.batchNorms = nn.ModuleList([])
		for i in range(len(fcSize)):
			self.fcs.append(nn.Linear(lastDim,fcSize[i]))
			self.batchNorms.append(nn.BatchNorm1d(lastDim))
			lastDim = fcSize[i]
		self.fcOutput = nn.Linear(fcSize[-1],self.outStateDim)
		dropout_p = motionModelArgs[2]
		self.dropout = nn.Dropout(dropout_p)
	def forward(self,data):
		rState = data[0]
		rMap = data[1]
		rAction = data[2]
		for i in range(len(self.convs)):
			rMap = self.convs[i](rMap)
			rMap = F.leaky_relu(rMap)
		rMap = rMap.view(-1,self.convOutputDim)
		connected = torch.cat((rMap,rState,rAction),axis=1)
		for i in range(len(self.fcs)):
			connected = self.batchNorms[i](connected)
			connected = self.dropout(connected)
			connected = self.fcs[i](connected)
			connected = F.leaky_relu(connected)
		connected = self.dropout(connected)
		output = self.fcOutput(connected)
		return output