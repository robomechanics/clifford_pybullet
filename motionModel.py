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
		dropout_ps = motionModelArgs[2]
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
		self.dropouts = nn.ModuleList([])
		self.batchNorms = nn.ModuleList([])
		for i in range(len(fcSize)):
			self.fcs.append(nn.Linear(lastDim,fcSize[i]))
			self.batchNorms.append(nn.BatchNorm1d(lastDim))
			self.dropouts.append(nn.Dropout(dropout_ps[i]))
			lastDim = fcSize[i]
		self.fcOutput = nn.Linear(fcSize[-1],self.outStateDim)
		#dropout_ps = motionModelArgs[2]
		#self.dropout = nn.Dropout(dropout_ps[0])
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
			connected = self.dropouts[i](connected)
			connected = self.fcs[i](connected)
			connected = F.leaky_relu(connected)
		output = self.fcOutput(connected)
		return output

class probabilisticMotionModel(nn.Module):
	def __init__(self, motionModelArgs):
		super(probabilisticMotionModel, self).__init__()
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
		dropout_ps = motionModelArgs[2]
		#set up conv networks
		self.convs = nn.ModuleList([])
		self.convBatchNorms = nn.ModuleList([])
		lastDim = [1,self.inMapDim]
		for i in range(len(convSize)):
			self.convBatchNorms.append(nn.BatchNorm2d(lastDim[0]))
			self.convs.append(nn.Conv2d(lastDim[0],convSize[i][0],convSize[i][1]))
			lastDim = [convSize[i][0],lastDim[1]-convSize[i][1]+1]
		self.convOutputDim = lastDim[0]*lastDim[1]*lastDim[1]
		#set up FC networks
		lastDim = self.convOutputDim+self.inStateDim+self.inActionDim # size of input to first fully connected neural network
		self.fcs = nn.ModuleList([])
		self.dropouts = nn.ModuleList([])
		self.fcBatchNorms = nn.ModuleList([])
		for i in range(len(fcSize)):
			self.fcs.append(nn.Linear(lastDim,fcSize[i]))
			self.fcBatchNorms.append(nn.BatchNorm1d(lastDim))
			self.dropouts.append(nn.Dropout(dropout_ps[i]))
			lastDim = fcSize[i]
		self.fcMeanOutput = nn.Linear(fcSize[-1],self.outStateDim)
		self.fcDiagOutput = nn.Linear(fcSize[-1],self.outStateDim)
		self.fcOffDiagOutput = nn.Linear(fcSize[-1],int((self.outStateDim)*(self.outStateDim-1)/2))
		self.meanDiffBatchNorm = nn.BatchNorm1d(self.outStateDim)
	def forward(self,data):
		rState = data[0]
		rMap = data[1]
		rAction = data[2]
		for i in range(len(self.convs)):
			rMap = self.convBatchNorms[i](rMap)
			rMap = self.convs[i](rMap)
			rMap = F.leaky_relu(rMap)
		rMap = rMap.view(-1,self.convOutputDim)
		connected = torch.cat((rMap,rState,rAction),axis=1)
		for i in range(len(self.fcs)):
			connected = self.fcBatchNorms[i](connected)
			connected = self.dropouts[i](connected)
			connected = self.fcs[i](connected)
			connected = F.leaky_relu(connected)
		means = self.fcMeanOutput(connected)
		#lDiags = torch.abs(self.fcDiagOutput(connected))
		logLDiags = self.fcDiagOutput(connected)
		LOffDiags = self.fcOffDiagOutput(connected)
		return (means,logLDiags,LOffDiags)
	def logLikelihood(self,groundTruths,distributions,isEval = False):
		means = distributions[0]
		logLDiags = distributions[1]
		LOffDiags = distributions[2]
		#halfLogDet = torch.sum(torch.log(LDiags),1)
		halfLogDet = torch.sum(logLDiags)
		lowerTriangular = torch.zeros(means.shape[0],means.shape[1],means.shape[1]).to(means.device)
		lowerTriangular[:,range(lowerTriangular.shape[1]),range(lowerTriangular.shape[2])] = torch.exp(logLDiags)#LDiags
		offDiagIndices = torch.ones(lowerTriangular.shape[1:3]).tril(diagonal=-1).nonzero()
		lowerTriangular[:,offDiagIndices[:,0],offDiagIndices[:,1]] = LOffDiags
		meanDiff = groundTruths-means
		temp = torch.matmul(torch.transpose(lowerTriangular,1,2),meanDiff.unsqueeze(2))
		logLikelihoods = means.shape[1]*np.log(2.*np.pi)/2.-halfLogDet + torch.matmul(torch.transpose(temp,1,2),temp).squeeze()/2.
		return torch.mean(logLikelihoods)

class simpleMotionModel(nn.Module):
	def __init__(self):
		super(simpleMotionModel, self).__init__()
		self.p1 = 0.75 #estimated wheelbase
		self.p2 = 1 #estimated scale
	def forward(self,data):
		throttle = data[0][:,0]
		steering = data[0][:,1]
		turnRadius = self.p1*torch.tan(steering)
		drivingDistance = self.p2*throttle
		estimatedHeadingChange = -drivingDistance/turnRadius
		estimatedXChange = -turnRadius*torch.sin(estimatedHeadingChange)
		estimatedYchange = turnRadius*(1.-torch.cos(estimatedHeadingChange))


























