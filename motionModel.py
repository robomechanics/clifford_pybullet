import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from scipy.spatial.transform import Rotation as R

class deterministicMotionModel(nn.Module):
	def __init__(self, motionModelArgs,batchNormsOn = True):
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
		self.batchNormsOn = batchNormsOn
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
			if self.batchNormsOn:
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
		lowerTriangular = torch.zeros(means.shape[0],means.shape[1],means.shape[1],requires_grad=True).to(means.device)
		lowerTriangular[:,range(lowerTriangular.shape[1]),range(lowerTriangular.shape[2])] = torch.exp(logLDiags)#LDiags
		offDiagIndices = torch.ones(lowerTriangular.shape[1:3]).tril(diagonal=-1).nonzero()
		lowerTriangular[:,offDiagIndices[:,0],offDiagIndices[:,1]] = LOffDiags
		meanDiff = groundTruths-means
		temp = torch.matmul(torch.transpose(lowerTriangular,1,2),meanDiff.unsqueeze(2))
		logLikelihoods = means.shape[1]*np.log(2.*np.pi)/2.-halfLogDet + torch.matmul(torch.transpose(temp,1,2),temp).squeeze()/2.
		return torch.mean(logLikelihoods)
	def logMeanLikelihoods(self,groundTruths,distributions,numParticles):
		means = distributions[0]
		logLDiags = distributions[1]
		LOffDiags = distributions[2]
		lowerTriangular = torch.zeros(means.shape[0],means.shape[1],means.shape[1],requires_grad=True).to(means.device)
		lowerTriangular[:,range(lowerTriangular.shape[1]),range(lowerTriangular.shape[2])] = torch.exp(logLDiags)#LDiags
		offDiagIndices = torch.ones(lowerTriangular.shape[1:3]).tril(diagonal=-1).nonzero()
		lowerTriangular[:,offDiagIndices[:,0],offDiagIndices[:,1]] = LOffDiags
		logDet = logLDiags.sum(dim=1)
		meanDiff = (groundTruths-means).unsqueeze(2)
		#likelihoods = ((logDet - meanDiff.transpose(1,2).matmul(lowerTriangular).matmul(lowerTriangular.transpose(1,2)).matmul(meanDiff).squeeze())/2.).exp()
		individualLogLikelihoods = ((logDet - meanDiff.transpose(1,2).matmul(lowerTriangular).matmul(lowerTriangular.transpose(1,2)).matmul(meanDiff).squeeze())/2.).reshape(-1,numParticles)
		# log(exp(a)+exp(b)) = log([exp(a-c)+exp(b-c)]*exp(c)) = c + log(exp(a-c)+exp(b-c))
		maxIndividualLogLikelihoods = individualLogLikelihoods.max(dim=1)[0]
		logLikelihoods = maxIndividualLogLikelihoods + (individualLogLikelihoods - maxIndividualLogLikelihoods.unsqueeze(1).repeat_interleave(numParticles,dim=1)).exp().mean(dim=1).log()
		#meanIndividualLogLikelihoods = individualLogLikelihoods.mean(dim=1)
		#logLikelihoods = meanIndividualLogLikelihoods + (individualLogLikelihoods-meanIndividualLogLikelihoods.unsqueeze(1).repeat_interleave(numParticles,dim=1)).exp().mean(dim=1).log()
		return logLikelihoods
	def sampleFromDist(self,distributions):
		means = distributions[0]
		logLDiags = distributions[1]
		lOffDiags = distributions[2]
		Lmatrix = torch.zeros(means.shape[0],means.shape[1],means.shape[1],requires_grad=True).to(means.device)
		Lmatrix[:,range(Lmatrix.shape[1]),range(Lmatrix.shape[2])] = torch.exp(logLDiags)
		offDiagIndices = torch.ones(Lmatrix.shape[1:3]).tril(diagonal=-1).nonzero()
		Lmatrix[:,offDiagIndices[:,0],offDiagIndices[:,1]] = lOffDiags
		u = torch.normal(torch.zeros_like(means),torch.ones_like(means))
		samples = Lmatrix.transpose(1,2).inverse().matmul(u.unsqueeze(2)).squeeze() + means
		return samples

class probabilisticMotionModelV2(nn.Module):
	def __init__(self, motionModelArgs):
		super(probabilisticMotionModelV2, self).__init__()
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
		#logLDiags = torch.abs(self.fcDiagOutput(connected))+0.0001
		logLDiags = self.fcDiagOutput(connected)
		lOffDiags = self.fcOffDiagOutput(connected)
		return (means,logLDiags,lOffDiags)
	def sampleFromDist(self,distributions):
		means = distributions[0]
		logLDiags = distributions[1]
		lOffDiags = distributions[2]
		Lmatrix = torch.zeros(means.shape[0],means.shape[1],means.shape[1],requires_grad=True).to(means.device)
		Lmatrix[:,range(Lmatrix.shape[1]),range(Lmatrix.shape[2])] = torch.exp(logLDiags)
		offDiagIndices = torch.ones(Lmatrix.shape[1:3]).tril(diagonal=-1).nonzero()
		Lmatrix[:,offDiagIndices[:,0],offDiagIndices[:,1]] = lOffDiags
		u = torch.normal(torch.zeros_like(means),torch.ones_like(means))
		samples = torch.matmul(Lmatrix,u.unsqueeze(2)).squeeze() + means
		#print(torch.matmul(Lmatrix,torch.transpose(Lmatrix,1,2)))
		return samples
	def getLogLoss(self,samples,groundTruths):
		numSamples = samples.shape[0]
		batchSize = groundTruths.shape[0]
		vecLength = groundTruths.shape[1]
		numParticles = int(numSamples/batchSize)
		sampleReshaped = samples.reshape(batchSize,numParticles,vecLength)
		means = torch.mean(sampleReshaped,dim=1)
		sampleDiffFromMean = sampleReshaped - torch.cat(sampleReshaped.shape[1]*[means.unsqueeze(1)],dim=1)
		sampleDiffFromMean = sampleDiffFromMean.reshape(numSamples,vecLength,1)
		covariances = torch.matmul(sampleDiffFromMean,torch.transpose(sampleDiffFromMean,1,2))
		covariances = covariances.reshape(batchSize,numParticles,vecLength,vecLength)
		covariances = torch.sum(covariances,dim=1)/(numParticles-1.)
		diffFromMeanGT = (groundTruths - means).unsqueeze(2)
		loglosses = torch.logdet(covariances) + torch.matmul(torch.transpose(diffFromMeanGT,1,2),torch.matmul(torch.inverse(covariances),diffFromMeanGT)).squeeze()
		return loglosses

class simpleMotionModel(nn.Module):
	def __init__(self):
		super(simpleMotionModel, self).__init__()
		self.wheelBase = torch.nn.Parameter(torch.tensor(0.9, dtype=torch.float32, requires_grad=True)) #estimated wheelbase
		self.register_parameter("wheelBase" , self.wheelBase )
		self.driveScale = torch.nn.Parameter(torch.tensor(0.0096,dtype=torch.float32,requires_grad=True)) #estimated driving scale
		self.register_parameter("driveScale" , self.driveScale )
	def forward(self,data):
		throttle = data[0][:,0]
		steering = data[0][:,1]
		turnRadius = -self.wheelBase/torch.tan(steering)
		drivingDistance = self.driveScale*throttle
		estimatedHeadingChange = drivingDistance/turnRadius
		estimatedXChange = turnRadius*torch.sin(estimatedHeadingChange)
		estimatedYchange = turnRadius*(1.-torch.cos(estimatedHeadingChange))
		estimatedHeadingChange = estimatedHeadingChange.clone()
		estimatedXChange = estimatedXChange.clone()
		estimatedYchange = estimatedYchange.clone()
		estimatedHeadingChange[steering==0] = 0
		estimatedXChange[steering==0] = drivingDistance[steering==0]
		estimatedYchange[steering==0] = 0
		estimatedXChange = estimatedXChange + self.wheelBase/2.*torch.cos(estimatedHeadingChange) - self.wheelBase/2.
		estimatedYchange = estimatedYchange + self.wheelBase/2.*torch.sin(estimatedHeadingChange)
		return torch.cat((estimatedXChange.unsqueeze(1),estimatedYchange.unsqueeze(1),estimatedHeadingChange.unsqueeze(1)),axis=1)






























