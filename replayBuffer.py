import torch
import numpy as np
from os import path

class ReplayBuffer(object):
	def __init__(self,bufferLength=0,sampleNNInput=[],sampleNNOutput=[],saveDataPrefix='',loadDataPrefix='',chooseCPU = False):
		if chooseCPU:
			self.device='cpu'
		else:
			self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.saveDataPrefix = saveDataPrefix
		self.loadDataPrefix = loadDataPrefix
		self.bufferLength = bufferLength
		self.bufferIndex = 0
		self.bufferFilled = False
		self.inputData = []
		self.outputData = []
		for data in sampleNNInput:
			self.inputData.append(torch.zeros((self.bufferLength,)+np.array(data).shape,device=self.device))
		for data in sampleNNOutput:
			self.outputData.append(torch.zeros((self.bufferLength,)+np.array(data).shape,device=self.device))
	def purgeData(self):
		self.bufferIndex = 0
		self.bufferFilled = False
	def addData(self,nnInput,nnOutputGroundTruth):
		inputData,outputData = self.processData(nnInput,nnOutputGroundTruth)
		for i in range(len(inputData)):
			self.inputData[i][self.bufferIndex,:] = inputData[i]
		for i in range(len(outputData)):
			self.outputData[i][self.bufferIndex,:] = outputData[i]
		self.bufferIndex+=1
		if self.bufferIndex == self.bufferLength:
			self.bufferIndex = 0
			self.bufferFilled = True
	def processData(self,nnInput,nnOutputGroundTruth):
		inputData = []
		outputData = []
		for data in nnInput:
			inputData.append(torch.from_numpy(np.array(data)).to(self.device).unsqueeze(0).float())
		for data in nnOutputGroundTruth:
			outputData.append(torch.from_numpy(np.array(data)).to(self.device).unsqueeze(0).float())
		return inputData,outputData
	def getData(self,batchSize=1):
		returnedData=[]
		index = 0
		maxIndex = self.bufferLength if self.bufferFilled else self.bufferIndex
		while index < maxIndex:
			endIndex = np.min([index+batchSize,maxIndex])
			returnedData.append([[self.inputData[i][index:endIndex,:] for i in range(0,len(self.inputData))],
								[self.outputData[i][index:endIndex,:] for i in range(0,len(self.outputData))]])
			index = endIndex
		return returnedData
	def getRandBatch(self,batchSize=1,device=''):
		if len(device)==0:
			device = self.device
		maxIndex = self.bufferLength if self.bufferFilled else self.bufferIndex
		indices = np.random.choice(maxIndex,batchSize) if maxIndex > batchSize else np.arange(maxIndex)
		#indices = np.random.randint(0,maxIndex,batchSize)
		returnedData =[[self.inputData[i][indices,:].to(device) for i in range(0,len(self.inputData))],
						[self.outputData[i][indices,:].to(device) for i in range(0,len(self.outputData))]]
		return returnedData
	def saveData(self):
		for i in range(len(self.inputData)):
			torch.save(self.inputData[i],self.saveDataPrefix+"simInputData"+str(i)+".pt")
		for i in range(len(self.outputData)):
			torch.save(self.outputData[i],self.saveDataPrefix+"simOutputData"+str(i)+".pt")
	def loadData(self,matchLoadSize=False):
		if matchLoadSize:
			self.inputData = []
			self.outputData = []
			i = 0
			while path.exists(self.loadDataPrefix+"simInputData"+str(i)+".pt"):
				self.inputData.append(torch.load(self.loadDataPrefix+"simInputData"+str(i)+".pt").to(self.device))
				i+=1
			i = 0
			while path.exists(self.loadDataPrefix+"simOutputData"+str(i)+".pt"):
				self.outputData.append(torch.load(self.loadDataPrefix+"simOutputData"+str(i)+".pt").to(self.device))
				i+=1
			self.bufferIndex = self.inputData[0].shape[0]
			self.bufferFilled = True
			self.bufferLength = self.bufferIndex
		else:
			data = torch.load(self.loadDataPrefix+"simInputData0.pt").to(self.device)
			self.bufferIndex = np.min(self.bufferLength,data.shape[0])
			self.bufferFilled = False if self.bufferIndex<self.bufferLength else True
			for i in range(len(self.inputData)):
				self.inputData[i][0:self.bufferIndex,:] = torch.load(self.loadDataPrefix+"simInputData"+str(i)+".pt").to(self.device)[0:self.bufferIndex,:]
			for i in range(len(self.outputData)):
				self.outputData[i][0:self.bufferIndex,:] = torch.load(self.loadDataPrefix+"simOutputData"+str(i)+".pt").to(self.device)[0:self.bufferIndex,:]
			print("data loaded buffer filled: " + str(self.bufferFilled) + " buffer index: " + str(self.bufferIndex))
	def inheritData(self,otherBuffer):
		while len(self.inputData)>0:
			del self.inputData[0]
		while len(self.outputData)>0:
			del self.outputData[0]
		torch.cuda.empty_cache()
		data = otherBuffer.getRandBatch(self.bufferLength)
		self.inputData = [data[0][i].to(self.device) for i in range(len(data[0]))]
		self.outputData = [data[1][i].to(self.device) for i in range(len(data[1]))]
		self.bufferIndex = self.inputData[0].shape[0]
		if self.bufferIndex == self.bufferLength:
			self.bufferFilled = True

