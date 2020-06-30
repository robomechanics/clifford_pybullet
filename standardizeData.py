import torch
import pybullet as p
class standardizeData(object):
    def __init__(self,device = 'cpu'):
        self.device = device
    def getDistribution(self,data):
        self.mean = data.mean(dim=0).to(self.device)
        self.var = data.std(dim=0).to(self.device)
    def whiten(self,data):
        output = (data.to(self.device)-self.mean)/self.var
        return output
    def raw(self,data):
        output = (data.to(self.device)*self.var)+self.mean
        return output


class cliffordStateTransformation(object):
    def __init__(self,startState,device = ''):
        if len(device)==0:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.currentState = torch.FloatTensor(startState).unsqueeze(0).to(device)
    def moveState(self,nnOutput):
        relativePos = nnOutput[:,0:3]
        relativeOrien = nnOutput[:,3:7]
        newTwist = nnOutput[:,7:13]
        newJointState = nnOutput[:,13:27]
        newPosition,newOrientation = self.transformMul(self.currentState[:,0:3],self.currentState[:,3:7],relativePos,relativeOrien)
        self.currentState = torch.cat((newPosition,newOrientation,newTwist,newJointState),dim=1)
        return self.currentState
    def stateToNNInState(self):
        Rwb = self.currentState[:,3:7]
        Rbw = self.qinv(Rwb)
        upDirWorld = torch.zeros(self.currentState.shape[0],3).to(self.currentState.device)
        upDirWorld[:,2] = 1.
        upDirRobot = self.qrot(Rbw,upDirWorld)
        tiltAngles = torch.cat((torch.acos(upDirRobot[:,2:3]),torch.atan2(upDirRobot[:,1:2],upDirRobot[:,0:1])),dim=1)
        twist = self.currentState[:,7:13]
        jointState = self.currentState[:,13:27]
        inState = torch.cat((tiltAngles,twist,jointState),dim=1)
        return inState
    def transformMul(self,p1,q1,p2,q2):
        qout = self.qmul(q1,q2)
        pout = p1+self.qrot(q1,p2)
        return (pout,qout)
    def qmul(self,q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4
        
        original_shape = q.shape
        
        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

        x = terms[:, 3, 0] + terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1]
        y = terms[:, 3, 1] + terms[:, 0, 2] + terms[:, 1, 3] - terms[:, 2, 0]
        z = terms[:, 3, 2] - terms[:, 0, 1] + terms[:, 1, 0] + terms[:, 2, 3]
        w = terms[:, 3, 3] - terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2]
        return torch.stack((x, y, z, w), dim=1).view(original_shape)
    def qrot(self,q, v):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]
        
        original_shape = list(v.shape)
        q = q.view(-1, 4)
        v = v.view(-1, 3)
        
        qvec = q[:, 0:3]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, 3:] * uv + uuv)).view(original_shape)
    def qinv(self,q):
        assert q.shape[-1] == 4
        qout = torch.cat((-q[:,0:3],q[:,3:4]),dim=1)/(q**2).sum(dim=1)
        return(qout)











