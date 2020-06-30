import torch
class cliffordStateTransformation(object):
    def __init__(self,startState):
        #if len(device)==0:
        #    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        #self.currentState = torch.FloatTensor(startState).unsqueeze(0).to(device)
        self.currentState = startState
        orientation = startState[:,3:7]
    def moveState(self,nnOutput):
        relativePos = nnOutput[:,0:3]
        relativeOrien = nnOutput[:,3:7]
        relativeOrien = relativeOrien/torch.sqrt((relativeOrien**2).sum(dim=1)).unsqueeze(1)
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
        #tiltAngles = torch.cat((torch.acos(upDirRobot[:,2:3]),torch.atan2(upDirRobot[:,1:2],upDirRobot[:,0:1])),dim=1)
        #if torch.isnan(torch.sum(tiltAngles)).item():
        #    print("bad tilt angle")
        #    stop
        twist = self.currentState[:,7:13]
        jointState = self.currentState[:,13:27]
        #inState = torch.cat((tiltAngles,twist,jointState),dim=1)
        inState = torch.cat((upDirRobot,twist,jointState),dim=1)
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
        
        #q = q/torch.sqrt((q**2).sum(dim=1)).unsqueeze(1)
        #r = r/torch.sqrt((r**2).sum(dim=1)).unsqueeze(1)
        #print("qNorm" + str(torch.sqrt((q**2).sum(dim=1))[0]))
        #print("rNorm" + str(torch.sqrt((r**2).sum(dim=1))[0]))

        original_shape = q.shape
        
        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

        x = terms[:, 3, 0] + terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1]
        y = terms[:, 3, 1] + terms[:, 0, 2] + terms[:, 1, 3] - terms[:, 2, 0]
        z = terms[:, 3, 2] - terms[:, 0, 1] + terms[:, 1, 0] + terms[:, 2, 3]
        w = terms[:, 3, 3] - terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2]
        qout = torch.stack((x, y, z, w), dim=1).view(original_shape)
        #qout = qout/torch.sqrt((qout**2).sum(dim=1)).unsqueeze(1)
        #print("qout Norm" + str(torch.sqrt((qout**2).sum(dim=1))[0]))
        return qout
    def qrot(self,q, v):
        qinv_ = self.qinv(q)
        p = torch.cat((v,torch.zeros((v.shape[0],1),device=v.device,dtype=v.dtype)),dim=1)
        p_prime = self.qmul(self.qmul(q,p),qinv_)
        return p_prime[:,:3]
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
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
        return (v + 2 * (q[:, 3:] * uv + uuv)).view(original_shape)"""
    def qinv(self,q):
        assert q.shape[-1] == 4
        if torch.isnan(torch.sum(q)).item():
            "print(bad qinv input)"
        if (torch.mean((q**2).sum(dim=1))) < 0.001:
            print("VERY SMALL quaternion_qinv")
            stop
        qout = torch.cat((-q[:,0:3],q[:,3:4]),dim=1)/(q**2).sum(dim=1).unsqueeze(1)
        if torch.isnan(torch.sum(qout)).item():
            "print(bad qinv output)"
        return(qout)

if __name__ == '__main__':
    test = cliffordStateTransformation([])
    quat = torch.FloatTensor([0,3,3,1]).unsqueeze(0)
    vec = torch.FloatTensor([1.,0.,1.]).unsqueeze(0)
    print(test.qrot(quat,vec))
    print(p.multiplyTransforms([0,0,0],[0,3,3,1],[0,0,1],[0,0,0,1]))