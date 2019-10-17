import torch

#
# Helper functions for making standard networks
#
def shallow(in_dim,hidden,out_dim,Act=torch.nn.ReLU):
    """Just make a shallow network. This is more of a macro."""
    return torch.nn.Sequential(
            torch.nn.Linear(in_dim,hidden),
            Act(),
            torch.nn.Linear(hidden,out_dim),
        )
def deep(widths,Act=torch.nn.ReLU):
    """Make a deep FCMLP given width specifications. Degenerates to a shallow layer if len(widths)==3"""
    raise NotImplemented()
    
#
# Basic Classes
#
def ShallowNet(class.nn.Module):
    """Just a basic shallow network"""
    def __init__(self, in_dim, out_dim, hidden=10):
        super(ShallowNet,self).__init__()
        self.net = shallow(in_dim,hidden,out_dim)
    def forward(self,x):
        return self.net(x)
    
def ShallowSkipNet(class.nn.Module):
    """A basic shallow network with a skip connection"""
    def __init__(self, dim, hidden=10):
        super(ShallowSkipNet,self).__init__()
        self.net = shallow(dim,hidden,dim)
    def forward(self,x):
        return x+self.net(x)

#
# Networks for ODEs. A different call structure
#
def ShallowODE(class.nn.Module):
    def __init__(self, dim, hidden=10):
        super(ShallowODE,self).__init__()
        self.net = shallow(dim,hidden,dim)
    def forward(self,t,x):
        return self.net(x)
    