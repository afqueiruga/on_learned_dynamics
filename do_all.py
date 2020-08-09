from collections import defaultdict
from functools import lru_cache
from itertools import cycle
from typing import Any, Dict, List, NamedTuple

import numpy as np
from matplotlib import pylab as plt
from matplotlib import cm

import torchdiffeq

import utils
import plotting
import helper
import models
import ode_helper

from helper import *
from utils import *
from analytical_solutions import *

plt.style.use('seaborn-paper')

# N_DATA = 1000
T_MAX_TO_TRAIN = 200.0
# N_ITER = 1000  # To debug.
# N_ITER = 10000  # Looks sufficient.
# N_ITER = 25000  # From paper.
N_ITER = 50000  # Push it harder.
N_SAMPLE = 5
DTS_TO_TRAIN = [ 0.4, 0.2, 0.1, 0.02, 0.01, 0.002, 0.001 ]
N_DATA_TO_TRAIN = [ int(T_MAX_TO_TRAIN//dt)+1 for dt in DTS_TO_TRAIN ]
SCHEMES = ['euler', 'midpoint', 'rk4']

T_MAX_TO_INFER = 2.0
HS_TO_INFER = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02,0.05,0.1,0.2,0.5,0.6,0.9,1.0,]
#1.1,1.5,2.0,5.0,7.5,10.0,12,15,20]
DTS_TO_INFER = [ 1.0 ]
for tnext in DTS_TO_TRAIN + [0.0001]:
    DTS_TO_INFER += list(np.logspace(np.log10(DTS_TO_INFER[-1]), np.log10(tnext), 5, base=10)[1:])
N_DATA_TO_INFER = [ int(T_MAX_TO_INFER//dt)+1 for dt in DTS_TO_INFER ]

THETA_0 = np.array([[1.5*np.pi/2, 0]])
THETA_TRUE_MAX = solution_pendulum_theta(np.array([T_MAX_TO_INFER,]), 1.5*np.pi/2)[0]
device = get_device()


class PendulumForceTheta(torch.nn.Module):
    """The true dynamics, as a torch.nn."""
    def __init__(self):
        super(PendulumForceTheta,self).__init__()
    def forward(self, t, State):
        theta, v = State[:,0], State[:,1]
        g = -9.81
        return torch.stack([
            v,
            g*torch.sin(theta),
        ],dim=-1)


@torch.no_grad()
def compute_errors(model, scheme):
    """Compute the errors for one torch.nn based model."""
    errors = {}
    for N_data in N_DATA_TO_INFER:
        ts = np.linspace(0, T_MAX_TO_INFER, N_data)
        pred = torchdiffeq.odeint(model, data_to_torch(THETA_0, device=device), data_to_torch(ts, device=device), method=scheme)
        errors[N_data] = np.linalg.norm(pred[-1].detach().cpu().numpy() - THETA_TRUE_MAX)
    return errors


def make_baseline_numerical_solutions():
    """Compute the errors using the ground truth."""
    FTrue = PendulumForceTheta()
    baseline = {}
    for scheme in SCHEMES:
        baseline[scheme] = compute_errors(FTrue, scheme)
    return baseline


class Result(NamedTuple):
    """Container struct to hold a trained G model and its errors."""
    model: Any
    training_scheme: str
    dt_data: float
    losses: List[Any]
    errors: Dict [float,float]

        
def make_dataset_for(dt_data=None, N_data=None, t_max=None):
    """Create an instance of the ground truth dataset.
    
    We impose dt_data, but only either N_data or t_max can be given."""
    if sum([1 if _ is not None else 0 for _ in (dt_data, N_data, t_max)])!=2:
        raise RuntimeError("Need to specify exactly two arguments.")
    if N_data is None:
        N_data = int(t_max // dt)+1
    if t_max is None:
        t_max = (N_data-1) * dt_data
    if dt_data is None:
        dt_data = t_max / (N_data-1)
    anal_ts = np.linspace(0,t_max,N_data)
    data = solution_pendulum_theta(anal_ts, 1.5*np.pi/2)
    return anal_ts, data


def infer_and_plot_model(model, model_ts, data, data_ts=None, method='rk4', idcs=0):
    """Make a plot from inference."""
    if data_ts is None:
        data_ts = model_ts
    with torch.no_grad():
        pred = torchdiffeq.odeint(model, data[0,:], model_ts, method=method)
    plt.plot(data_ts.cpu().numpy(), data.detach().cpu().numpy()[:,idcs], '--')
    plt.plot(model_ts.cpu().numpy(), pred.detach().cpu().numpy()[:,idcs],'x-')



#@lru_cache(maxsize=256)
def train_a_model(scheme, dt_data=None, N_data=None):
    """Do one instance of a NN embedded inside of scheme with dt_data sampling.
    
    Perform one datum of the experiment by training a new network and performing
    the convergence test.
    """
    if dt_data is not None: 
        ts, thetas = make_dataset_for(dt_data=dt_data, N_data=N_DATA)
    else:
        ts, thetas = make_dataset_for(N_data=N_data, t_max=T_MAX_TO_TRAIN)
        dt_data = T_MAX_TO_TRAIN/(N_data+1)
    t_data = data_to_torch(ts, device)
    theta_data = data_to_torch(thetas, device)
    model = models.ShallowODE(2,50, Act=torch.nn.Tanh).double().to(device)
    _, losses = ode_helper.train_a_neural_ode_multi_method(
                      theta_data, t_data, model, methods=(scheme,), N_iter=N_ITER, batch_size=128)
    plt.semilogy(losses)
    errors = compute_errors(model, scheme)
    return Result(model, scheme, dt_data, losses, errors)


def train_all_models_for_fixed_data():
    """Loop over all of the models."""
    all_results = { s:[] for s in SCHEMES }
    for scheme in SCHEMES:
        for dt_data in DTS_TO_TRAIN:
            print(f"Training for {scheme} at dt={dt_data}.")
            all_results[scheme].append(train_a_model(scheme, dt_data=dt_data))
    return all_results

def train_ndata(scheme, N_data):
    return train_a_model(scheme, N_data=N_data)

def train_all_models_for_infinite_data():
    """Loop over all of the models."""
    all_results = { s:[] for s in SCHEMES }
    for scheme in SCHEMES:
        for N_data in N_DATA_TO_TRAIN:
            for sample in range(N_SAMPLE):
                print(f"Training for {scheme} at dt={T_MAX_TO_TRAIN/(N_data-1)}.")
                dt_data = T_MAX_TO_TRAIN / (N_data-1)
                all_results[scheme].append( train_a_model(scheme, N_data=N_data) )
    return all_results

