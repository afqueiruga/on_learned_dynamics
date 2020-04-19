import numpy as np
from matplotlib import pylab as plt
import cmocean, seaborn
import plotly
import plotly.graph_objects as go

import utils, plotting, helper, models, ode_helper, \
        analytical_solutions
    
from utils import *
from helper import *
from ode_helper import *

class Experiment():
    """
    This is an awkward container class to store experiment
    results for the wave equation.
    """
    def __init__(self, data, ts, model_shape,
                learning_rate=None,gamma_L1=0,gamma_L2=0, batch_size=100, ode=False,methods=('euler',),
                 optimizer_key='adam'):
        # Save the data
        self.data = data
        self.ts = ts
        # Save the model interpretation and training parameters
        self.ode = ode
        self.methods=methods
        self.gamma_L1 = gamma_L1
        self.gamma_L2 = gamma_L2
        self.batch_size = batch_size
        self.optimizer_key = optimizer_key
        # Construct the models
        if ode:
            self.learning_rate = 1.0e-2 if learning_rate is None else learning_rate
            self.model = models.MultiLinearODE(model_shape,model_shape,bias=False).double().to(device)
        else:
            self.learning_rate = 5.0e-4 if learning_rate is None else learning_rate
            self.model = models.MultiLinear(model_shape,model_shape,bias=False).double().to(device)
        # This is the awkard state
        self.losses = np.array([])
        
        self.save_schedule = [2.0**-i for i in range(-1,100)]
        self.next_save = self.save_schedule.pop(0)
        self.weight_dump = []
        self.all_metrics = []
        
    def name(self):
        "Returns a string identifier for plot labels."
        pfx = ( "ode_"+"".join([s[0] for s in self.methods]) ) if self.ode else "mat"
        return f"""{pfx}_{self.gamma_L1:1.1e}_{self.gamma_L2:1.1e}_{self.learning_rate:1.1e}_{self.batch_size}"""

    def _callback(self, model, opt_iter, loss, do_it=False):
        if do_it or loss < self.next_save:
            Om = self.model.net.weight.detach().cpu().numpy().copy()
            self.weight_dump.append(Om)
            self.all_metrics.append((opt_iter,loss))
            print("Saving at ",opt_iter, " with loss ",loss)
            self.next_save = self.save_schedule.pop(0)
            
    def train(self,N_iter=10000):
        "Train up to N_iter total iterations."
        N_left = N_iter - len(self.losses)
        print(f"Doing {N_left} iterations on {self.name()}")
        callback = lambda m,o,l, do_it=False : self._callback(m,o,l, do_it=do_it)
        if N_left <= 1: return
        if not self.ode:
            _,losses = learn_rnn(self.data, self.model,
                N_iter=N_left, batch_size= self.batch_size,
                N_print=100, callback=callback,
                learning_rate=self.learning_rate, gamma_L1=self.gamma_L1, gamma_L2=self.gamma_L2,
                device=device)
        else:
            _,losses = train_a_neural_ode_multi_method(self.data, self.ts,
                self.model,
                N_iter=N_left, batch_size=self.batch_size,
                learning_rate=self.learning_rate, gamma_L1=self.gamma_L1, gamma_L2=self.gamma_L2,
                methods=self.methods,
                N_print=100,callback=callback,
                device=device)
        self.losses = np.append(self.losses,losses)
        
    def post(self):
        "Do interpretation of the trained model to extract dynamics/"
        dt = self.ts.cpu().numpy()[1]-self.ts.cpu().numpy()[0]
        if not self.ode:
            self.all_omegas = self.weight_dump.copy()
            self.np_omegas = np.array(self.all_omegas)
            self.UVs = [helper.integrate_matrix(self.np_omegas[i,:,:],
                                    self.data[0:1,:,:].cpu().numpy(), 25 )
                        for i in range(self.np_omegas.shape[0]) ]
            self.all_lambdas = [ lambda_of_omega(om,dt)
                                for om in self.all_omegas ]
            self.np_lambdas = np.array(self.all_lambdas)
        else:
            self.all_lambdas = self.weight_dump.copy()
            self.np_lambdas = np.array(self.all_lambdas)
            # TODO self.UVs = 
            self.all_omegas = [operator_factory(L, dt,method=self.methods[-1]) for L in self.all_lambdas]
            self.np_omegas = np.array(self.all_omegas)

    def make_animation(self,interval=1,transpose=False):
        img = self.np_omegas
        metrics = self.all_metrics
        UVs = self.UVs
        fig, ax = plt.subplots(figsize=(6,3))
        minmax = np.max(np.abs(img)) * 0.65
        tr = lambda x : x.T if transpose else x
        ax_left = plt.subplot(1,2,1)
        ax_left.set_title(f"Iteration #{metrics[0][0]}     Loss = {metrics[0][1]:1.2e}")
        canvas = plt.imshow(tr(img[0,:,:]), interpolation='none',
                            cmap=cmocean.cm.balance, 
                            #interpolation='bicubic', 
                            vmin=-minmax, vmax=minmax)
        plt.axis('off')
        plt.text(-3,5,'$\\frac{du}{dt}$',fontsize=16)
        plt.text(-3,15,'$\\frac{dv}{dt}$',fontsize=16)
        plt.text(5,21,'$u$')
        plt.text(15,21,'$v$')
        ax = plt.subplot(1,2,2)
        ax.set_ylim((-0.5, 0.5))
        ax.set_title("$u(x,t)$")
        xs = np.linspace(0,1,UVs[0].shape[-1])
        line, = ax.plot(xs,UVs[0][0,0,:], lw=2)
        plt.tight_layout()
        def animate(i):
            frame = i//UVs[0].shape[0]
            time = i%UVs[0].shape[0]
            ax_left.set_title(f"Iteration #{metrics[frame][0]}     Loss = {metrics[frame][1]:1.2e}")
            line.set_data(xs,UVs[frame][time,0,:])
            canvas.set_array(tr(img[frame,:,:]))
            return canvas,
        ani = animation.FuncAnimation(fig, animate, frames=img.shape[0]*UVs[0].shape[0],
                                      interval = interval)
        return ani



def plot_losses(stash):
    "Make a plotly figure of the loss functions"
    data = []
    for config in stash:
        key = config #tuple(config.items())
        exp = stash[key]
        try:
            data.append(go.Scatter(y=exp.losses[::20],name=exp.name()))
        except:
            print("Skipping ",config)
    return go.Figure(data=data,layout=dict(yaxis_type="log") )

def plot_omega_lambda(stash):
    "Makes a grid of all of the omegas and lambdas"
    cols = 2
    sp_y,sp_x = int(len(stash)/cols+0.5), 2*cols
    plt.figure(figsize=(4*sp_x,4*sp_y))
    for i,exp in enumerate(stash.values()):
        try:
            exp.post()
            plt.subplot(sp_y,sp_x,2*i+1)
            plt.title(exp.name())
            seaborn.heatmap(exp.all_lambdas[-1], cmap='RdBu', center=0,vmin=-200,vmax=200)
            plt.subplot(sp_y,sp_x,2*i+2)
            seaborn.heatmap(exp.all_omegas[-1], cmap='RdBu', center=0, vmin=-2,vmax=2)
        except Exception as e:
            print(e)
            print("Skipping ",exp.name())
    plt.show()


def DT_Run(t_max, model_params, N_iter=5000, N_data_points=500, save_root=None):
    """DEPRECATED Do a set of experiments for a given DT"""
    # TODO Make more data!
    # Make the dataset
    ts, data = analytical_solutions.make_wave_dataset(10, N_data_points, t_max=t_max,
                             params=analytical_solutions.WAVE_PARAMS[1])
    torch_data = data_to_torch(data, device=device)
    torch_ts = data_to_torch(ts, device=device)
    NT,_,NX = data.shape
    dt = ts[1]-ts[0]
    print(dt)
    # Loop over the results
    # TODO load it
    fname = save_root+f"/save_wave_{int(t_max)}.pkl"
    try:
        stash = torch.load(fname)
    except:
        stash = {}
    # TODO: parallelize
    for exp in model_params:
        try:
            exp_obj = stash[tuple(exp.items())]
        except:
            exp_obj = Experiment(torch_data,torch_ts,(2,NX),**exp)
            stash[tuple(exp.items())] = exp_obj
        exp_obj.train(N_iter=N_iter)
    # Post process

    # save the stash
    if not save_root is None:
        with open(fname,"wb") as f:
            torch.save(stash,f)
    # return the pointer to the stash
    return stash

def run_sim(t_max, Nx, meaning, gamma_L1, gamma_L2, learning_rate, batch_size):
    # Make the dataset
    # TODO Change the delta X
    N_data_points=1000
    ts, data = analytical_solutions.make_wave_dataset(Nx, N_data_points, t_max=t_max,
                             params=analytical_solutions.WAVE_PARAMS[1])
    torch_data = data_to_torch(data, device=device)
    torch_ts = data_to_torch(ts, device=device)
    NT,_,NX = data.shape

    keymap={'e':'euler','m':'midpoint','r':'rk4'}
    if '_' in meaning:
        methods = [ keymap[e] for e in meaning.split('_')[1] ]
    else:
        methods = None
    exp_obj = Experiment(torch_data,torch_ts,(2,NX),
                         ode=meaning[0]=='o',
                         methods=methods,
                         gamma_L1=gamma_L1, gamma_L2=gamma_L2, 
                         learning_rate=learning_rate, batch_size=batch_size)
    N_iter=200000
    exp_obj.train(N_iter=N_iter)
    return exp_obj,

if __name__=='__main__':
    device=get_device()
    set_seed()
    model_params = [
        dict(ode=False,gamma_L1 = 0, gamma_L2 = 0,learning_rate=1.0e-2,batch_size=250),
        dict(ode=False,gamma_L1=0, gamma_L2 = 1.0e-7, learning_rate=1.0e-2,batch_size=250),
        dict(ode=True,methods=('euler',),gamma_L1 = 0, gamma_L2 = 0,learning_rate=1.0e-2,batch_size=250),
        #dict(ode=True,methods=('midpoint','rk4'),gamma_L1 = 0, gamma_L2 = 0,learning_rate=1.0e-2,batch_size=250),
        #dict(ode=True,methods=('midpoint','rk4'),gamma_L1 = 0, gamma_L2 = 0,learning_rate=1.0e-3,batch_size=250),
        #dict(ode=True,methods=('midpoint','rk4'),gamma_L1 = 0, gamma_L2 = 0,learning_rate=1.0e-4,batch_size=250),
        dict(ode=True,methods=('midpoint','rk4'),gamma_L1 = 0, gamma_L2 = 0,learning_rate=1.0e-1,batch_size=250),
        dict(ode=True,methods=('midpoint','rk4'),gamma_L1 = 0, gamma_L2 = 0,learning_rate=2.0e-1,batch_size=250),
        dict(ode=True,methods=('euler','midpoint','rk4'),gamma_L1 = 0, gamma_L2 = 0,learning_rate=1.0e-1,batch_size=250),
        dict(ode=True,methods=('euler','midpoint','rk4'),gamma_L1 = 0, gamma_L2 = 0,learning_rate=2.0e-1,batch_size=250),
        #dict(ode=True,methods=('midpoint','rk4'),gamma_L1 = 0, gamma_L2 = 0,learning_rate=1.0,batch_size=250),
    ]
    ts = [ 1.0 ] #0.8,0.9,1.0,1.25,1.5,2.0,3.0,] # 6.0,13.0, 23.0,43.0, 53.0, 63.0, 73.0,78.0, 83.0 ]
    xs = [ 3,4,10, 15, 20, 25 ]
    from SimDataDB import SimDataDB
    sdb = SimDataDB('results/wave_2.sqlite')
    run = sdb.Decorate('wave',
                  [('t_max','FLOAT'),('Nx','INT'),
                   ('meaning','VARCAHR(30)'),('gamma_L1','FLOAT'),
                  ('gamma_L2','FLOAT'),('learning_rate','FLOAT'),('batch_size','INT') ],
                  [('experiment','pickle')],
                      memoize=False)(run_sim)
    
    for t_max in ts:
        for Nx in xs:
            for params in model_params:
                if params['ode']:
                    meaning = 'ode_' + ''.join([m[0] for m in params['methods']])
                else:
                    meaning='mat'
                run(t_max, Nx, meaning, params['gamma_L1'],params['gamma_L2'], 
                    params['learning_rate'], params['batch_size'],)

