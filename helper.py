import numpy as np
import scipy as sci
import torch
import torch.nn.init as init

#
# Torch helpers to keep environments uniform.
#
def set_seed():
    """Set one seed for reproducibility."""
    np.random.seed(10)
    torch.manual_seed(10)

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device

def data_to_torch(data, device):
    """Keep the types uniform everywhere."""
    return torch.tensor(data, dtype=torch.float64, device=device)

#
# Helpers for making results
#
def integrate_model(step_func, u0, nsteps):
    with torch.no_grad():
        us = [u0.cpu().numpy()]
        for i in range(nsteps):
            un = step_func(u0)
            us.append(un.cpu().numpy())
            u0 = un
    U = np.array(us).reshape((-1,)+ u0.shape[1:])
    return U

def integrate_and_plot(step_func, u0, nsteps, ylim=None):
    pass


def integrate_matrix(Omega, u0, nsteps):
    """Integrate when it's a numpy matrix"""
    shape = u0.shape
    us = [u0]
    for i in range(nsteps):
        un = Omega @ u0.flatten()
        us.append(un.reshape(shape))
        u0 = un
    return np.concatenate(us,axis=0)

#
# Helpers for training
#
def get_batch(data, t, N_batch, N_future):
    idcs = np.random.choice(np.arange(data.shape[0] - (N_future+1) , dtype=np.int64), N_batch, replace=False)
    s = torch.from_numpy(idcs)
    batch_u0 = data[s]  # (M, D)
    batch_t= t[:(N_future+1)]  # (T)
    batch_u = torch.stack([ data[s+i] for i in range(N_future+1)], dim=0)  # (T, M, D)
    return batch_u0, batch_t, batch_u



def exp_lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
    if epoch in decayEpoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_rate
        return optimizer
    else:
        return optimizer

#
# Training code.
#
def learn_omega(data, batch_size=25, n_future=1, verbose=False, device=None):
    """Perform the one-step learning for a linear matrix."""
    if device is None:
        device = get_device()

    model = torch.nn.Linear(2,2,bias=False).double().to(device)
    optim = torch.optim.Adam(model.parameters(),1.0e-1)
    loss = torch.nn.MSELoss()
    losses=[]
    omega_trace = [ model.weight.data.cpu().numpy() ]

    N_iter = 1000
    N_print = N_iter
    N_trace = 100
    nsamp = data.shape[0] # The harmonic oscillator is periodic so a test set is meaningless
    for opt_iter in range(N_iter):
        idcs = torch.LongTensor(np.random.choice(nsamp-n_future, size=batch_size)).to(device)
        yy = [ torch.index_select(data ,0, idcs+i) for i in range(n_future+1) ]
        yy_pred = model(yy[0])
        L = loss(yy[1], yy_pred) # n_future=1
        # try multiple steps into the future
        for fut in range(2,n_future+1):
            yy_pred = model(yy_pred)
            L += loss(yy[fut], yy_pred)
        # Do the backward step and optimize
        optim.zero_grad()
        L.backward()
        optim.step()
        losses.append(L.cpu().detach().numpy())
        # Save omega to let us analyze its trajectory
        if opt_iter%N_trace==N_trace-1:
            omega_trace.append(model.weight.data.cpu().numpy())
        # Print diagonistics during training
        if verbose and opt_iter%N_print==N_print-1:
            print(opt_iter,L.item())
            print(list(model.parameters()))
            #integrate_and_plot(model, ylim=None,nsteps=1000)
    if verbose:
        print("Converged with L1: ",losses[-1])
        #plt.semilogy(losses)
    return model, np.array(losses), omega_trace



def learn_lambda(data, batch_size=25, n_future=1, verbose=False,
                 methods=('FW','BW','TR'), device=None, dt=1):
    """Perform the one-step learning for a linear matrix."""
    if device is None:
        device = get_device()

    # Implementations of one-step methods
    I = torch.eye(2, dtype=torch.double, device=device)
    fwstep = lambda model, y : y + dt*model(y)
    bwstep = lambda model, y : torch.einsum("ij,aj->ai",torch.inverse(I - dt*model.weight),y)
    trstep = lambda model, y : torch.einsum("ij,aj->ai",torch.inverse(I-0.5*dt*model.weight), ( y + 0.5*dt*model(y) ))

    model = torch.nn.Linear(2,2,bias=False).double().to(device)
    optim = torch.optim.Adam(model.parameters(),lr=5.0e-2, weight_decay=0.0)
    loss = torch.nn.MSELoss()
    losses=[]
    #do_a_path_and_plot(model)

    N_iter = 1000
    N_print = N_iter+1 #//10
    nsamp = data.shape[0] # The harmonic oscillator is periodic so a test set is meaningless
    for opt_iter in range(N_iter):
        idcs = torch.LongTensor(np.random.choice(nsamp-n_future, size=batch_size)).to(device)
        yy = [ torch.index_select(data,0,idcs+i) for i in range(n_future+1) ]
        y_pred_fw = fwstep(model, yy[0])
        y_pred_bw = bwstep(model, yy[0])
        y_pred_tr = trstep(model, yy[0])

        if methods == ('FW','BW','TR'):
            L = loss(yy[1], y_pred_fw) + loss(yy[1], y_pred_bw) + loss(yy[1], y_pred_tr)
        elif methods == ('FW','BW'):
            L = loss(yy[1], y_pred_fw) + loss(yy[1], y_pred_bw)
        elif methods == ('FW',):
            L = loss(yy[1], y_pred_fw)
        elif methods == ('BW','TR'):
            L = loss(yy[1], y_pred_bw) + loss(yy[1], y_pred_tr)
        elif methods == ('BW',):
            L = loss(yy[1], y_pred_bw)
        else:
            L = loss(yy[1], y_pred_tr)

        optim.zero_grad()
        L.backward()
        optim.step()
        losses.append(L)
        if verbose and opt_iter%N_print==N_print-1:
            print(opt_iter,L.item())
            print(list(model.parameters()))
            #do_a_path_and_plot(model, trstep, ylim=None,nsteps=1000) # This is wrong now

        exp_lr_scheduler(optim, opt_iter, lr_decay_rate=0.3, decayEpoch=[200,500,800])

    if verbose:
        print("Converged with L1: ",losses[-1])

    nump_mat = model.weight.data.cpu().numpy()
    op_tr = np.linalg.inv(np.eye(2) - 0.5*dt*nump_mat).dot( np.eye(2) + 0.5*dt*nump_mat )
    op_fw = np.eye(2) + dt*nump_mat
    op_bw = np.linalg.inv(np.eye(2) - dt*nump_mat)
    return model, np.array([l.cpu().detach().numpy() for l in losses]), nump_mat, op_tr, op_fw, op_bw



#
# Learn a self-feeding model
# TODO: RNN is the wrong word
def learn_rnn(data, model=None, batch_size=25, n_future=1, 
              learning_rate = 1.0e-4, N_iter = 50000, N_print=1e10, 
              gamma_L1=0.0, gamma_L2=0.0,
              verbose=False, device=None, callback=None):
    """Perform the learning for an arbitrary model."""
    if device is None:
        device = get_device()
    if model==None:
        model = torch.nn.Linear(data.shape[-1],data.shape[-1],bias=False).double().to(device)
    optim = torch.optim.Adam(model.parameters(),learning_rate)
    loss = torch.nn.MSELoss()
    losses=[]
    N_print, N_trace = N_print, 100
    nsamp = data.shape[0] # The harmonic oscillator is periodic so a test set is meaningless
    for opt_iter in range(N_iter):
        idcs = torch.LongTensor(np.random.choice(nsamp-n_future, size=batch_size)).to(device)
        yy = [ torch.index_select(data ,0, idcs+i) for i in range(n_future+1) ]
        yy_pred = model(yy[0])
        L = loss(yy[1], yy_pred) # n_future=1
        # try multiple steps into the future
        for fut in range(2,n_future+1):
            yy_pred = model(yy_pred)
            L += loss(yy[fut], yy_pred)
        # Add regularizaiton
        # TODO: detect weights; this only works on one type of model
        if gamma_L1 > 0:
            L += gamma_L1*torch.sum(torch.abs(model.net.weight))
        if gamma_L2 > 0:
            L += gamma_L2*torch.sum((model.net.weight)**2)
        # Do the backward step and optimize
        optim.zero_grad()
        L.backward()
        optim.step()
        losses.append(L.cpu().detach().numpy())

        #model.net.weight.data[:] = torch.nn.functional.softshrink(model.net.weight.data, gamma)

        if not callback is None and opt_iter%N_print==N_print-1:
            callback(model,opt_iter,L.item())
        # Print diagonistics during training
        if verbose and opt_iter%N_print==N_print-1:
            print(opt_iter,L.item())
    if verbose:
        print("Converged with L1: ",losses[-1])
    if not callback is None:
        callback(model,opt_iter,L.item())
    return model, np.array(losses),
