import numpy as np
import scipy as sci
import torch

def get_device():
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device

def data_to_torch(data, device):
    return torch.tensor(data, dtype=torch.float64, device=device)



#
# Helpers for making results
#
def integrate_model(step_func, u0, steps):
    with torch.no_grad():
        u0 = initial_point
        us = [u0.cpu().numpy()]
        for i in range(nsteps):
            un = step_func(u0)
            us.append(un.cpu().numpy())
            u0 = un
        U = np.array(us).reshape((-1, u0.shape[-1]))
    return U

def integrate_and_plot(step_func, u0, nsteps, ylim=None):
    pass


def get_batch(data, t, N_batch, N_future):
    idcs = np.random.choice(np.arange(data.shape[0] - N_future , dtype=np.int64), N_batch, replace=False)
    s = torch.from_numpy(idcs)
    batch_u0 = data[s]  # (M, D)
    batch_t= t[:N_future]  # (T)
    batch_u = torch.stack([ data[s + i] for i in range(N_future)], dim=0)  # (T, M, D)
    return batch_u0, batch_t, batch_u



#
# Training code.
#
def learn_omega(data, batch_size=25, n_future=1, verbose=False, device=None):
    if device is None:
        device = get_device()

    model = torch.nn.Linear(2,2,bias=False).double().to(device)
    optim = torch.optim.Adam(model.parameters(),1.0e-1)
    loss = torch.nn.MSELoss()
    losses=[]
    #do_a_path_and_plot(model)

    N_iter = 1000
    N_print = N_iter
    nsamp = data.shape[0] # The harmonic oscillator is periodic so a test set is meaningless
    for opt_iter in range(N_iter):
        idcs = torch.LongTensor(np.random.choice(nsamp-n_future, size=batch_size)).to(device)
        yy = [ torch.index_select(data ,0, idcs+i) for i in range(n_future+1) ]
        yy_pred = model(yy[0])
        L = loss(yy[1], yy_pred) # n_future=1
        for fut in range(2,n_future+1):
            yy_pred = model(yy_pred)
            L += loss(yy[fut], yy_pred)
        optim.zero_grad()
        L.backward()
        optim.step()
        losses.append(L)
        if verbose and opt_iter%N_print==N_print-1:
            print(opt_iter,L.item())
            print(list(model.parameters()))
            #integrate_and_plot(model, ylim=None,nsteps=1000)
    if verbose:
        print("Converged with L1: ",losses[-1])
        #plt.semilogy(losses)
    return model, np.array([l.cpu().detach().numpy() for l in losses])

def learn_lambda():
    pass
