import torch
import torchdiffeq

from helper import *

def train_a_neural_ode(data, ts, model=None, batch_size=25, n_future=1, 
                       learning_rate = 1.0e-4, N_iter = 50000,
                       verbose=False, device=None, method='euler'):
    if device is None:
        device = get_device()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    N_print, N_trace = N_iter, 100
    for itr in range(1, N_iter):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(data, ts,
                                               batch_size, n_future)
        pred_y = torchdiffeq.odeint(model, batch_y0, batch_t,
                                    method=method)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        if itr % 1000 == 0:
            with torch.no_grad():
                pred_y = torchdiffeq.odeint(model, batch_y0, batch_t)
                loss = torch.mean(torch.abs(pred_y - batch_y))
    return model,np.array(losses)


def train_a_neural_ode_multi_method(data, ts, model=None, batch_size=25, n_future=1, 
                        learning_rate = 1.0e-4, weight_decay = 0, N_iter=50000, optimizer_key = 'adam',
                        gamma_L1 = 0, gamma_L2 = 0,
                        N_print=100,callback=None,
                        verbose=False, device=None, methods=('euler','midpoint','rk4')):
    if device is None:
        device = get_device()
    if optimizer_key == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    loss = torch.nn.MSELoss()
    #  loss ::= torch.mean(torch.abs(pred_y - batch_y)) # what we were using previously
    losses = []
    for opt_iter in range(N_iter):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(data, ts, batch_size, n_future)
        
        pred_y = torchdiffeq.odeint(model, batch_y0, batch_t, method=methods[0])
        L = loss(pred_y, batch_y)
        for met in methods[1:]:
            pred_y = torchdiffeq.odeint(model, batch_y0, batch_t, method=met)
            L += loss(pred_y, batch_y) 
        raw_loss = L.detach().cpu().numpy() / len(methods) / n_future # Normalize it
        # Add regularizaiton
        # TODO: detect weights; this only works on one type of model
        if gamma_L1 > 0:
            L += gamma_L1*torch.sum(torch.abs(model.net.weight))
        if gamma_L2 > 0:
            L += gamma_L2*torch.sum((model.net.weight)**2)
        # Do the backward step and optimize
        L.backward()
        optimizer.step()
        losses.append(raw_loss)
        if not callback is None and opt_iter%N_print==N_print-1:
            callback(model,opt_iter,raw_loss)
        #if opt_iter % 1000 == 0:
        #    with torch.no_grad():
        #        pred_y = torchdiffeq.odeint(model, batch_y0, batch_t)
        #        loss = torch.mean(torch.abs(pred_y - batch_y))
    if not callback is None:
        callback(model,opt_iter,raw_loss, do_it=True)
    return model,np.array(losses)



def solve_and_plot(model, model_ts, data, data_ts=None, method='rk4', idcs=0):
    if data_ts is None:
        data_ts = model_ts
    with torch.no_grad():
        pred = torchdiffeq.odeint(model, data[0,:], torch_ts, method=method)
    plt.plot(data_ts.cpu().numpy(), data.detach().cpu().numpy()[:,idcs], '--')
    plt.plot(model_ts.cpu().numpy(), pred.detach().cpu().numpy()[:,idcs])
