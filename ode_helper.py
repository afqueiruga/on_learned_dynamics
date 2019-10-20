import torch
import torchdiffeq

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
        losses.append(loss.detach().numpy())
        if itr % 1000 == 0:
            with torch.no_grad():
                pred_y = torchdiffeq.odeint(model, batch_y0, batch_t)
                loss = torch.mean(torch.abs(pred_y - batch_y))
    return model,np.array(losses)


def solve_and_plot(model, model_ts, data, data_ts=None, method='rk4', idcs=0):
    if data_ts is None:
        data_ts = model_ts
    with torch.no_grad():
        pred = torchdiffeq.odeint(model, data[0,:], torch_ts, method=method)
    plt.plot(data_ts.cpu().numpy(), data.detach().cpu().numpy()[:,idcs], '--')
    plt.plot(model_ts.cpu().numpy(), pred.detach().cpu().numpy()[:,idcs])