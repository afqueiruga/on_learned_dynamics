import numpy as np
import scipy as sci
import torch

def data_to_torch(data)
    try:
        device = torch.device('cuda')
        to_data = torch.tensor(data, dtype=torch.float64, device=device)
        print("Found a GPU")
    except:
        print("Using the CPU")
        device = torch.device('cpu')
        to_data = torch.tensor(data, dtype=torch.float64, device=device)

    return torch_data
