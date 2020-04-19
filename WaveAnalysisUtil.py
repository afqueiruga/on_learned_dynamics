import numpy as np
from matplotlib import pylab as plt

def make_stencils(Nx):
    Dx = 1.0/(Nx-1.0) # confident it's 1/9
    three_pt = np.array([1,-2,1])/Dx**2
    five_pt = np.array([-1,16,-30,16,-1])/(12*Dx**2)
    seven_pt = np.array([1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90])/Dx**2
#     print(three_pt)
#     print(five_pt)
#     print(seven_pt)
    return three_pt, five_pt, seven_pt

def plot_stencil(experiments, which='lambda'):
    np.set_printoptions(precision=3)
    plt.figure(figsize=plotting.FIG_SIZE_FULL)
    off = 4
    if which=='lambda':
        plt.plot(np.r_[np.zeros(off-1), three_pt,np.zeros(10-off-3)], 'o-', label='known three point')
        plt.plot(np.r_[np.zeros(off-2), five_pt, np.zeros(10-off-5)], 'o-', label='known five point')
        plt.plot(np.r_[np.zeros(off-3), seven_pt], 'o-', label='known seven point')

    for exp in experiments:
        if not hasattr(exp,'all_lambdas'):
            exp.post()
        if which=='lambda':
            row = exp.all_lambdas[-1][10+off,0:10]
        else:
            row = exp.all_omegas[-1][10+off,:]
        print(exp.name(),row[off])
        plt.plot(row,'-.',label=exp.name())
    plt.legend(loc='right')
    plt.xlabel('Stencil index (5 is center)')
    plt.ylabel('A[5,j]')
    plt.show()
    
def plot_stencil_mag(sdb,meaning,gamma_L2,pwargs):
    lrs = sdb.Query(f'select distinct learning_rate from wave where meaning="{meaning}"')
    for lr, in lrs:
        res = sdb.Query(f'select t_max,experiment from wave where meaning="{meaning}" and gamma_L2={gamma_L2} and learning_rate={lr}')
        for t,exp in res:
            exp.post()
        res = sorted(res, key= lambda x : x[0])
        for off in range(4,5):
            plt.plot([t/500 for t,exp in res],[exp.all_lambdas[-1][10+off,off] for t,exp in res],label=lr,*pwargs)
    plt.legend()
    plt.hlines(three_pt[1],0,83/500)
    plt.hlines(five_pt[2],0,83/500)
    plt.hlines(seven_pt[3],0,83/500)
    plt.xlabel('Delta T')
    plt.ylabel('Stencil')