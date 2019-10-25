import scipy.io#
from matplotlib import pylab as plt
import numpy as np
import matplotlib.animation as animation
import cmocean
from IPython.display import HTML

#from colorspace import diverging_hcl
#pal = diverging_hcl(palette='Blue-Red 2')


#
# Universal settings for what paper figures should look like
#
FIG_SIZE_FULL = (16,6)

#
# 2D Plots
#
def trajectory_plots(trajs):
    """Plot Orbit and time series"""
    plt.figure(figsize=FIG_SIZE_FULL)
    plt.subplot(1,2,1)
    for data in trajs:
        plt.plot(data[:,0], lw=2, c='#1f78b4')
        plt.plot(data[:,1], lw=2, c='#e31a1c')
    plt.ylabel('u', fontsize=18)
    plt.xlabel('time', fontsize=18)
    #plt.legend(loc="best", fontsize=22)
    plt.tick_params(axis='y', labelsize=16)
    plt.tick_params(axis='x', labelsize=16)      
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=6)
   
    plt.subplot(1,2,2)
    for data in trajs:
        plt.plot(data[:,0],data[:,1], '-', c='#fbb4ae', lw=3)
    plt.ylabel('velocity', fontsize=18)
    plt.xlabel('theta', fontsize=18)
    plt.tick_params(axis='y', labelsize=16)
    plt.tick_params(axis='x', labelsize=16)   
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=6) 
    plt.ylim(-1.2,1.2)
    plt.xlim(-1.2,1.2)

def plot_embedding(TRANSFER, experimental_omegas, path_omega, known_omegas, traces=None):
    """Embed and plot the operators, given a callback to the embedding object."""
    X_trans_alpha = TRANSFER(path_omega)
    X_trans_exps = TRANSFER(experimental_omegas)
    X_trans_known = { n:TRANSFER(omega.reshape(-1,4)) for n,omega in known_omegas.items()}
    all_points = cycle('soo****')
    markerdict = defaultdict(lambda : next(all_points))
    plt.figure(figsize=(12,7))
    # That path of the family
    plt.plot(X_trans_alpha[:,0],X_trans_alpha[:,1],'-', lw=3)
    # Known operators
    for n,X in X_trans_known.items():
        plt.plot(X[:,0],X[:,1],markerdict[n],label=n, markersize=20)
    # fix axis
    #xlim= plt.xlim()
    # The end points
    #for t in traces:
    #    X=TRANSFER(t)
    #    plt.plot(X[:,0],X[:,1],'--')
        
    gap = 0.1
    #plt.xlim((X_trans_alpha[:,0].min()-gap,X_trans_alpha[:,0].max()+gap))
    #plt.ylim((X_trans_alpha[:,1].min()-gap,X_trans_alpha[:,1].max()+gap))
    # The paths
    plt.plot(X_trans_exps[:,0],X_trans_exps[:,1],'P',label='experiments', markersize=15, c='k')
    plt.xlabel("first principal component", fontsize=22)
    plt.ylabel("second principal component", fontsize=22)
    plt.set_cmap('Set3')
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=4)
    plt.legend(fontsize=16)
    plt.tight_layout()

#
# Colorplate plots
#
def plot_flow(X, m, n):
    X = X.reshape(m,n)

    x2 = np.arange(0, m, 1)
    y2 = np.arange(0, n, 1)
    mX, mY = np.meshgrid(x2, y2)

    minmax = np.max(np.abs(X)) * 0.85
    #plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
    im = plt.imshow(X.T, cmap=cmocean.cm.balance, interpolation='bicubic', vmin=-minmax, vmax=minmax)
    plt.contourf(mX, mY, X.T, 90, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)

    #im = plt.imshow(X.T, cmap=pal.cmap(name = "Normal Color Vision"), interpolation='bicubic', vmin=-minmax, vmax=minmax)
    #plt.contourf(mX, mY, X.T, 80, cmap=pal.cmap(name = "Normal Color Vision"), alpha=1, vmin=-minmax, vmax=minmax)

    #im = plt.imshow(X.T, cmap=pal.cmap(100, name = "Color Map with 100 Colors"), interpolation='bicubic', vmin=-minmax, vmax=minmax)
    #plt.contourf(mX, mY, X.T, 80, cmap=pal.cmap(100, name = "Color Map with 100 Colors"), alpha=1, vmin=-minmax, vmax=minmax)


    circ=plt.Circle((50,99), radius=30, color='#636363', fill=True)
    im.axes.add_patch(circ)

    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    #plt.close()




def make_animation(img, interval=10, transpose=True):
    fig, ax = plt.subplots()
    minmax = np.max(np.abs(img)) * 0.65
    tr = lambda x : x.T if transpose else x
    canvas = plt.imshow(tr(img[0,:,:]), interpolation='none',
                        cmap=cmocean.cm.balance, 
                        #interpolation='bicubic', 
                        vmin=-minmax, vmax=minmax)
    plt.axis('off')
    plt.tight_layout()
    def animate(i):
        canvas.set_array(tr(img[i,:,:]))
        return canvas,
    ani = animation.FuncAnimation(fig, animate, frames=img.shape[0], interval = interval)
    return ani.to_jshtml()
