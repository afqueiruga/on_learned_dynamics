import scipy.io#
from matplotlib import pylab as plt
import numpy as np
import matplotlib.animation as animation
import cmocean
from IPython.display import HTML

def plot_flow(X, m, n):
    X = X.reshape(m,n)

    x2 = np.arange(0, m, 1)
    y2 = np.arange(0, n, 1)
    mX, mY = np.meshgrid(x2, y2)

    minmax = np.max(np.abs(X)) * 0.65
    #plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
    im = plt.imshow(X.T, cmap=cmocean.cm.balance, interpolation='bicubic', vmin=-minmax, vmax=minmax)
    plt.contourf(mX, mY, X.T, 90, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    
    
    circ=plt.Circle((50,99), radius=30, color='#636363', fill=True)
    im.axes.add_patch(circ)   
        
    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    #plt.close()
    
    



def make_animation(img, interval=10, transpose=True):
    fig, ax = plt.subplots()
    minmax = np.max(np.abs(img)) * 0.65
    canvas = plt.imshow(img[0,:,:].T, interpolation='none',
                        cmap=cmocean.cm.balance, 
                        #interpolation='bicubic', 
                        vmin=-minmax, vmax=minmax)
    plt.axis('off')
    plt.tight_layout()
    def animate(i):
        canvas.set_array(img[i,:,:].T)
        return canvas,
    ani = animation.FuncAnimation(fig, animate, frames=img.shape[0], interval = 10)
    return ani.to_jshtml()