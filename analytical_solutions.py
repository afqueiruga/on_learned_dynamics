import numpy as np
from scipy.special import ellipj, ellipk

#
# Analytical solutions
#
def solution_harmonic_osc(u0, ts):
    """Deprecated"""
    theta_expr = u0*np.sin(ts)
    d_theta_d_t = -u0*np.cos(ts)
    return np.stack([theta_expr, d_theta_d_t], axis=1)

def solution_general_linear(Lambda, u0, ts):
    """Lambda is a square numpy array"""
    eigs, V = np.linalg.eig(Lambda)
    interior = np.exp(eigs.reshape((-1,1))*ts.reshape(1,-1))
    u = np.einsum("ij,ja,jk,k->ai",
                V, interior, np.linalg.inv(V), u0 )
    return np.real(u)

def true_omega_linear(Lambda, dt):
    eigs, V = np.linalg.eig(Lambda)
    Omega = V.dot( np.diag(np.exp(eigs*dt)).dot( np.linalg.inv(V) ) )
    return np.real(Omega)

#
# The wave
#
def solution_wave(k, ts, u0):
    """Uses detest"""
    # TODO
    pass

#
# The nonlinear pendulum
#
def solution_pendulum_theta(t, theta0):
    """Solution for the nonlinear pendulum in theta space."""
    S = np.sin(0.5*(theta0) )
    K_S = ellipk(S**2)
    omega_0 = np.sqrt(9.81)
    sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
    theta = 2.0*np.arcsin( S*sn )
    d_sn_du = cn*dn
    d_sn_dt = -omega_0 * d_sn_du
    d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
    return np.stack([theta, d_theta_dt],axis=1)

def true_step_pendulum_xy(x,y,vx,vy, Dt):
    """True solution to one step"""
    pass
def force_pendulum_xy(x,y,vx,vy):
    pass

def solution_pendulum_xy(t, theta0):
    """Solution for the nonlinear pendulum in xy space."""
    thetas = solution_pendulum_theta(t,theta0)
    return np.stack([  np.cos(thetas[:,0]), # x
                       np.sin(thetas[:,0]), # y
                      -np.sin(thetas[:,0])*thetas[:,1], # v_x
                       np.cos(thetas[:,0])*thetas[:,1], # v_y
                     ]).T

#
# The double pendulum
#