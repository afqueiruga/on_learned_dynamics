import numpy as np
import torch
from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

#
# Analytical solutions
#
# deprecated
def solution_harmonic_osc(u0, ts):
    theta_expr = u0*np.sin(ts)
    d_theta_d_t = -u0*np.cos(ts)
    return np.stack([theta_expr, d_theta_d_t], axis=1)

#ts = np.linspace(0., 25., N_data)
def solution_general_linear(Lambda, x0, ts):
    """Lambda is a square numpy array"""
    eigs, V = np.linalg.eig(Lambda)
    interior = np.exp(eigs.reshape((-1,1))*ts.reshape(1,-1))
    x = np.einsum("ij,ja,jk,k->ai",
                V, interior, np.linalg.inv(V), u0 )
    return np.real(u)

def true_omega_linear(Lambda, dt):
    eigs, V = np.linalg.eig(Lambda)
    Omega = V.dot( np.diag(np.exp(eigs*dt)).dot( np.linalg.inv(V) ) )
    return np.real(Omega)


def solution_wave(k, ts, u0):
    """Uses detest"""
    # TODO
    pass

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

def solution_pendulum_xy(t, theta0):
    """Solution for the nonlinear pendulum in xy space."""
    thetas = solution_pendulum_theta(t,theta0)
    return np.stack([  np.cos(thetas[:,0]), # x
                       np.sin(thetas[:,0]), # y
                      -np.sin(thetas[:,0])*thetas[:,1], # v_x
                       np.cos(thetas[:,0])*thetas[:,1], # v_y
                     ]).T



#
# Routines for making numerical integrators
#
#import afqsrungekutta as ark
#rk_table = ark.exRK_table
# Taken out of ark to reduce dependencies:
array = np.array
rk_table = {
    'FWEuler': {'a': array([[0.]]),
                'b': array([1.]),
                'c': array([0.])},
    'RK2-trap': {'a': array([[0., 0.],
                            [1., 0.]]),
                'b': array([0.5, 0.5]),
                'c': array([0., 1.])},
    'RK2-mid': {'a': array([[0. , 0. ],
                            [0.5, 0. ]]),
                'b': array([0., 1.]),
                'c': array([0. , 0.5])},
    'RK3-1': {'a': array([[0.        , 0.        , 0.        ],
                        [0.66666667, 0.        , 0.        ],
                        [0.33333333, 0.33333333, 0.        ]]),
             'b': array([0.25, 0.  , 0.75]),
             'c': array([0.        , 0.66666667, 0.66666667])},
    'RK4': {'a': array([[0. , 0. , 0. , 0. ],
                        [0.5, 0. , 0. , 0. ],
                        [0. , 0.5, 0. , 0. ],
                        [0. , 0. , 1. , 0. ]]),
            'b': array([0.16666667, 0.33333333, 0.33333333, 0.16666667]),
            'c': array([0. , 0.5, 0.5, 1. ])}
}

def operator_from_tableau(lmbda, dt, tableau):
    """Make the discrete operator for a linear ODE for an explicit RK tableau"""
    # Assuming explicit
    a = tableau['a']
    b = tableau['b']
    c = tableau['c']
    I = np.eye(lmbda.shape[0])
    Omegas = [ lmbda ]
    for i in range(1,len(c)):
        dui_du0 = I.copy()
        for j in range(i):
            dui_du0 += dt*a[i,j]*Omegas[j]
        Omegai = lmbda.dot( dui_du0 )
        Omegas.append(Omegai)
    Omega_full = I.copy()
    for j in range(len(b)):
        Omega_full += dt*b[j]*Omegas[j]
    return Omega_full

if __name__=='__main__':
    # Test with FWEuler
    hand_test_fweuler = (np.eye(2)+dt*true_A.numpy())
    assert( np.linalg.norm(operator_from_tableau(true_A.numpy(), dt, rk_table['FWEuler'])
                       - hand_test_fweuler )<1.0e-8 )
    # Test with Trapezoidal
    hand_test_trap = np.eye(2)+dt*true_A.numpy() + dt**2/2.0*true_A.numpy().dot(true_A.numpy())
    assert( np.linalg.norm(operator_from_tableau(true_A.numpy(), dt, rk_table['RK2-trap'])
         - hand_test_trap ) < 1.0e-8 )


def operator_factory(lmbda, dt, method='euler'):
    if method=='euler':
        return np.eye(2)+dt*lmbda_approx_induced
    elif method=='explicit_adams':
        raise Exception("Didn't implement it")
    elif method=='midpoint':
        ark_key = 'RK2-mid'
        tableau = rk_table[ark_key]
        return operator_from_tableau(lmbda, dt, tableau)
    elif method=='rk4':
        ark_key = 'RK4'
        tableau = rk_table[ark_key]
        return operator_from_tableau(lmbda, dt, tableau)
def make_integrator(Lambda, method, dt):
    return np.eye(2,writeable=False)


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

def integrate_and_plot(step_func, u0, steps):
    pass


#
# Training code.
#
def learn_omega():
    pass

def learn_lambda():
    pass
