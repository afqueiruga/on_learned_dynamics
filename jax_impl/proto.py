from matplotlib import pylab as plt
import numpy as onp

import flax
import jax
import jax.numpy as jnp

#
# Two models: One shallow ODE, and the true solution.
#
def FTrue(_, x):
    """Nonlinear pendulum with g=-9.81, m=1.

    Args:
      _: No op for params.
      x: A jnp or np array of [theta, dtheta/dt].
    Returns:
      [dtheta/dt, d^2theta/d^2t].
    """
    return jnp.array([
                      x[1],
                      -9.81*jnp.sin(x[0])
                      ])


class ShallowOde(flax.nn.Module):
    def apply(self, x):
        x = flax.nn.Dense(x, features=50)
        x = flax.nn.tanh(x)
        x = flax.nn.Dense(x, features=2, bias=False)
        return x


#
# Integrators
#
def Euler(params, x, f=None, Dt=1.0):
    return x + Dt * f(params, x)


def Midpoint(params, x, f=None, Dt=1.0):
    k1 = f(params, x)
    x1 = x + 0.5*Dt*k1  # t = 1/2
    return x + Dt*f(params, x1)


def RK4(params, x, f=None, Dt=1.0):
    k1 = f(params, x)  # t = 0
    x1 = x + 0.5*Dt*k1
    k2 = f(params, x1)  # t = 1/2
    x2 = x + 0.5*Dt*k2
    k3 = f(params, x2)  # t = 1/2
    x3 = x + Dt*k3
    k4 = f(params, x3)  # t = 1
    return x + Dt*(1.0/6.0*k1 + 1.0/3.0*k2 + 1.0/3.0*k3 + 1.0/6.0*k4)


SCHEME_TABLE = {
    'Euler': Euler,
    'Midpoint': Midpoint,
    'RK4': RK4,
    }


#
# Integration loops
#
def OdeIntWithPoints(params, x0, f=None, scheme=Euler, Dt=1.0, N_step=1):
    """Integrate a function f in time, and return the complete trajectory.

    Args:
      params: The optimizable parameters of f.
      x0: The initial point at x(t=0).
      f: The rate, of the signature dx/dt:=f(params, x(t)).
      scheme: A callable function that implements one step of f with Dt. See the
        above options.
      Dt: The fixed time step size.
      N_step: How many fixed steps in time to take.
    
    Returns:
      [ [x(0)], [x(Dt)], [x(2*Dt)] ... [x(N_step * Dt)] ]"""
    x = x0
    xs = onp.empty((N_step+1,) + x0.shape)
    xs[0,:] = x0
    for i in range(N_step):
        x = scheme(params, x, f, Dt)
        xs[i+1,:] = x
    return xs


def OdeIntFast(params, x0, f=None, scheme=Euler, Dt=1.0, N_step=1):
    """Numerical integrate f quickly, without saving the trajectory.

    This version is meant to be used for training.

    It returns [x(N_step*Dt)] only, with the same shape as x0.
    """
    x = x0
    for i in range(N_step):
        x = scheme(params, x, f, Dt)
    return x
