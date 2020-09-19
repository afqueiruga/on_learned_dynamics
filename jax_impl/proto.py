from matplotlib import pylab as plt
import numpy as onp

import flax
import jax
import jax.numpy as jnp


class ShallowOde(flax.nn.Module):
    def apply(self, x):
        x = flax.nn.Dense(x, features=50)
        x = flax.nn.tanh(x)
        x = flax.nn.Dense(x, features=2, bias=False)
        return x


def FTrue(_, x):
    """Nonlinear pendulum with g/m=1."""
    return jnp.array([
                      x[1],
                      -9.81*jnp.sin(x[0])
                      ])


class constant_params():
    """Param instances are immutable. They should extend from a parameter dict,
    but also have the f(t) functionality. Maybe they should be a module?"""
    def __init__(self, params):
        self.params = params
    def __call__(self, t):
        return self.params


#
# Integrators
#
def Euler(params_of_t, x, f=None, Dt=None):
    return x + Dt * f(params_of_t, x)


def Midpoint(params_of_t, x, f=None, Dt=None):
    k1 = f(params_of_t, x)
    x1 = x + 0.5*Dt*k1
    return x + Dt*f(params_of_t, x1)


def RK4(params_of_t, x, f=None, Dt=None):
    k1 = f(params_of_t, x)  # t = 0
    x1 = x + 0.5*Dt*k1
    k2 = f(params_of_t, x1)  # t = 1/2
    x2 = x + 0.5*Dt*k2
    k3 = f(params_of_t, x2)  # t = 1/2
    x3 = x + Dt*k3
    k4 = f(params_of_t, x3)  # t = 1
    return x + Dt*(1.0/6.0*k1 + 1.0/3.0*k2 + 1.0/3.0*k3 + 1.0/6.0*k4)


def OdeIntWithPoints(param_of_t, x0, f=None, scheme=Euler, Dt=1.0, N_step=1):
    x = x0
    xs = onp.empty((N_step,) + x0.shape)
    for i in range(N_step):
        x = scheme(param_of_t, x, f, Dt)
        xs[i,:] = x
    return xs


def OdeIntFast(params_of_t, x0, f=None, scheme=Euler, Dt=1.0, N_step=1):
    x = x0
    for i in range(N_step):
        x = scheme(params_of_t, x, f, Dt)
    return x


class OdeBlock(flax.nn.Module):
    def apply(self, x, G=None, scheme=Euler, N_depth=1):
        return OdeIntFast(G.call, x, 1.0/N_depth, scheme, N_depth)
