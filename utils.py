import numpy as np

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

def one_step_factory(Lambda, dt, alpha):
    return np.linalg.inv( np.eye(2) - alpha*dt*Lambda ).dot(
                np.eye(2) + (1.0-alpha)*dt*Lambda )

def operator_factory(Lambda, dt, method='euler'):
    """The keys correspond to torchdiffeq, not afqsrungekutta"""
    if method=='euler':
        return np.eye(2)+dt*Lambda
    elif method=='bweuler':
        return np.linalg.inv( np.eye(2) - dt*Lambda)
    elif method=='implicit_trap':
        return one_step_factory(Lambda,dt,0.5)
    elif method=='explicit_adams':
        raise Exception("Didn't implement it")
    elif method=='midpoint':
        ark_key = 'RK2-mid'
        tableau = rk_table[ark_key]
        return operator_from_tableau(Lambda, dt, tableau)
    elif method=='rk4':
        ark_key = 'RK4'
        tableau = rk_table[ark_key]
        return operator_from_tableau(Lambda, dt, tableau)

    
if __name__=='__main__':
    # Test with FWEuler
    hand_test_fweuler = (np.eye(2)+dt*true_A.numpy())
    assert( np.linalg.norm(operator_from_tableau(true_A.numpy(), dt, rk_table['FWEuler'])
                       - hand_test_fweuler )<1.0e-8 )
    # Test with Trapezoidal
    hand_test_trap = np.eye(2)+dt*true_A.numpy() + dt**2/2.0*true_A.numpy().dot(true_A.numpy())
    assert( np.linalg.norm(operator_from_tableau(true_A.numpy(), dt, rk_table['RK2-trap'])
         - hand_test_trap ) < 1.0e-8 )