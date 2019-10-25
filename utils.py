import numpy as np

#
# Routines for analyzing systems
#
def lambda_of_omega(Omega):
    """Takes the matrix log of a discrete operator to extract the continuous operator."""
    w,V = np.linalg.eig(Omega)
    Lambda = 1/dt * V @ np.diag(np.log(w)) @ np.linalg.inv(V)
    Lambda = np.array(Lambda,dtype=np.double)
    return Lambda

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

def operator_from_tableau(Lambda, dt, tableau):
    """Make the discrete operator for a linear ODE for an explicit RK tableau"""
    Lambda.shape[0]
    # Assuming explicit
    a = tableau['a']
    b = tableau['b']
    c = tableau['c']
    I = np.eye(Lambda.shape[0])
    Omegas = [ lmbda ]
    for i in range(1,len(c)):
        dui_du0 = I.copy()
        for j in range(i):
            dui_du0 += dt*a[i,j]*Omegas[j]
        Omegai = Lambda @ dui_du0
        Omegas.append(Omegai)
    Omega_full = I.copy()
    for j in range(len(b)):
        Omega_full += dt*b[j]*Omegas[j]
    return Omega_full


def one_step_factory(Lambda, dt, alpha):
    dim = Lambda.shape[0]
    return np.linalg.inv( np.eye(dim) - alpha*dt*Lambda ).dot(
                np.eye(dim) + (1.0-alpha)*dt*Lambda )


def operator_factory(Lambda, dt, method='euler'):
    """Make an Omega discrete transfer function from the continuous companion matrix Lambda. 
    The keys correspond to torchdiffeq, not afqsrungekutta"""
    if method=='euler':
        return np.eye(Lambda.shape[0])+dt*Lambda
    elif method=='bweuler':
        return np.linalg.inv( np.eye(Lambda.shape[0]) - dt*Lambda)
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
    A = np.array([[0.1,1.0],[-1.0,0.1]])
    hand_test_fweuler = (np.eye(2)+dt*true_A)
    assert( np.linalg.norm(operator_from_tableau(A, dt, rk_table['FWEuler'])
                       - hand_test_fweuler )<1.0e-8 )
    # Test with Trapezoidal
    hand_test_trap = np.eye(2)+dt*A + dt**2/2.0*A.dot(A)
    assert( np.linalg.norm(operator_from_tableau(A, dt,rk_table['RK2-trap'])
         - hand_test_trap ) < 1.0e-8 )