import numpy as np
from numpy import outer, trace, dot, vdot, pi, log2, exp, sin, cos, sqrt, sign, diag, linspace, arange, array, inf, zeros, eye, arccos, arcsin, arctan, mean, std, concatenate, kron, sign, ceil, log
from numpy.random import uniform, normal, randint
from scipy.linalg import eig, eigh, norm, expm, sqrtm
from numpy.linalg import svd, norm, inv, pinv
from scipy.stats import sem
from scipy.optimize import minimize
from functools import reduce
from itertools import product
import qutip as qp


P0 = np.array([[1., 0.],
               [0., 0.]]) # |0><0|
P1 = np.array([[0., 0.],
               [0., 1.]]) # |1><1|
X = np.array([[0.,1.],
              [1.,0.]]) # X Pauli matrix
Y = np.array([[0.,-1.j],
              [1.j, 0.]]) # Y Pauli matrix
Z = np.array([[1., 0.],
              [0.,-1.]]) # Z Pauli matrix
I = np.array([[1.,0.],
              [0.,1.]]) # 2x2 identity matrix

# some functions # 

def kron_A_N(A, N): # fast kron(A, eye(N))
    m,n = A.shape
    out = zeros((m, N, n, N), dtype=A.dtype)
    r = arange(N)
    out[:, r, :, r] = A
    out.shape = (m*N, n*N)
    return out
    
def kron_N_A(A, N): # fast kron(eye(N), A)
    m,n = A.shape
    out = zeros((N, m, N, n), dtype=A.dtype)
    r = np.arange(N)
    out[r, :, r, :] = A
    out.shape = (m*N, n*N)
    return out

def trace_distance(A, B):
    sub = A - B
    return trace(sqrtm(dot(sub.conj().T, sub))).real / 2

def fidelity(A, B):
    res = reduce(dot, [sqrtm(A), B, sqrtm(A)])
    res = sqrtm(res)
    return trace(res).real**2

def sup_fidelity(A, B):
    """ An upper bound for the usual fidelity """
    t1 = trace(A@B).real
    t2 = max(0, 1 - trace(A@A).real)
    t3 = max(0, 1 - trace(B@B).real)
    return t1 + sqrt(t2)*sqrt(t3)

def partial_trace(dm, m=None, n=None, subsystem=0):
    """ Simple and fast, but cuts only in halves. """
    if (m is None) or (n is None): # cut in equal halves
        N = log2(len(dm))
        m = int(N / 2)
        n = int(N - m)
        m = 2**m
        n = 2**n
    if subsystem == 0:
        return trace(dm.reshape((m, n, m, n)), axis1=0, axis2=2)
    elif subsystem == 1:
        return trace(dm.reshape((m, n, m, n)), axis1=1, axis2=3)

def concurrence_pure(dm):
    dm_red = partial_trace(dm)
    return sqrt(2*(1 - trace(dm_red@dm_red).real))

def concurrence(dm):
    YY = kron(Y, Y)
    dm_t = YY@dm.conj()@YY
    R = dm_t@dm
    lambdas = [l if l > 0 else 0 for l in np.sort(eig(R)[0].real)]
    c = sqrt(lambdas[3]) - sqrt(lambdas[2]) - sqrt(lambdas[1]) - sqrt(lambdas[0])
    return max(0, c)

def two_subsys_negativity(dm):
    def partial_transpose(A, n, m):
        A_c = array(A)
        Bt = A[:n, m:].copy()
        Ct = A[n:, :m].copy()
        A_c[:n, m:] = Ct
        A_c[n:, :m] = Bt
        return A_c
    dm_ptrans = partial_transpose(dm, int(len(dm)/2), int(len(dm)/2))
    lambda_min = eigh(dm_ptrans)[0][0]
    return 2*abs(min(0, lambda_min))

def prev_to_next_ansatz(pars, n_tot_p, n_meas_p, n_layers_p, n_tot_n, n_meas_n, n_layers_n, subsval=0):
    """Extends the outcome values and the ansatz hea_cry_rzrx to new n_tot and n_layers, filling the angles with zeros."""
    x0 = []
    it = iter(pars)
    for q in range(n_tot_p):
        x0.append(next(it))
        x0.append(next(it))
    for q in range(n_tot_n - n_tot_p):
        x0.append(subsval)
        x0.append(subsval)   
    for l in range(n_layers_p):
        for q in range(n_tot_p - 1):
            x0.append(next(it))
        for q in range(n_tot_p - 1, n_tot_n - 1):
            x0.append(subsval)
        for q in range(n_tot_p):
            x0.append(next(it))
            x0.append(next(it))
        for q in range(n_tot_p, n_tot_n):
            x0.append(subsval)
            x0.append(subsval)
    for l in range(n_layers_n - n_layers_p):
        for q in range(n_tot_n - 1):
            x0.append(subsval)
        for q in range(n_tot_n):
            x0.append(subsval)
            x0.append(subsval) 
    x0 = x0 + list(kron(diag([next(it) for i in range(2**n_meas_p)]), eye(2**(n_meas_n - n_meas_p))).diagonal())
    return x0


def gen_even_ent_data(n, n_inp=2, mixed=True, marks="neg", n_chunks=100):
    """ Generates a data set of states with evenly distributed entanglements """
    
    d = 2**n_inp
    
    if marks == "neg":
        ent_measure_func = two_subsys_negativity
    elif marks == "con":
        ent_measure_func = concurrence 

    if mixed == True:
        rank = d
    else:
        rank = 1
    
    ent_count_max = int(ceil(n/n_chunks))
    ent_line = linspace(0, 1, n_chunks + 1)[1:]
    ent_counts = [0]*n_chunks
    count = 0
    dms = []
    labels = []
    while count < n:
        print("%d" %count, end="\r")
        dm = array(qp.rand_dm_ginibre(d, randint(1, rank + 1))) # lame, but works faster for mixed states
        ent = ent_measure_func(dm)
        ent_diffs = ent_line - ent
        ind = np.abs(ent_diffs).argmin()
        if sign(ent_diffs[ind]) == -1:
            ind += 1
        if ent_counts[ind] < ent_count_max:
            ent_counts[ind] += 1        
            count += 1
            dms.append(dm)
            labels.append(ent)
            
    return dms, labels


# quantum state generators # 

def rand_sv(n_qubits):
    d = 2**n_qubits
    sv = uniform(-1, 1, d) + 1j*uniform(-1, 1, d)
    return sv/norm(sv)

def rand_dm(n_qubits):
    d = 2**n_qubits
    H = uniform(-1, 1, [d, d]) + 1j*uniform(-1, 1, [d, d])
    dm = H@H.conj().T
    dm = dm/trace(dm).real
    return dm
  

# Fisher informations #

def cfi(channel_func, dm, p, channel_args, povm, n_copies=1, n_ext=0, dp=1e-5):
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_m = reduce(kron, [channel_func(dm, p-dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 0
    for op in povm:
        prob = trace(dot(dm_n, op)).real
        if prob > 0:
            prob_p = trace(dot(dm_n_p, op)).real
            prob_m = trace(dot(dm_n_m, op)).real
            der = (prob_p - prob_m)/(2*dp)
            fi += der**2/prob
    return fi

def qfi(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-2):
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(fidelity(dm_n, dm_n_p))) / dp**2
    return fi

def qfi_central(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-2):
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_m = reduce(kron, [channel_func(dm, p-dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(fidelity(dm_n_m, dm_n_p))) / dp**2/4
    return fi

def sup_qfi(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-5):
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(sup_fidelity(dm_n, dm_n_p))) / dp**2
    return fi


def sld(channel_func, channel_par, channel_args, dm_ini, n_copies=1, n_ext=0, dp=1e-5, return_fi=False):
    """
        Numerically finds the SLD operator L.
        Optionally returns the classical and quantum Fisher informations.
    """
    
    n_inp = int(log2(len(dm_ini)))*n_copies
    n_tot = n_inp + n_ext
    d = 2**n_tot
        
    dm_ext = diag([1] + [0]*(2**(n_ext) - 1))
    dm_n = reduce(kron, [channel_func(dm_ini, channel_par, *channel_args)]*n_copies + [dm_ext])
    dm_n_p = reduce(kron, [channel_func(dm_ini, channel_par+dp, *channel_args)]*n_copies + [dm_ext])
    dm_n_m = reduce(kron, [channel_func(dm_ini, channel_par-dp, *channel_args)]*n_copies + [dm_ext])
    dm_n_der = (dm_n_p - dm_n_m)/(2*dp)
    
    evals, evecs = eigh(dm_n)
    evecs = evecs.T
    
    L = zeros([d, d], dtype=complex)
    for i in range(d):
        for j in range(d):
            denom = evals[i] + evals[j]
            if denom > 1e-5:
                numer = evecs[i].conj().T@dm_n_der@evecs[j]
                oper = outer(evecs[i], evecs[j].conj().T)
                L += 2*numer/denom*oper
    # print("Incompliance with the definition of SLD:", norm( (L@dm_n + dm_n@L)/2 - dm_n_der ))
    
    if return_fi == True:
        evecs_L = eigh(L)[1].T
        projs_L = [outer(vec, vec.conj().T) for vec in evecs_L]
        CFI = cfi(channel_func, dm_ini, channel_par, channel_args, projs_L, n_copies=n_copies, n_ext=n_ext, dp=dp)
        QFI = trace(L@L@dm_n).real
        return L, CFI, QFI
    else:
        return L

### channels ###

def hw_channel(dm, p):
    """ Holevo-Werner channel """
    d = len(dm)
    return ((d - p)*eye(d) + (d*p - 1)*dm.T)/(d**2 - 1)
    
def depolarizing_channel(dm, p):
    d = len(dm)
    return (1 - p)*dm + p/d*eye(d)

def generalized_amplitude_damping_channel(dm, g, N, target_qubit):
    
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    
    K1 = array([[1,           0],
                [0, sqrt(1 - g)]])*sqrt(1 - N)
    K1 = reduce(kron, [eye(dl), K1, eye(dr)]) # inefficient
    
    K2 = array([[0, sqrt(g*(1 - N))],
                [0,               0]])
    K2 = reduce(kron, [eye(dl), K2, eye(dr)])
    
    K3 = array([[sqrt(1 - g), 0],
                [          0, 1]])*sqrt(N)
    K3 = reduce(kron, [eye(dl), K3, eye(dr)])
            
    K4 = array([[0,         0],
                [sqrt(g*N), 0]])
    K4 = reduce(kron, [eye(dl), K4, eye(dr)])
    
    dm1 = reduce(dot, [K1, dm, K1.conj().T])
    dm2 = reduce(dot, [K2, dm, K2.conj().T])
    dm3 = reduce(dot, [K3, dm, K3.conj().T])
    dm4 = reduce(dot, [K4, dm, K4.conj().T])
        
    return dm1 + dm2 + dm3 + dm4


def another_generalized_amplitude_damping_channel(dm, g, N, target_qubit):
    """ Adapted from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.012317 """
    
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    
    K1 = array([[1,           0],
                [0,     sqrt(g)]])*sqrt(N)
    K1 = reduce(kron, [eye(dl), K1, eye(dr)]) # inefficient
    
    K2 = array([[0, sqrt((1 - g))],
                [0,             0]])*sqrt(N)
    K2 = reduce(kron, [eye(dl), K2, eye(dr)])
    
    K3 = array([[sqrt(g), 0],
                [      0, 1]])*sqrt(1 - N)
    K3 = reduce(kron, [eye(dl), K3, eye(dr)])
            
    K4 = array([[0,           0],
                [sqrt(1 - g), 0]])*sqrt(1 - N)
    K4 = reduce(kron, [eye(dl), K4, eye(dr)])
    
    dm1 = reduce(dot, [K1, dm, K1.conj().T])
    dm2 = reduce(dot, [K2, dm, K2.conj().T])
    dm3 = reduce(dot, [K3, dm, K3.conj().T])
    dm4 = reduce(dot, [K4, dm, K4.conj().T])
        
    return dm1 + dm2 + dm3 + dm4

    
def X_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [X]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def Y_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [Y]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def Z_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [Z]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def random_channel(dm, p, pars, p_index, pauli_basis=None):
    d = len(dm)
    n_inp = int(log2(d))
    n_ext = 2*n_inp
    n_tot = n_inp + n_ext
    pars_conc = concatenate([pars[:p_index], [p], + pars[p_index:]])
    V = su2n(n_tot, pars_conc, pauli_basis=pauli_basis)
    dm_ext = diag([1] + [0]*(2**n_ext - 1))
    dm_n = V@kron(dm, dm_ext)@V.conj().T
    dm_n = trace(dm_n.reshape(2**n_inp, 2**n_ext, 2**n_inp, 2**n_ext), axis1=1, axis2=3) # partial trace with respect to the extension
    return dm_n


### Hamiltonians ###

def ising_ham(n_qubits, h, J=1, bc="closed"):
    d = 2**n_qubits
    Hx = zeros((d, d), dtype=complex)
    for q in range(n_qubits):
        X_op = [I]*q + [X] + [I]*(n_qubits-q-1)
        Hx = Hx + reduce(kron, X_op)
    Hzz = zeros((d, d), dtype=complex)
    for q in range(n_qubits-1):
        Hzz = Hzz + reduce(kron, [I]*q + [Z, Z] + [I]*(n_qubits-q-2))
    if bc == "closed" and n_qubits > 2:
        Hzz = Hzz + reduce(kron, [Z] + [I]*(n_qubits-2) + [Z])
    if n_qubits == 1: # lame
        Hzz = 1*Z
    return -J*(Hzz + h*Hx)


def schwinger_ham(n_qubits, m, w=1, g=1):
    
    d = 2**n_qubits
    sp = (X + 1j*Y)/2
    sm = (X - 1j*Y)/2
    
    term_1 = 1j * zeros((d, d))
    for j in range(n_qubits):
        k = (j + 1) % n_qubits
        crea = [I]*j + [sp] + [I]*(n_qubits - j - 1)
        anni = [I]*k + [sm] + [I]*(n_qubits - k - 1)
        crea = reduce(kron, crea)
        anni = reduce(kron, anni)
        op = crea@anni
        term_1 = term_1 + op + op.conj().T
    term_1 = w * term_1

    term_2 = 1j * zeros((d, d))
    for j in range(n_qubits):
        operator = [I]*j + [Z] + [I]*(n_qubits - 1 - j)
        term_2 = term_2 + (-1)**(j + 1) * reduce(kron, operator)
    term_2 = m / 2 * term_2

    term_3 = 1j * zeros((d, d))
    for j in range(n_qubits):
        L = 1j * zeros((d, d))
        for l in range(j + 1):
            operator = [I]*n_qubits
            operator[l] = Z + (-1)**(l + 1) * I
            L = L - 0.5 * reduce(kron, operator)
        term_3 = term_3 + L@L
    term_3 = g * term_3
        
    return term_1 + term_2 + term_3


# ansatzes #

def su2(pars):
    """ Universal single-qubit rotation """
    return array([[exp(1j * (-pars[1] - pars[2])) * cos(pars[0]), -exp(1j * (-pars[1] + pars[2])) * sin(pars[0])],
                  [exp(1j * ( pars[1] - pars[2])) * sin(pars[0]),  exp(1j * ( pars[1] + pars[2])) * cos(pars[0])]])

def su2n(n_qubits, pars, pauli_basis=None):
    if pauli_basis is None:
        pauli_basis = [reduce(kron, paulis_tring) for paulis_tring in list(product([I, X, Y, Z], repeat=n_qubits))[1:]]
    op = zeros([2**n_qubits, 2**n_qubits], dtype=complex)
    for par, pauli_string in zip(pars, pauli_basis):
        op += par*pauli_string
    return expm(-1j*op)

def xz_rot(pars):
    """ xz-rotation """
    cos_par_0 = cos(pars[0])
    sin_par_0 = sin(pars[0])
    exp_par_1 = exp(1j * pars[1])
    return np.array([[exp_par_1.conjugate() * cos_par_0, -1j*exp_par_1.conjugate() * sin_par_0],
                     [        -1j*exp_par_1 * sin_par_0,                 exp_par_1 * cos_par_0]])

def cr_y(n_qubits, q1, q2, par):
    """ Controlled y-rotation """
    cry_1 = kron_N_A(P0, 2**q1)
    cry_1 = kron_A_N(cry_1, 2**(n_qubits - q1 -1))
    cos_par = cos(par)
    sin_par = sin(par)
    ry = array([[cos_par, -sin_par], [sin_par, cos_par]])
    if q2 > q1:
        op_l = kron_N_A(P1, 2**q1)
        op_l = kron_A_N(op_l, 2**(q2 - q1 - 1))
        op_r = kron_A_N(ry, 2**(n_qubits - q2 - 1))
        cry_2 = kron(op_l, op_r)
    else:
        op_l = kron_N_A(ry, 2**q2)
        op_l = kron_A_N(op_l, 2**(q1 - q2 - 1))
        op_r = kron_A_N(P1, 2**(n_qubits - q1 - 1))
        cry_2 = kron(op_l, op_r)
    return cry_1 + cry_2

def cx(n_qubits, q1, q2):
    """ CX gate """
    cx_1 = reduce(kron, [eye(2**q1), P0, np.eye(2**(n_qubits-q1-1))])
    if q2 > q1:
        cx_2 = [eye(2**q1), P1, eye(2**(q2-q1-1)), X, eye(2**(n_qubits - q2 - 1))]
    else:
        cx_2 = [eye(2**q2), X, eye(2**(q1-q2-1)), P1, eye(2**(n_qubits - q1 - 1))]
    cx_2 = reduce(kron, cx_2)
    return cx_1 + cx_2

def cx_cascade(n_qubits):
    if n_qubits == 1:
        return eye(2)
    else:
        op_l = []
        for q in range(n_qubits - 1):
            op_l.append(cx(n_qubits, q, q + 1))
        if n_qubits > 2:
            op_l.append(cx(n_qubits, n_qubits - 1, 0))
        return reduce(dot, op_l)

def hea_cx_rzrx(n_qubits, n_layers, CXq, pars):
    """ Requires pre-computed chain of CX-gates, CXq=cx_cascade(n_qubits) """
    it = iter(pars)
    op = reduce(kron, [xz_rot([next(it), next(it)]) for q in range(n_qubits)])
    for l in range(n_layers):
        op = CXq@op
        op = reduce(kron, [xz_rot([next(it), next(it)]) for q in range(n_qubits)])@op        
    return op

def hea_cry_rzrx(n_qubits, n_layers, pars):
    it = iter(pars)
    op = reduce(kron, [xz_rot([next(it), next(it)]) for q in range(n_qubits)])
    for l in range(n_layers):
        for q in range(n_qubits - 1):
            op = cr_y(n_qubits, q, q+1, next(it))@op
        op = reduce(kron, [xz_rot([next(it), next(it)]) for q in range(n_qubits)])@op
    return op



# measurers #    
    
def measure_povm(density_matrix, povm, outcomes, n_shots, return_probs=False):
    """
    Measures an obsrvble given as (POVM, outcomes) in a given state with a given number of shots.
    Input: density matrix, observable a dict of POVM-elemets and the corresponding measurement outcomes, number of shots.
    Output: estimated expected value. 
    """
            
    probs = {outcomes[j]: 0 for j in range(len(outcomes))} # a dictionary which will contain probabilities of a certain measuremet outcome; just for a case
    prob_intervals = [] # will contain the probability intervals
    p_sum = 0 # auxiliary variable; must be equal to 1 at the end of the loop below
    for j in range(len(povm)):
        p = trace(dot(density_matrix, povm[j])).real # probability of the outcome E_j
        probs[outcomes[j]] += p # store the probability
        prob_intervals.append( (p_sum, p_sum+p, outcomes[j]) ) # store the tripple (previous_probability, previous_probability + current_probability, projector's_binary_number)
        p_sum += p # this part may be optimized a bit for not obtaining one sum twice
    
    if n_shots == inf:
        expectation_exact = 0
        for outcome, prob in probs.items():
            expectation_exact += outcome * prob 
        return expectation_exact
    
    else:
        counts = {outcomes[j]: 0 for j in range(len(outcomes))} # a dictionary which will contain counts of a certain measuremet outcomes obtained in an 'experiment'
        ps = uniform(0, 1, n_shots) # generate the probabilities of obtaining a specific outcome
        for s in range(n_shots):
            # find which outcome we obtained in the 'experiment' 
            for i in range(len(outcomes) - 1):
                if (ps[s] >= prob_intervals[i][0]) and (ps[s] < prob_intervals[i][1]):
                    counts[prob_intervals[i][2]] += 1 # add one 'click'
                    break
            # treat the last interval properly
            if (ps[s] >= prob_intervals[-1][0]) and (ps[s] <= prob_intervals[-1][1]):
                counts[prob_intervals[-1][2]] += 1 # add one 'click'
        # "experimental" probabilities
        expectation_exper = 0
        for outcome, count in counts.items():
            expectation_exper += outcome * count
        expectation_exper /= n_shots
        
        if return_probs == True:
            return expectation_exper, probs
        else:
            return expectation_exper


### train ###

# to be shared later #


