import numpy as np
from warnings import warn
from math import log
from scipy.optimize import minimize

"""
    aprēķina adaptīvo slieksni pēc gpd sadalījuma

    args:
        q: riska līmenis (false positive rate)
        gamma: gpd formas parametrs
        sigma: mēroga parametrs
        n: pīķu skaits virs sliekšņa
        N: kopējais datu punktu skaits
        t: sākotnējais slieksnis

    return:
        z: gala anomāliju slieksnis
"""
# aprēķina galīgo anomāliju slieksni, balstoties uz pot rezultātiem
def CalcThreshold(
    q,
    gamma,
    sigma,
    n,
    N,
    t,
):
    assert n != 0 # nedrīkst būt tukšs pīķu kopums
    # ja gamma nav nulle, lieto standarta formulu
    if gamma != 0:
        z = t + (sigma / gamma) * (pow(q * N / n, -gamma) - 1)
    else:
        # ja gamma = 0, lieto logaritmisko variantu
        warn("gamma ir 0")
        z = t - sigma * np.log(q * N / n)
    return z




    """
    grimshaw optimizācija gpd parametriem

    ideja:
    - pārvērš 2d optimizāciju par 1d problēmu
    - meklē labāko gamma un sigma

    args:
        peaks: ekstremālās vērtības virs sliekšņa
        threshold: sākotnējais slieksnis
        num_candidates: kandidātu skaits optimizācijai
        epsilon: skaitliskā stabilitāte

    return:
        gamma: formas parametrs
        sigma: mēroga parametrs
    """
def Grimshaw(
        peaks: np.array, 
        threshold: float, 
        num_candidates: int = 10, 
        epsilon: float = 1e-8,
    ):
    min = peaks.min()
    max = peaks.max()
    mean = peaks.mean()
    # pielāgo epsilon, ja dati ir ļoti tuvu robežai
    if abs(-1 / max) < 2 * epsilon:
        epsilon = abs(-1 / max) / num_candidates
    # robežu definēšana kandidātiem
    a = -1 / max + epsilon
    b = 2 * (mean - min) / (mean * min)
    c = 2 * (mean - min) / (min ** 2)
    # meklē kandidātus gamma un sigma
    candidate_gamma = solve(
        function=lambda threshold: function(peaks, threshold), 
        dev_function=lambda threshold: dev_function(peaks, threshold), 
        bounds=(a, -epsilon), 
        num_candidates=num_candidates
    )
    candidate_sigma = solve(
        function=lambda threshold: function(peaks, threshold), 
        dev_function=lambda threshold: dev_function(peaks, threshold), 
        bounds=(b, c), 
        num_candidates=num_candidates
    )
    candidates = np.concatenate([candidate_gamma, candidate_sigma])
    # sākotnējā labākā vērtība
    gamma_best = 0
    sigma_best = mean
    log_likelihood_best = cal_log_likelihood(peaks, gamma_best, sigma_best)
    # pārbauda visus kandidātus
    for candidate in candidates:
        gamma = np.log(1 + candidate * peaks).mean()
        sigma = gamma / candidate
        log_likelihood = cal_log_likelihood(peaks, gamma, sigma)
        if log_likelihood > log_likelihood_best:
            gamma_best = gamma
            sigma_best = sigma
            log_likelihood_best = log_likelihood

    return gamma_best, sigma_best

# palīgfunkcija grimshaw vienādojumam
def function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    return u * v - 1

# atvasinājums
def dev_function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    dev_u = (1 / threshold) * (1 - v)
    dev_v = (1 / threshold) * (-v + np.mean(1 / s ** 2))
    return u * dev_v + v * dev_u

# objektfunkcija optimizācijai
def obj_function(x, function, dev_function):
    m = 0
    n = np.zeros(x.shape)
    for index, item in enumerate(x):
        y = function(item)
        m = m + y ** 2
        n[index] = 2 * y * dev_function(item)
    return m, n

# optimizācijas risinātājs
def solve(function, dev_function, bounds, num_candidates):
    step = (bounds[1] - bounds[0]) / (num_candidates + 1)
    x0 = np.arange(bounds[0] + step, bounds[1], step)
    optimization = minimize(
        lambda x: obj_function(x, function, dev_function), 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        bounds=[bounds]*len(x0)
    )
    x = np.round(optimization.x, decimals=5)
    return np.unique(x)

# log-likelihood funkcija 
def cal_log_likelihood(peaks, gamma, sigma):
    if gamma != 0:
        tau = gamma/sigma
        log_likelihood = -peaks.size * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * peaks)).sum()
    else: 
        log_likelihood = peaks.size * (1 + log(peaks.mean()))
    return log_likelihood
# peak-over-threshold algoritms

    """
    pot algoritms anomāliju sliekšņa noteikšanai

    ideja:
    - izvēlas sākotnējo slieksni
    - ņem pīķus virs tā
    - pielāgo gpd sadalījumu
    - aprēķina gala slieksni
    """
def PeakOverThreshold(
        data: np.array, 
        num_candidates: int,
        risk: float = 1e-2, 
        init_level: float = 0.9, 
        epsilon: float = 1e-8
    ):
    t = SetInitialThreshold(data, init_level)

    # pīķi virs sliekšņa
    y = data[data > t] - t

    # gpd parametri
    gamma, sigma = Grimshaw(
        peaks=y, 
        threshold=t, 
        num_candidates=num_candidates, 
        epsilon=epsilon
    )

    # gala slieksnis
    z = CalcThreshold(
        q = risk,
        gamma = gamma,
        sigma = sigma,
        n = y.size,
        N = data.size,
        t = t,
    )

    return z, t
    
# sākotnējā sliekšņa izvēle 
def SetInitialThreshold(x: np.array, init_level: float = 0.98):
    t = np.sort(x)[int(init_level * len(x))]
    return t


    """
    Paņemts no: 
        Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
        Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
        Discovery and Data Mining. 2017.
    """