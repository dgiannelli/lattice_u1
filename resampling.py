import numpy as np

def binning(f, B=50):
    f = np.array(f)
    N = f.size
    N -= N%B
    f = f[:N]
    bsize = N//B
    f_b = np.empty(B)
    idx = np.arange(N)
    for b in range(B):
        idx_b = np.logical_and(idx>=b*bsize,idx<(b+1)*bsize)
        f_b[b] = f[idx_b].mean()
    return f_b.mean(), np.sqrt(f_b.var()/(B-1))

def jackknife(f, theta, B=50):
    f = np.array(f)
    N = f.size
    N -= N%B
    f = f[:N]
    bsize = N//B
    theta_jack_b = np.empty(B)
    idx = np.arange(N)
    for b in range(B):
        idx_jack_b = np.logical_or(idx<b*bsize,idx>=(b+1)*bsize)
        theta_jack_b[b] = theta(f[idx_jack_b])
    return theta_jack_b.mean(), np.sqrt((B-1)*theta_jack_b.var())

def tau(f, B=50):
    s_f_mean_sq = f.var()/(f.size-1)
    s_f_b_mean_sq = binning(f,B)[1]**2
    return 0.5*(s_f_b_mean_sq/s_f_mean_sq-1.)

def tau_jack(f, B=50):
    return jackknife(f,tau,B)

