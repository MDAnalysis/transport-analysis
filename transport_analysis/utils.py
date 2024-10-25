import numpy as np

def autocorrFFT(x):
    """
    Calculates the autocorrelation function using the fast Fourier transform.

    :param x: array[float], function on which to compute autocorrelation function
    :return: acf: array[float], autocorrelation function
    """
    N= len(x)
    F = np.fft.fft(x, n=2*N)  
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   
    n=N*np.ones(N)-np.arange(0,N) 
    acf = res/n
    return acf

def msd_fft_1d(r):
    """Calculates mean square displacement of the array r using the fast Fourier transform."""
    # Algorithm based on https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
    N = len(r)
    D = np.square(r)
    D = np.append(D, 0)
    S2 = autocorrFFT(r)
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2 * S2


def cross_corr(x, y):
    N = len(x)
    F1 = np.fft.fft(
        x, n=2 ** (N * 2 - 1).bit_length()
    )  # 2*N because of zero-padding, use next highest power of 2
    F2 = np.fft.fft(y, n=2 ** (N * 2 - 1).bit_length())
    PSD = F1 * F2.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
    return res / n

def msd_variance_1d(r, msd):

    # compute A1, recursive relation with D = r^4
    N = len(r)
    D = r**4
    D = np.append(D, 0)
    Q = 2 * D.sum()
    A1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        A1[m] = Q / (N - m)

    # compute A2, autocorrelation of r^2
    A2 = cross_corr(r**2, r**2)

    # compute A3 and A4, cross correlations of r and r^3
    A3 = cross_corr(r, r**3)
    A4 = cross_corr(r**3, r)

    var_x = A1 + 6*A2 - 4*A3 - 4*A4 - msd**2
    n_minus_m = N * np.ones(N) - np.arange(0, N)   # divide by (N-m)^2 (Var[E[X]] = Var[X]/n)

    return var_x/n_minus_m

def average_directions(r, dim):
    if dim == 3:
        return (r[:,0] + r[:,1] + r[:,2])/3
    if dim == 2:
        return (r[:,0] + r[:,1])/2
    if dim == 1:
        return r[:,0]

