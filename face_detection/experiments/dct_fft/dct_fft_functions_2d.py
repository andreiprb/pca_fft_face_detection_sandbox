import numpy as np

def fft(x: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    if N <= 1:
        return x
    if N % 2 != 0:
        raise ValueError("Length of input must be a power of 2.")
    even = fft(x[::2])
    odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([even + factor * odd, even - factor * odd])

def ifft(X: np.ndarray) -> np.ndarray:
    N = X.shape[0]
    if N <= 1:
        return X
    if N % 2 != 0:
        raise ValueError("Length of input must be a power of 2.")
    X_conj = np.conjugate(X)
    time_domain = fft(X_conj)
    return np.conjugate(time_domain) / N

def dct(x: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    basis = np.cos(np.pi / N * (n + 0.5) * k)
    result = np.dot(basis, x)
    result[0] *= np.sqrt(1 / N)
    result[1:] *= np.sqrt(2 / N)
    return result

def idct(X: np.ndarray) -> np.ndarray:
    N = X.shape[0]
    k = np.arange(N)
    n = k.reshape((N, 1))
    basis = np.cos(np.pi / N * (n + 0.5) * k)
    X_scaled = X.copy()
    X_scaled[0] *= np.sqrt(1 / N)
    X_scaled[1:] *= np.sqrt(2 / N)
    return np.dot(basis, X_scaled)

def fft2(x: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(fft, axis=0, arr=np.apply_along_axis(fft, axis=1, arr=x))

def ifft2(X: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(ifft, axis=0, arr=np.apply_along_axis(ifft, axis=1, arr=X))

def dct2(x: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(dct, axis=0, arr=np.apply_along_axis(dct, axis=1, arr=x))

def idct2(X: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(idct, axis=0, arr=np.apply_along_axis(idct, axis=1, arr=X))
