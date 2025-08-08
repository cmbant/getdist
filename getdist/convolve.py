import numpy as np
from scipy import fftpack

# numbers of the form 2^n3^m5^r, even only and r<=1
fastFFT = np.array(
    [
        2,
        4,
        6,
        8,
        10,
        12,
        16,
        20,
        24,
        32,
        40,
        48,
        64,
        80,
        96,
        128,
        144,
        160,
        192,
        256,
        288,
        320,
        384,
        432,
        480,
        512,
        576,
        640,
        720,
        768,
        864,
        960,
        1024,
        1152,
        1280,
        1440,
        1536,
        1728,
        1920,
        2048,
        2304,
        2560,
        2880,
        3072,
        3456,
        3840,
        4096,
        4608,
        5120,
        5760,
        6144,
        6912,
        7680,
        8192,
        9216,
        10240,
        11520,
        12288,
        13824,
        15360,
        16384,
        18432,
        20480,
        23040,
        24576,
        27648,
        30720,
        32768,
        36864,
        40960,
        46080,
        49152,
        55296,
        61440,
        65536,
        73728,
        81920,
        92160,
        98304,
        110592,
        122880,
        131072,
        147456,
        163840,
        184320,
        196608,
        221184,
        245760,
        262144,
        294912,
        327680,
        368640,
        393216,
        442368,
        491520,
        524288,
        589824,
        655360,
        737280,
        786432,
        884736,
        983040,
        1048576,
        1179648,
        1310720,
        1474560,
        1572864,
        1769472,
        1966080,
        2097152,
        2359296,
        2621440,
        2949120,
        3145728,
        3538944,
        3932160,
        4194304,
        4718592,
        5242880,
        5898240,
        6291456,
        7077888,
        7864320,
        8388608,
        9437184,
        10485760,
        11796480,
        12582912,
        14155776,
        15728640,
        16777216,
        18874368,
        20971520,
        23592960,
        25165824,
        28311552,
        31457280,
        33554432,
        37748736,
        41943040,
        47185920,
        50331648,
        56623104,
        62914560,
        67108864,
        75497472,
        83886080,
        94371840,
        100663296,
        113246208,
        125829120,
        134217728,
        150994944,
        167772160,
        188743680,
        201326592,
        226492416,
        234881024,
        251658240,
        268435456,
        301989888,
        335544320,
        377487360,
        402653184,
        452984832,
        503316480,
        536870912,
        603979776,
        671088640,
        754974720,
        805306368,
        905969664,
        1006632960,
        1207959552,
        1342177280,
        1358954496,
        1509949440,
        1610612736,
        1811939328,
        2013265920,
    ],
    dtype=int,
)


def nearestFFTnumber(x):
    return np.maximum(x, fastFFT[np.searchsorted(fastFFT, x)])


def convolve1D(x, y, mode, largest_size=0, cache=None, cache_args=(1, 2)):
    if mode == "periodic":
        return convolve1D_periodic(x, y, cache, cache_args)
    if min(x.shape[0], y.shape[0]) > 1000:
        return convolveFFT(x, y, mode, largest_size=largest_size, cache=cache, cache_args=cache_args)
    else:
        return np.convolve(x, y, mode)


def convolve2D(x, y, mode, largest_size=0, cache=None, cache_args=(1, 2)):
    if mode in ("periodic", "periodic_both"):
        return convolve2D_periodic(x, y, cache, cache_args, periodic_x=True, periodic_y=True)
    elif mode == "periodic_x":
        return convolve2D_periodic(x, y, cache, cache_args, periodic_x=True, periodic_y=False)
    elif mode == "periodic_y":
        return convolve2D_periodic(x, y, cache, cache_args, periodic_x=False, periodic_y=True)
    return convolveFFTn(x, y, mode, largest_size, cache, cache_args=cache_args)


def convolve2D_periodic(x, y, cache=None, cache_args=(1, 2), periodic_x=True, periodic_y=True):
    """
    2D convolution with periodic boundary conditions in specified dimensions.

    This implements periodic convolution by extending the 1D approach to 2D.
    For periodic dimensions, the input is made circular by averaging boundary values.

    :param x: 2D input array (data)
    :param y: 2D kernel array
    :param cache: optional cache for FFTs
    :param cache_args: which arrays to cache (1=x, 2=y)
    :param periodic_x: whether to use periodic boundaries in x (second) dimension
    :param periodic_y: whether to use periodic boundaries in y (first) dimension
    :return: convolved array of same size as x
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("convolve2D_periodic requires 2D arrays")

    ny, nx = x.shape
    ky, kx = y.shape

    # Create the circular version of x based on periodicity
    if periodic_x and periodic_y:
        # Both dimensions periodic
        x_circ = x[:-1, :-1].copy()  # Remove last row and column
        x_circ[0, :] += x[-1, :-1]  # Add last row to first row
        x_circ[:, 0] += x[:-1, -1]  # Add last column to first column
        x_circ[0, 0] += x[-1, -1]  # Add corner element
        N_y, N_x = x_circ.shape

    elif periodic_x and not periodic_y:
        # Only x dimension periodic
        x_circ = x[:, :-1].copy()  # Remove last column
        x_circ[:, 0] += x[:, -1]  # Add last column to first column
        N_y, N_x = x_circ.shape

    elif periodic_y and not periodic_x:
        # Only y dimension periodic
        x_circ = x[:-1, :].copy()  # Remove last row
        x_circ[0, :] += x[-1, :]  # Add last row to first row
        N_y, N_x = x_circ.shape

    else:
        # Neither periodic (shouldn't happen, but handle gracefully)
        return convolveFFTn(x, y, "same")

    # Prepare kernel for circular convolution
    hpad = np.zeros((N_y, N_x), dtype=float)
    hpad[:ky, :kx] = y
    # Roll kernel to center it properly for convolution
    hpad = np.roll(hpad, -(ky // 2), axis=0)
    hpad = np.roll(hpad, -(kx // 2), axis=1)

    # Cache keys
    if cache is not None:
        if 1 in cache_args:
            key_x = ("2D_circ", N_y, N_x, periodic_x, periodic_y, id(x))
        if 2 in cache_args:
            key_y = ("2D_kernel", N_y, N_x, id(y))

    # Get or compute FFT of kernel
    yfft = None
    if cache is not None and 2 in cache_args:
        yfft = cache.get(key_y)
    if yfft is None:
        yfft = np.fft.rfftn(hpad)
        if cache is not None and 2 in cache_args:
            cache[key_y] = yfft

    # Get or compute FFT of data
    xfft = None
    if cache is not None and 1 in cache_args:
        xfft = cache.get(key_x)
    if xfft is None:
        xfft = np.fft.rfftn(x_circ)
        if cache is not None and 1 in cache_args:
            cache[key_x] = xfft

    # Perform convolution in frequency domain
    result_fft = xfft * yfft
    result = np.fft.irfftn(result_fft, (N_y, N_x), axes=(0, 1))

    # Extend result back to original size for periodic dimensions
    if periodic_x and periodic_y:
        # Both periodic: append first row/column as last
        final_result = np.empty((ny, nx))
        final_result[:-1, :-1] = result
        final_result[-1, :-1] = result[0, :]  # Last row = first row
        final_result[:-1, -1] = result[:, 0]  # Last column = first column
        final_result[-1, -1] = result[0, 0]  # Corner element
        return final_result

    elif periodic_x and not periodic_y:
        # Only x periodic: append first column as last
        final_result = np.empty((ny, nx))
        final_result[:, :-1] = result
        final_result[:, -1] = result[:, 0]  # Last column = first column
        return final_result

    elif periodic_y and not periodic_x:
        # Only y periodic: append first row as last
        final_result = np.empty((ny, nx))
        final_result[:-1, :] = result
        final_result[-1, :] = result[0, :]  # Last row = first row
        return final_result

    else:
        # Neither periodic (fallback)
        return result


def convolve1D_periodic(x, y, cache=None, cache_args=(1, 2)):
    """
    Circular (periodic) 1D convolution of x with kernel y. Returns same-length result.
    Input x is assumed to have equal boundaries with average half bin count, so
    periodic bins formed by adding last bin to first.
    Returns array of same size of x, with last bins equal.

    Notes:
    - Uses FFT of size N=len(x) for true circular convolution. Kernel y is centered
      (rolled by -(len(y)//2)) before FFT so that the kernel's central element aligns
      with the current bin, matching the usual "same" semantics used elsewhere.
    - Basic caching supported via provided cache dict.
    """
    x_circ = x[:-1].copy()
    x_circ[0] += x[-1]
    N = x_circ.shape[0]
    M = y.shape[0]
    # Prepare padded, centered kernel for circular convolution
    if cache is not None and 2 in cache_args:
        key_y = ("circ", N, M, id(y))
        yfft = cache.get(key_y)
    else:
        yfft = None
    if yfft is None:
        hpad = np.zeros(N, dtype=float)
        hpad[:M] = y
        # roll so that center element aligns with current index in convolution
        hpad = np.roll(hpad, -(M // 2))
        yfft = np.fft.rfft(hpad)
        if cache is not None and 2 in cache_args:
            cache[key_y] = yfft
    if cache is not None and 1 in cache_args:
        key_x = ("circ", N, id(x))
        xfft = cache.get(key_x)
    else:
        xfft = None
    if xfft is None:
        xfft = np.fft.rfft(x_circ)
        if cache is not None and 1 in cache_args:
            cache[key_x] = xfft
    res = np.fft.irfft(xfft * yfft, n=N)
    return np.append(res, res[0])


# noinspection PyUnboundLocalVariable
def convolveFFT(x, y, mode="same", yfft=None, xfft=None, largest_size=0, cache=None, cache_args=(1, 2)):
    """
    convolution of x with y; ffts can be cached.
    Be careful with caches, key uses id which can be reused for different object if object is freed.
    """
    size = x.size + y.size - 1
    fsize = nearestFFTnumber(np.maximum(largest_size, size))

    if yfft is None:
        if cache is not None and 2 in cache_args:
            key = (fsize, y.size, id(y))
            yfft = cache.get(key)
        if yfft is None:
            yfft = np.fft.rfft(y, fsize)
            if cache is not None and 2 in cache_args:
                cache[key] = yfft
    if xfft is None:
        if cache is not None and 1 in cache_args:
            key = (fsize, x.size, id(x))
            xfft = cache.get(key)
        if xfft is None:
            xfft = np.fft.rfft(x, fsize)
            if cache is not None and 1 in cache_args:
                cache[key] = xfft
    res = np.fft.irfft(xfft * yfft)[0:size]
    if mode == "same":
        return res[(y.size - 1) // 2 : (y.size - 1) // 2 + x.size]
    elif mode == "full":
        return res
    elif mode == "valid":
        return res[y.size - 1 : x.size]


# noinspection PyUnboundLocalVariable
def convolveFFTn(in1, in2, mode="same", largest_size=0, cache=None, yfft=None, xfft=None, cache_args=(1, 2)):
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    size = s1 + s2 - 1
    fsize = nearestFFTnumber(np.maximum(largest_size, size))
    axes = list(range(-len(fsize), 0))
    if cache is not None:
        if xfft is None and 1 in cache_args:
            key = (tuple(fsize), tuple(in1.shape), id(in1))
            xfft = cache.get(key)
        if yfft is None and 2 in cache_args:
            key2 = (tuple(fsize), tuple(in2.shape), id(in2))
            yfft = cache.get(key2)
    if xfft is None:
        xfft = np.fft.rfftn(in1, fsize, axes)

        if cache is not None and 1 in cache_args:
            cache[key] = xfft
    if yfft is None:
        yfft = np.fft.rfftn(in2, fsize, axes)
        if cache is not None and 2 in cache_args:
            cache[key2] = yfft

    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = np.fft.irfftn(xfft * yfft, fsize, axes)[fslice]

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    startind = (np.array(arr.shape) - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def autoCorrelation(x, n=None, normalized=True, start_index=0):
    """
    Calculates auto-correlation of x, result[k] = sum_{i=0}^{n-k} x_i x_{i+k}/(n-k)
    if normalized=True, divides by the variance (for if start_index=0, first number is one)
    """
    result = autoConvolve(x - x.mean(), n, normalize=True)
    if normalized:
        result /= result[0]
    return result[start_index:]


def autoConvolve(x, n=None, normalize=True):
    """
    Calculates auto-covariance of x, result[k] = sum_i x_i x_{i+k}
    n is maximum size to return (k = 0..n-1)
    if normalize=True then normalize convolution by the number of terms for each k
    (can input x-mean(x) and divide result by variance to get auto correlation)
    """
    s = nearestFFTnumber(2 * x.size)
    #    yt = np.fft.rfft(x, s)
    #    yt *= yt.conj()
    #    return np.fft.irfft(yt)[0:x.size]
    xt = fftpack.rfft(x, s)
    auto = np.empty((xt.size // 2) + 1)
    auto[0] = xt[0] ** 2
    auto[-1] = xt[-1] ** 2
    auto[1:-1] = xt[1:-2:2] ** 2 + xt[2:-1:2] ** 2
    n = n or x.size
    res = fftpack.idct(auto, type=1)[0:n] / s
    if normalize:
        res /= np.arange(x.size, x.size - n, -1)
    return res


# noinspection PyUnboundLocalVariable
def convolveGaussianDCT(x, sigma, pad_sigma=4, mode="same", cache=None):
    """
    1D convolution of x with Gaussian of width sigma pixels
    If pad_sigma>0, pads ends with zeros by int(pad_sigma*sigma) pixels
    Otherwise does unpadded fast cosine transform, hence reflection from the ends
    """

    fill = int(pad_sigma * sigma)
    actual_size = x.size + fill * 2
    if fill > 0:
        s = nearestFFTnumber(actual_size)
        fill2 = s - x.size - fill
        padded_x = np.pad(x, (fill, fill2), mode="constant")
    else:
        padded_x = x

    s = padded_x.size
    hnorm = sigma / float(s)
    gauss = cache.get((s, hnorm)) if cache is not None else None
    if gauss is None:
        gauss = np.exp(-((np.arange(0, s) * (np.pi * hnorm)) ** 2) / 2.0)
        cache[(s, hnorm)] = gauss
    res = fftpack.idct(fftpack.dct(padded_x, overwrite_x=fill > 0) * gauss, overwrite_x=fill > 0) / (2 * s)
    if fill == 0:
        return res
    elif mode == "same":
        return res[fill:-fill2]
    elif mode == "valid":
        return res[fill * 2 : -fill2 - fill]
    else:
        raise ValueError("mode not supported for convolveGaussianDCT")


def convolveGaussian(x, sigma, sigma_range=4, cache=None):
    """
    1D convolution of x with Gaussian of width sigma pixels
    x_max = int(sigma_range*sigma) the zero padding range at ends
    This uses periodic boundary conditions, and mode = 'same'
    This is the fastest fft version
    """
    fill = int(sigma_range * sigma)
    actual_size = x.size + 2 * fill
    if fill > 0:
        s = nearestFFTnumber(actual_size)
    else:
        s = actual_size
    gauss = None if cache is None else cache.get((fill, actual_size, sigma))
    if gauss is None:
        hnorm = sigma / float(s)
        ps = np.arange(1, s + 1) // 2
        gauss = np.exp(-((ps * (np.pi * hnorm)) ** 2) * 2)
        if cache is not None:
            cache[(fill, actual_size, sigma)] = gauss
    res = fftpack.irfft(fftpack.rfft(x, s) * gauss, s)
    return res[: x.size]


def convolveGaussianTrunc(x, sigma, sigma_range=4, mode="same", cache=None):
    """
    1D convolution of x with Gaussian of width sigma pixels
    x_max = int(sigma_range*sigma) determines the finite support (in pixels) of the truncated gaussian
    This uses normalized finite range approximation to Gaussian
    """
    fill = int(sigma_range * sigma)
    actual_size = x.size + 2 * fill
    s = nearestFFTnumber(actual_size)
    gauss = None if cache is None else cache.get((fill, actual_size, sigma))
    if gauss is None:
        points = np.arange(-fill, fill + 1)
        win = np.exp(-((points / sigma) ** 2) / 2.0)
        win /= np.sum(win)
        gauss = np.fft.rfft(win, s)
        if cache is not None:
            cache[(fill, actual_size, sigma)] = gauss
    res = np.fft.irfft(np.fft.rfft(x, s) * gauss, s)[:actual_size]
    if mode == "same":
        return res[fill:-fill]
    elif mode == "full":
        return res
    elif mode == "valid":
        return res[2 * fill : -2 * fill]


def dct2d(a):
    return fftpack.dct(fftpack.dct(a, axis=0), axis=1)


def idct2d(a):
    return fftpack.idct(fftpack.idct(a, axis=1), axis=0)
