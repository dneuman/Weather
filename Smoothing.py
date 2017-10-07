# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd

def Triangle(size, clip=1.0):
    """Return a triangle weighting window, with optional clipping.

    Paramters
    ---------
    size : int
        Length of the returned weights

    **Optionals**

    clip : float
        Any weight above `clip` will be forced to `clip`.
    """
    w = np.bartlett(size+2)
    w = w[1:-1]  # remove zeros at endpoints
    w = np.array([min(clip, i) for i in w])
    return (w / max(w))

def Hanning(size):
    w = np.hanning(size+2)
    w = np.array(w[1:-1])  # remove zeros at endpoints
    return (w / max(w))

def Lowess(data, f=2./3., pts=None, itn=3, order=1, pad=True):
    """Fits a nonparametric regression curve to a scatterplot.

    Parameters
    ----------
    data : pandas.Series
        Data points in the scatterplot. The
        function returns the estimated (smooth) values of y.

    **Optionals**

    f : float
        The fraction of the data set to use for smoothing. A
        larger value for f will result in a smoother curve.
    pts : int
        The explicit number of data points to be used for
        smoothing instead of f.
    itn : int
        The number of robustifying iterations. The function will run
        faster with a smaller number of iterations.
    order : int
        The order of the polynomial used for fitting. Defaults to 1
        (straight line). Values < 1 are made 1. Larger values should be
        chosen based on shape of data (# of peaks and valleys + 1)
    pad : Boolean
        If True, will pad the data and the beginning and end by the amount
        of data used for smoothing (f*n or pts). This may provide better
        tracking of data at the ends.

    Returns
    -------
    pandas.Series containing the smoothed data.
    """
    # Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
    #            original
    #          Dan Neuman <https://github.com/dneuman>
    #            converted to Pandas series and extended to polynomials
    #            and added padding option.
    # License: BSD (3-clause)

    n = len(data)
    if pts is None:
        f = np.min([f, 1.0])
        r = int(np.ceil(f * n))
    else:  # allow use of number of points to determine smoothing
        r = int(np.min([pts, n]))
    r = min([r, n-1])
    order = max([1, order])
    if pad:
        tx = np.array(data.index, dtype=float)
        ty = data.values
        x = np.zeros(len(data)+2*r)
        y = np.zeros(len(data)+2*r)
        # pad data as a reflection of original data. eg use indexes:
        # 2, 1, 0, 1, 2, 3, 4, 5 and
        # n-3, n-2, n-1, n-2, n-3, n-4
        # x values keep the distances between them but go backwards at end and
        # forward at the end. y values are not changed.
        for i in range(r):  # pad beginning
            x[i] = tx[0] - (tx[r-i] - tx[0])
            y[i] = ty[r-i]
        for i in range(n):  # add actual data
            x[i+r] = tx[i]
            y[i+r] = ty[i]
        for i in range(r):  # pad end
            x[i+r+n] = tx[n-1] + (tx[n-1] - tx[n-2-i])
            y[i+r+n] = ty[n-2-i]
        n = len(y)
    else:
        x = np.array(data.index, dtype=float)
        y = data.values
    # condition x-values to be between 0 and 1 to reduce errors in linalg
    x = x - x.min()
    x = x / x.max()
    # Create matrix of 1, x, x**2, x**3, etc, by row
    xm = np.array([x**j for j in range(order+1)])
    # Create weight matrix, one column per data point
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    # Set up output
    yEst = np.zeros(n)
    delta = np.ones(n)  # Additional weights for iterations
    for iteration in range(itn):
        for i in range(n):
            weights = delta * w[:, i]
            xw = np.array([weights * x**j for j in range(order+1)])
            b = xw.dot(y)
            a = xw.dot(xm.T)
            beta = np.linalg.solve(a, b)
            yEst[i] = sum([beta[j] * x[i]**j for j in range(order+1)])
        # Set up weights to reduce effect of outlier points on next iteration
        residuals = y - yEst
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    if pad:
        n = len(data)
        return pd.Series(yEst[r:n+r], index=data.index, name='Trend')
    else:
        return pd.Series(yEst, index=data.index, name='Trend')

def WeightedMovingAverage(s, size, pad=True, winType=Hanning, wts=None):
    """Apply a weighted moving average on the supplied series.

    Parameters
    ----------
    s : pandas.Series
        data to be averaged
    size : integer
        how wide a window to use
    pad : Boolean (optional, default = True)
        flag determining whether to pad beginning and end of data with a
        weighted average of the last `size` points. This provides better
        smoothing at the beginning and end of the line, but it tends to have
        zero slope.
    winType : Function (optional, default = Hanning)
        Window function that takes an integer (window size) and returns a list
        of weights to be applied to the data. The default is Hanning, a
        weighted cosine with non-zero endpoints. Other possible windows are:

        * np.bartlett (triangular with endpoints of 0)
        * np.blackman (3 cosines creating taper)
        * np.hamming (weighted cosine)
        * np.hanning (weighted cosine with endpoints of 0)
        * Triangle (triangle with non-zero enpoints, and option to
                    clip top of triangle)
    wts : list (optional, default = None)
        List of weights to use. `size` becomes the length of wts. Use this
        option to provide a custom weighting function. The length of wts
        should be odd, but this is not enforced.
    Returns
    -------
    Pandas Series containing smoothed data

    Notes
    -----
    Defaults to using a triangular window for weights, centered on
    each point. For points near the beginning or end of data, special
    processing is required that isn't in built-in functions.
    """
    def SetLimits(i, hw):
        # i: current data location where window is centred
        # hw: half window width
        ds = max(0, (i-hw))         # data start
        de = min(n-1, (i+hw)) # data end
        ws = hw - (i - ds)          # window start
        we = hw + (de - i)          # window end
        return ds, de, ws, we

    if type(wts) == type(None):
        size += (size+1) % 2  # make odd
        window = winType(size)
        window /= window.sum()  # normalize window
    else:
        window = wts / wts.sum()
        size = len(wts)
    n = len(s)
    hw = int(size / 2) # half window width
    # convolve has boundary effects when there is no overlap with the window
    # Begining and end of 'a' must be adjusted to compensate.
    # np.average() effectively scales the weights for the different sizes.
    if pad: # pad the data with reflected values
        # create padded beginning
        y = np.zeros(n+2*hw)
        for i in range(hw):
            y[i] = s.iloc[hw-i]
        for i in range(hw):
            y[i+n+hw] = s.iloc[n-i-1]
        for i in range(n):
            y[i+hw] = s.iloc[i]
        yc = np.convolve(y, window, mode='same')
        a = pd.Series(yc[hw:n+hw],
                      index=s.index,
                      name=s.name)
    else: # clip window as available data decreases
        a = pd.Series(np.convolve(s, window, mode='same'),
                      index=s.index,
                      name=s.name)
        for i in range(hw+1):  # fix the start
            (ds, de, ws, we) = SetLimits(i, hw)
            a.iloc[i] = np.average(s.iloc[ds:de], weights=window[ws:we])
        for i in range(n-hw-1, n):  # fix the end
            (ds, de, ws, we) = SetLimits(i, hw)
            a.iloc[i] = np.average(s.iloc[ds:de], weights=window[ws:we])
    return a

def SSA(s, m, rtnRC=1, pad=True):
    """Implement Singular Spectrum Analysis for pandas Series

    Parameters
    ----------
    s : pandas.Series
        Input data, in series or single columns of a data frame. Any necessary
        normalization (e.g. for anomolies from baseline) should be done.
    m : int
        Order or number of time lags to calculate over. A larger number gives
        more smoothing in the first returned column.

    **Optionals**

    rtnRC : int
        Number of reconstructed principles to return. Set to None to get all
        of them. Most smoothing is done in first returned column, but other
        columns may be useful to see periodicities.
    pad : Boolean
        Flad to pad data and principle components instead of using zeros.
        Reflected data is used, mirrored around the end point near the zeros.

    Returns
    -------
    pandas.DataFrame containing the reconstructed principles (or just the first
    one if allRC is True).

    Notes
    -----
    Computes the first m principle components (PCs) using Singular Spectrum
    Analysis. Most of the trend information is in the first reconstructed PC
    (RC), so the function returns just the first RC by default. This RC will
    look like smoothed data, and the amount of smoothing is determined by how
    large `m` is. Note that padding is added to prevent a drop towards 0 at
    beginning and end.

    Examples
    --------
    from:
    http://environnement.ens.fr/IMG/file/DavidPDF/SSA_beginners_guide_v9.pdf
    ::

        %precision 2
        import pandas as pd
        import numpy as np
        import matplotlib as plt

        y = [1.0135518, - 0.7113242, - 0.3906069, 1.565203, 0.0439317,
             - 1.1656093, 1.0701692, 1.0825379, - 1.2239744, - 0.0321446,
             1.1815997, - 1.4969448, - 0.7455299, 1.0973884, - 0.2188716,
             - 1.0719573, 0.9922009, 0.4374216, - 1.6880219, 0.2609807]

        rc = SSA(pd.Series(y), 4, allRC=None, pad=False)
        plt.plot(rc)
        plt.show()

        rc[0].values.flatten()

        array([ 0.3 , -0.31, -0.33,  0.82, -0.06, -0.82,  0.54,  0.53, -0.88,
                0.07,  0.83, -0.66, -0.36,  0.83, -0.18, -0.72,  0.63,  0.23,
               -0.68,  0.24])

    """
    n = len(s)
    y = np.array(s.values)
    mr = range(m)
    avg = y[int(n*0.9):].mean()  # use last 10% average to pad
    ys = np.ones((n,m)) * avg    # time shifted y-values
    for i in mr:
        ys[:n-i,i] = y[i:]
        if pad: ys[n-i:,i] = y[n-2:n-i-2:-1] # reflect end data
    # get autocorrelation at first `order` time lags
    cor = np.correlate(y, y, mode='full')[n-1:n-1+m]/n
    # make toeplitz matrix (diagonal, symmetric)
    c = np.array([[cor[abs(i-j)] for i in mr] for j in mr])
    # get eigenvalues and eigenvectors
    lam, rho = np.linalg.eig(c)
    pc = ys.dot(rho)  # principle components
    # reconstruct the components in proper time frame
    if rtnRC is None:
        desired = m
    else:
        desired = min(m, rtnRC)
    rc = np.zeros((n, desired))
    for j in range(desired):
        z = np.zeros((n, m))
        for i in mr:  # make time shifted principle component matrix
            z[i:,i] = pc[:n-i, j]
            if pad: z[:i,i] = pc[i:0:-1, j]  # reflect beginning data
        rc[:,j] = z.dot(rho[:, j]) / m
    return pd.DataFrame(rc, index=s.index)

