#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Texture Module
**************

Module to add simple textures to matplotlib charts

Based on the `Matplotlib agg_filter Demo
<https://matplotlib.org/gallery/misc/demo_agg_filter.html>`_

Usage
-----

Create a Texture object with the desired style and options, then supply
this object to the ``agg_filter`` keyword in matplotlib methods. Running
``Texture``, or ``texture_pie`` will create a sample chart with representative
textues.::

    from Texture import Texture
    import matplotlib.pyplot as plt

    filt1 = Texture('noise')
    filt2 = Texture('hash', block=2, light=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.fill_between([0,1], [1,1], agg_filter=filt1)
    ax.fill_between([1,2], [1,1], agg_filter=filt2)
    plt.show()

See ``texture_pie(ax)`` for an example of using multiple texture filters in
one chart.

Routines
--------

    * texture_pie(ax) - Demonstrate example textures
    * shadow_line(ax) - Demonstrate shadows
    * test(style) - test a style without drawing (for debugging)

Requires
--------
    * Python 3.6 (tested on Anaconda 3.6.1)
    * Matplotlib 2.2.2

"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from matplotlib.colors import LightSource

class Texture(object):
    """Simple Matplotlib agg_filter textures

    Usage
    -----

    Create a Texture object with the desired options, then pass this
    object to the agg_filter keyword in matplotlib routines.

    Example
    -------
    ::

        from Texture import Texture
        import matplotlib.pyplot as plt

        filt = Texture('noise', block=4, light=True)
        shadow = Texture('shadow', pad=.5, offset=(-.25, .25))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.fill_between([0,1], [1,1], agg_filter=shadow)  # create shadow first
        ax.fill_between([0,1], [1,1], agg_filter=filt)
        plt.show()

    Parameters
    ----------

    style : string default 'noise'
        One of the following styles

        * 'noise' - strictly random points
        * 'hash' - small lines oriented randomly
        * 'shadow' - create background shadow
    block : int default 1
        Size of the points, in pixels. Larger values give coarser textures.
    light : bool default False
        Use noise as a depth map and light the surface. Works best when
        ``block`` is at least 2.

    Other Options
    -------------

    'noise' : options to set amount and darkness of points
        * prob : float (0.0 - 1.0) - probability a point will be changed
        * dark : float (0.0 - 1.0) - how dark to make random points. Small
          values are better.

    'hash' : options to set spacing and darkness of lines
        * space : int default 4 - roughly how far apart lines are
        * dark : float (0.0 - 1.0) - how dark to make random points. Small
          values are better.

    light : options passed to the matplotlib.colors.LightSource.rgb_shade method.
        * fraction : float (0.0 - 1.0) alpha value
        * vert_exag : float amount of vertical exaggeration.
        * blend_mode : string ['soft'|'overlay'|'hsv'] blending mode

    shadow : options to define the shadow. Supplied object becomes the shadow.
        * direction : ['up'|'all'] Use 'up' for upward shadows
        * pad : float 0.5 How large to make the shadow (screen inches)
        * alpha : float 0.5 How dark to make the shadow
        * color : (0, 0, 0) Color to use for shadow
        * cut_figure : False Remove shape from shadow (use if main object
            is semi-transparent).
        * offset : (0, 0) Where to offset shadow (screen inches)
        * win : ['gaussian'|'hanning'] Shadow type. 'hanning' gives more
            spread-out shadow, 'gaussian' is more realistic.

    """

    def __init__(self, style='noise', **kwargs):
        """Initialize object with values to be used when called.
        """
        self.style = style
        if style == 'shadow':
            self.direction = kwargs.get('direction', 'up')
            self.pad = kwargs.get('pad', .5)
            self.alpha = kwargs.get('alpha', .5)
            self.color = kwargs.get('color', (0, 0, 0))
            self.cut_figure = kwargs.get('cut_figure', False)
            self.offset = kwargs.get('offset', (0, 0))
            self.win = kwargs.get('win', 'gaussian')
            return
        self.block = kwargs.get('block', 1)
        self.light = kwargs.get('light', False)
        if self.light:
            prob = kwargs.get('prob', .15)
            dark = kwargs.get('dark', .15)
            self.vert = kwargs.get('vert_exag', 2.)
            self.blend = kwargs.get('blend_mode', 'soft')
            self.fraction = kwargs.get('fraction', .5)
            self.light_source = LightSource()
        else:
            prob = kwargs.get('prob', .2)
            dark = kwargs.get('dark', .02)
        if style == 'hash':
            # hash spreads out points somewhat evenly, but random within
            # spaces. Up to 2 points are chosen per space.
            # Set up arrays and their probabilities
            if not self.light: dark = kwargs.get('dark', .04)
            self.dark = dark
            self.space = kwargs.get('space', 4)
            self.frac = kwargs.get('frac', .8)  # fraction of spaces to fill
            self.clip = (1. - dark)**3  # clip limit (max darkness)
            # make list of directions and pairs of directions.
            # Directions are tuples of (dtn, axis) for np.roll
            dtns = [(-1, 0), (-1, 1), (1, 0), (1, 1)]  # directions
            self.pairs = []  # list of pairs of directions
            for i in range(3):
                for j in range(i+1,4):
                    self.pairs.append([dtns[i], dtns[j]])
        else:
            self.choice = list(1. - np.array([1,2,3]) * dark) + [1.]
            self.p = 3 * [prob] + [1. - 3 * prob]
        self.kwargs = kwargs

    def __call__(self, im, dpi=100):
        """Main call routine, which passes image to other routines for
           processing.
        """
        if self.style == 'noise': return self._noise(im), 0, 0
        if self.style == 'hash': return self._hash(im), 0, 0
        if self.style == 'shadow': return self._shadow(im, dpi)
        return im, 0, 0  # Do nothing if style not found

    def _expand(self, small, large):
        """Expand small 2D array into large 2D array. Clipping not supported.
            small : 2D np.array
            large : 2D np.array must be n-times larger than small
            Note: Make sure to only supply 2D arrays, not 3D, ie
            self._expand(small, large[:, :, 0]) if necessary.
        """
        n = self.block
        if n == 1:
            large[...] = small
            return
        nx, ny = large.shape
        dx = nx % n
        dy = ny % n
        for a in range(n):
            for b in range(n):
                large[a:nx-dx:n, b:ny-dy:n] = small

    def _combine(self, rgb, alpha):  # return new array with rgb and alpha
        nx, ny = alpha.shape
        tgt = np.empty((nx, ny, 4))
        tgt[..., :3] = rgb
        tgt[..., 3] = alpha
        return tgt

    def _light(self, rgb, elevation):
        """Shade the supplied rgb array with the 2D elevation array.
            Note: elevation must be 2D, ie use noise[:, :, 0] if necessary.
        """
        rgb2 = self.light_source.shade_rgb(rgb, elevation,
                                           blend_mode=self.blend,
                                           fraction=self.fraction,
                                           vert_exag=self.vert)
        return rgb2

    def _noise(self, im):
        """ Simple noise texture with block and light support. Points are
            one of three darkness levels.
        """
        n = self.block
        rgb = im[...,:3]  # (nx, ny, 3)
        alpha = im[...,3]   # (nx, ny)
        nx, ny = alpha.shape
        shape = (nx//n + 1, ny//n + 1)  # expand shape slightly
        small = rnd.choice(self.choice, shape, p=self.p)  # (nx, ny)
        large = np.ones((shape[0]*n, shape[1]*n))
        self._expand(small, large)
        noise = np.ones((nx, ny, 1))  # (nx, ny, 1)
        noise[..., 0] = large[:nx, :ny]
        if self.light:
            rgb = self._light(rgb, noise[..., 0])
        else:
            rgb *= noise  # (nx, ny, 3) = (nx, ny, 3) * (nx, ny, 1)
        return self._combine(rgb, alpha)


    def _hash(self, im):
        """Similar to 'noise', but using small lines instead of points
        """
        n = self.block
        space = self.space
        rgb = im[...,:3]  # (nx, ny, 3)
        alpha = im[...,3]   # (nx, ny)
        nx, ny = alpha.shape
        # Expand image size so it takes multiples of block and space.
        # The buffers will be clipped later to fit into supplied size
        shape = ((nx+n)//n, (ny+n)//n)
        shape = ((shape[0]+(n+2)*space)//space * space,
                 (shape[1]+(n+2)*space)//space * space)
        # calculate number of points to be plotted.
        px = shape[0]//space
        py = shape[1]//space
        plen = px * py
        p = np.arange(plen)
        ix = []
        iy = []
        for i in range(2): # Do this for each set of points
            # make the indices for the point array, spread out by space
            # if space==4, then ix goes up 0, 0, 0,... 0, 4, 4,...
            # and y goes up 0, 4, 8,..., 0, 4, 8,...
            ix.append((p//py) * space)
            iy.append((p % px) * space)
            # now put them randomly in the spaces
            ix[i] += rnd.randint(0, space, plen)
            iy[i] += rnd.randint(0, space, plen)
        # a[ix, iy] will be the locations of random but evenly spread points.
        # Next step is to randomly choose these points to be given one of
        # the 4 possible directions. p is already an index into the points,
        # so just scramble p up and divide into 4 groups. Choosing fewer
        # points will create blank spots.
        # But I want 2 points per space, and I don't want to repeat the
        # directions used. This makes 6 possible pairs of directions, so
        # create 2 sets of points and divide them into 6 groups by reshaping
        # the vectors into (n//6, 6) arrays.
        r = []  # This will hold arrays of randomized indices
        slen = int((plen * self.frac)//6)  # length of subsets
        rlen = slen * 6  # total length of randomize indices
        for i in range(2):
            temp = rnd.choice(p, plen, replace=False)
            temp = temp[:rlen]  # truncate to needed size
            r.append(temp.reshape((slen, 6)))
        # Now stamp the points onto a buffer
        buffer = np.ones(shape)
        for i in range(6):
            for j in range(2):
                sub = np.ones(shape)
                dtn, axis = self.pairs[i][j]
                x = ix[j][r[j][:,i]]
                y = iy[j][r[j][:,i]]
                sub[x, y] = 1. - self.dark
                buffer *= sub
                s2 = sub * sub
                buffer *= np.roll(s2, dtn, axis)
                buffer *= np.roll(s2 * sub, 2 * dtn, axis)

        np.clip(buffer, self.clip, 1, buffer)
        large = np.ones((shape[0]*n, shape[1]*n))
        self._expand(buffer, large)
        noise = np.ones((nx, ny, 1))
        noise[..., 0] = large[:nx, :ny]
        if self.light:
            rgb = self._light(rgb, noise[..., 0])
        else:
            rgb *= noise  # (nx, ny, 3) = (nx, ny, 3) * (nx, ny, 1)
        return self._combine(rgb, alpha)

    def _shadow(self, im, dpi):
        """Create a drop shadow from the supplied image
        """
        def gaussian(n=50, lim=3):
            """ Gaussian window, with endpoints near 0. ``lim`` controls
                how narrow the peak is, and how 'tight' the shadow becomes.
                A larger value is narrower, with more of the shadow closer
                to the object.
            """
            x = np.linspace(-lim, lim, n)
            s = -0.5 * x**2
            y = np.exp(s)
            y -= y[0] - .01
            y /= y.max()
            return y

        alpha = im[...,3]   # (nx, ny)
        nx, ny = alpha.shape
        is_up = (self.direction == 'up')

        # Calculate and apply padding to image
        pix = int(self.pad * dpi)
        if is_up == 'up':
            padded = np.zeros((nx, ny+pix))
            xs = ys = 0  # image start location
        else:
            padded = np.zeros((nx + 2*pix, ny + 2*pix))
            xs = ys = pix

        # padded starts with chart object shape which gets blurred (smoothed).
        # It will then become the alpha channel for the expanded image.
        padded[xs:nx+xs, ys:ny+ys] = alpha

        # Create window for smoothing
        # Hanning is used instead of Hamming to reduce abruptness at ends
        if self.win == 'gaussian':
            w = gaussian(2 * pix + 1)
        else:
            w = np.hanning(2 * pix + 1) + .01
        if is_up:
            w[pix+1:] = 0 # just use beginning of window
        w /= w.sum()  # normalize

        # Smoth image in appropriate directions
        # Note no extra padding is used on convolutions, so may cause
        # artifacts for edge cases. See arg_filter demo for example.
        if not is_up:
            for x in range(padded.shape[0]):  # horizontal
                y = padded[x, :]
                # add some additional padding for the convolution
                yc = np.convolve(w, y, mode='same')
                padded[x, :] = yc
        for y in range(padded.shape[1]):  # vertical
            x = padded[:, y]
            # add some additional padding for the convolution
            xc = np.convolve(w, x, mode='same')
            padded[:, y] = xc

        # Normalize padded so that it has uniform darkness between figures
        padded /= padded.max()

        # Cut out original figure if desired
        if self.cut_figure:
            reverse = 1. - alpha
            padded[xs:nx+xs, ys:ny+ys] *= reverse

        # Return image
        rgb = np.ones((padded.shape[0], padded.shape[1], 3)) * self.color
        padded *= self.alpha
        new_im = self._combine(rgb, padded)
        return (new_im,
                self.offset[0] * dpi - xs,
                self.offset[1] * dpi - ys)

def wintest(lim=3):
    """ Show difference between a Hanning window and a Gaussian window.
        ``lim`` controls width of gaussian peak.
    """

    def gaussian(n=50, lim=3):
        x = np.linspace(-lim, lim, n)
        s = -0.5 * x**2
        y = np.exp(s)
        y -= y[0] - .01
        y /= y.max()
        return y

    fig = plt.figure(1)
    fig.clear()
    ax = fig.add_subplot(111)
    n = 40
    wh = np.hanning(n)
    wg = gaussian(n, lim)
    ax.plot(wh, label='Hanning')
    ax.plot(wg, label='Gaussian')
    plt.legend()
    plt.show()


def test(style, **kwargs):
    """ Simple test rig to get interpreter exceptions. Matplotlib will not
        raise exceptions when actually drawing, so use this to debug if
        the matplotlib code results in a blank window.
    """
    global filt
    filt = Texture(style=style, **kwargs)
    im = np.ones((20,20,4))
    res, a, b = filt.__call__(im)

def shadow_line(ax):
    """Demonstration of drop shadows.
    """
    filtH = Texture('shadow', alpha=0.5, pad=.5,
                    cut_figure=False, direction='all',
                    offset=(.1, .2), win='hanning')
    filtG = Texture('shadow', alpha=0.5, pad=.5,
                    cut_figure=False, direction='all',
                    offset=(.1, .2), win='gaussian')
    filtU = Texture('shadow', alpha=0.5, pad=.5,
                    cut_figure=True, direction='up',
                    offset=(0., 0.), win='gaussian')

    xH = np.array([0, 1])
    xG = xH + .5
    xU = xH + 1
    y = [0, 1]
    ax.plot(xH, y, lw=7, zorder=10, label='Hanning')
    ax.plot(xH, y, lw=7, zorder=9, agg_filter=filtH)
    ax.plot(xG, y, lw=7, zorder=10, label='Gaussian')
    ax.plot(xG, y, lw=7, zorder=9, agg_filter=filtG)
    ax.plot(xU, y, lw=7, zorder=10, label='Up Gaussian')
    ax.plot(xU, y, lw=7, zorder=9, agg_filter=filtU)
    plt.legend()

def texture_pie(ax):
    """ Demonstration routine showing multiple examples of textures and
        how to make them.
    """
    tf = []
    tf.append(Texture(style='noise', block=2))
    tf.append(Texture(style='noise', block=8))
    tf.append(Texture(style='noise', light=True, block=2))
    tf.append(Texture(style='noise', light=True, block=6))
    tf.append(Texture(style='hash', light=False, block=2, frac=.9, space=3))
    tf.append(Texture(style='hash', light=True, block=8, frac=.9, space=3))
    labels = []
    labels.append('Fine Noise')
    labels.append('Coarse Noise')
    labels.append('Shaded Noise')
    labels.append('Shaded\nCoarse Noise')
    labels.append('Fine Hash')
    labels.append('Shaded\nCoarse Hash')

    fracs = len(tf) * [10]
    explode = tuple([0.05] + (len(tf) - 1) * [0])
    pies = ax.pie(fracs, explode=explode, labels=labels, labeldistance=0.7)

    for pie, texture_filter in zip(pies[0], tf):
        pie.set_agg_filter(texture_filter)
        pie.set_rasterized(True)  # to support mixed-mode renderers
        pie.set(ec="none", lw=2)


if __name__=='__main__':
    demos = ['pie', 'line']
    if '__testing' not in globals():
        __testing = demos[0]
    if __testing not in demos:
        test(__testing)
    else:
        fig = plt.figure('Texture Demo', figsize=(8,8))
        fig.clear()
        plt.subplots_adjust(left=0.05, right=0.95)

        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        if __testing == 'line':
            shadow_line(ax)
        else:
            texture_pie(ax)
        ax.set_frame_on(True)

        plt.show()