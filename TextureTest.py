#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:33:27 2018

@author: dan
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from matplotlib.colors import LightSource

class Texture(object):
    "simple texture filter"

    def __init__(self, style='noise', **kwargs):
        self.style = style
        self.block = kwargs.get('block', 1)
        self.light = kwargs.get('light', False)
        if self.light:
            prob = kwargs.get('prob', .15)
            dark = kwargs.get('dark', .15)
            self.fraction = kwargs.get('fraction', 0.2)
            self.vert = kwargs.get('vert_exag', 2.)
            self.blend = kwargs.get('blend_mode', 'soft')
            self.fraction = kwargs.get('fraction', .5)
            self.light_source = LightSource()
        else:
            prob = kwargs.get('prob', .2)
            dark = kwargs.get('dark', .02)
        if style == 'hash':
            prob /= 4.
            self.choice = [1. - dark, 1.]
            self.p = [prob, 1. - prob]
        elif style == 'hash2':
            # hash2 spreads out points somewhat evenly, but random within
            # 3x3 blocks. Up to 2 points are chosen per block.
            # Set up arrays and their probabilities
            self.p = 9 * [1./11] + [2./11]  # 2 chances to get blank
            self.choice = list(range(9)) + [-1]
            self.clip = (1. - dark)**3  # clip limit (max darkness)
            blank = np.ones((3,3))
            self.template = {-1: blank.copy()}
            blank[0, 0] = 1. - dark
            for i in range(9):
                    self.template[i] = blank
                    blank = np.roll(blank, 1)
        else:
            self.choice = list(1. - np.array([1,2,3]) * dark) + [1.]
            self.p = 3 * [prob] + [1. - 3 * prob]
        self.kwargs = kwargs

    def __call__(self, im, dpi=100):
        if self.style == 'noise': return self._noise(im), 0, 0
        if self.style == 'hash': return self._hash(im), 0, 0
        if self.style == 'hash2': return self._hash2(im), 0, 0
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

    def _combine(self, rgb, clip):  # return new array with rgb and clip
        nx, ny = clip.shape
        tgt = np.empty((nx, ny, 4))
        tgt[..., :3] = rgb
        tgt[..., 3] = clip
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
        n = self.block
        rgb = im[...,:3]  # (nx, ny, 3)
        clip = im[...,3]   # (nx, ny)
        nx, ny = clip.shape
        shape = (nx//n, ny//n)
        half = rnd.choice(self.choice, shape, p=self.p)  # (nx, ny)
        noise = np.ones((nx, ny, 1))  # (nx, ny, 1)
        self._expand(half, noise[..., 0])
        if self.light:
            rgb = self._light(rgb, noise[..., 0])
        else:
            rgb *= noise  # (nx, ny, 3) = (nx, ny, 3) * (nx, ny, 1)
        return self._combine(rgb, clip)

    def _hash(self, im):
        n = self.block
        rgb = im[...,:3]  # (nx, ny, 3)
        clip = im[...,3]   # (nx, ny)
        nx, ny = clip.shape
        shape = (nx//n, ny//n)
        noise = np.ones((nx, ny, 1))
        buffer = np.ones(shape)
        for axis in [0, 1]:
            for dtn in [-1, 1]:
                small = rnd.choice(self.choice, shape, p=self.p)
                buffer *= small
                s2 = small * small
                buffer *= np.roll(s2, dtn, axis)
                buffer *= np.roll(s2 * small, 2 * dtn, axis)
        np.clip(buffer, self.clip, 1, buffer)
        self._expand(buffer, noise[..., 0])
        if self.light:
            rgb = self._light(rgb, noise[..., 0])
        else:
            rgb *= noise  # (nx, ny, 3) = (nx, ny, 3) * (nx, ny, 1)
        return self._combine(rgb, clip)

    def _hash2(self, im):
        n = self.block
        rgb = im[...,:3]  # (nx, ny, 3)
        clip = im[...,3]   # (nx, ny)
        nx, ny = clip.shape
        shape = (nx//n, ny//n)
        small = {}
        for dtn in [-1, 1]:
            for axis in [0, 1]:
                small[(dtn, axis)] = np.ones(shape)
        keys = list(small.keys())
        self.keys = keys #test
        self.small = small #test
        for x in range(nx//n-3):
            for y in range(ny//n-3):
                # Pick two each of direction buffers and templates
                # can't choose tuples directly, so use index to keys list
                k = rnd.choice(len(keys), 2, replace=False)
                t = rnd.choice(self.choice, 2, replace=False, p=self.p)
                self.k = k #test
                self.t = t #test
                for i in range(2):
                    b = small[keys[k[i]]]
                    b[x:x+3, y:y+3] *= self.template[t[i]]
        buffer = np.ones(shape)
        for dtn in [-1, 1]:
            for axis in [0, 1]:
                s = small[(dtn, axis)]
                s2 = s * s
                buffer *= s
                buffer *= np.roll(s2, dtn, axis)
                buffer *= np.roll(s2 * s, 2 * dtn, axis)
        np.clip(buffer, self.choice[0]**3, 1, buffer)
        noise = np.ones((nx, ny, 1))
        self._expand(buffer, noise[..., 0])
        if self.light:
            rgb = self._light(rgb, noise[..., 0])
        else:
            rgb *= noise  # (nx, ny, 3) = (nx, ny, 3) * (nx, ny, 1)
        return self._combine(rgb, clip)

def test(style, **kwargs):
    global filt
    filt = Texture(style=style, **kwargs)
    im = np.ones((20,20,4))
    res, a, b = filt.__call__(im)


def texture_pie(ax):
    tf = []
    tf.append(Texture(style='noise', block=2))
    tf.append(Texture(style='noise', block=8))
    tf.append(Texture(style='noise', light=True, block=2))
    tf.append(Texture(style='noise', light=True, block=6))
    tf.append(Texture(style='hash2', light=False, block=3))
    tf.append(Texture(style='hash2', light=True, block=6))
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
#    test('hash2')
    fig = plt.figure(figsize=(8,8))
    fig.clear()
    plt.subplots_adjust(left=0.05, right=0.95)

    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    texture_pie(ax)
    ax.set_frame_on(True)

    plt.show()