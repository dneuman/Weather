#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:33:27 2018

@author: dan
"""

import matplotlib.pyplot as plt
import numpy as np
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
        else:
            self.choice = list(1. - np.array([1,2,3]) * dark) + [1.]
            self.p = 3 * [prob] + [1. - 3 * prob]
        self.kwargs = kwargs

    def __call__(self, im, dpi):
        if self.style == 'noise': return self._noise(im), 0, 0
        if self.style == 'hash': return self._hash(im), 0, 0
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
        half = np.random.choice(self.choice, shape, p=self.p)  # (nx, ny)
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
                half = np.random.choice(self.choice, shape, p=self.p)
                buffer *= half
                h2 = half * half
                buffer *= np.roll(h2, dtn, axis)
                buffer *= np.roll(h2 * half, 2 * dtn, axis)
        np.clip(buffer, self.choice[0]**3, 1, buffer)
        self._expand(buffer, noise[..., 0])
        if self.light:
            rgb = self._light(rgb, noise[..., 0])
        else:
            rgb *= noise  # (nx, ny, 3) = (nx, ny, 3) * (nx, ny, 1)
        return self._combine(rgb, clip)


def texture_pie(ax):
    tf = []
    tf.append(Texture(style='noise', block=2))
    tf.append(Texture(style='noise', block=8))
    tf.append(Texture(style='noise', light=True, block=2))
    tf.append(Texture(style='noise', light=True, block=6))
    tf.append(Texture(style='hash', light=False, block=3))
    tf.append(Texture(style='hash', light=True, block=6))
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
    fig = plt.figure(figsize=(8,8))
    fig.clear()
    plt.subplots_adjust(left=0.05, right=0.95)

    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    texture_pie(ax)
    ax.set_frame_on(True)

    plt.show()