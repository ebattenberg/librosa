#!/usr/bin/env python
"""
CREATED:2015-03-01 by Eric Battenberg <ebattenberg@gmail.com>
unit tests for librosa core.constantq

Run me as follows:
    cd tests/
    nosetests -v --with-coverage --cover-package=librosa
"""
from __future__ import division

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import librosa
import numpy as np

from nose.tools import nottest, eq_, raises

def __test_cqt_size(y, sr, hop_length, fmin, n_bins, bins_per_octave,
            tuning, resolution, aggregate, norm, sparsity):

    cqt_output = librosa.cqt(y,
                             sr=sr,
                             hop_length=hop_length,
                             fmin=fmin,
                             n_bins=n_bins,
                             bins_per_octave=bins_per_octave,
                             tuning=tuning,
                             resolution=resolution,
                             aggregate=aggregate,
                             norm=norm,
                             sparsity=sparsity)

    assert cqt_output.shape[0] == n_bins

    return cqt_output


def test_cqt():

    sr = 11025


    # Impulse train
    y = np.zeros(int(5.0 * sr))
    y[::sr] = 1.0


    # Hop size not long enough for num octaves
    # num_octaves = 6, 2**6 = 64 > 32
    yield (raises(ValueError)(__test_cqt_size), y, sr, 32, None, 72,
           12, None, 2, None, 1, 0.01)

    # Filters go beyond Nyquist. 500 Hz -> 4 octaves = 8000 Hz > 11000 Hz
    yield (raises(ValueError)(__test_cqt_size), y, sr, 512, 500, 48,
           12, None, 2, None, 1, 0.01)


    # Test for no errors and correct output size
    for fmin in [None, librosa.note_to_hz('C3')]:
        for n_bins in [1, 12, 24, 48, 72, 74, 76]:
            for bins_per_octave in [12, 24]:
                for tuning in [0, 0.25]:
                    for resolution in [1, 2]:
                        for norm in [1, 2]:
                            yield (__test_cqt_size, y, sr, 512, fmin, n_bins,
                                bins_per_octave, tuning,
                                resolution, None, norm, 0.01)

    # Test with fmin near Nyquist
    for fmin in [3000, 4800]:
        for n_bins in [1, 2]:
            for bins_per_octave in [12]:
                yield (__test_cqt_size, y, sr, 512, fmin, n_bins,
                       bins_per_octave, None, 2, None, 1, 0.01)


def test_cqt_band_equivalence():


    def __compare_upper(upper_start, y, sr, hop_length, fmin, n_bins,
                        bins_per_octave, tuning, resolution, aggregate,
                        norm, sparsity):
        """
        Test that upper bands of CQT are equivalent to a CQT that
        starts at upper bands.
        """

        full_cqt = __test_cqt_size(y, sr, hop_length, fmin, n_bins,
                                   bins_per_octave, tuning, resolution,
                                   aggregate, norm, sparsity)

        fmin_upper = fmin * 2**(upper_start/bins_per_octave)
        n_bins_upper = n_bins - upper_start
        upper_cqt = __test_cqt_size(y, sr, hop_length, fmin_upper, n_bins_upper,
                           bins_per_octave, tuning, resolution, aggregate,
                           norm, sparsity)

        assert np.allclose(full_cqt[upper_start:], upper_cqt)

    def __compare_lower(lower_n_bins, y, sr, hop_length, fmin, n_bins,
                        bins_per_octave, tuning, resolution, aggregate,
                        norm, sparsity):
        """
        Test that lower bands of CQT are equivalent to those of a CQT that
        starts at the same `fmin` but has a larger `n_bins`.
        (This is to test the effect of different resampling optimizations.)
        """

        full_cqt = __test_cqt_size(y, sr, hop_length, fmin, n_bins,
                          bins_per_octave, tuning, resolution, aggregate,
                          norm, sparsity)

        lower_cqt = __test_cqt_size(y, sr, hop_length, fmin, lower_n_bins,
                           bins_per_octave, tuning, resolution, aggregate,
                           norm, sparsity)

        assert np.allclose(full_cqt[:lower_n_bins], lower_cqt)


    sr = 11025


    # Impulse train
    y = np.zeros(int(5.0 * sr))
    y[::sr] = 1.0


    # Test that upper bands of CQT are equivalent to a CQT that
    # starts at upper bands.
    for upper_start in [1, 11, 24, 30]:
        for n_bins in [36, 60, 70]:
            yield (__compare_upper, 5, y, sr, 512, 100, n_bins,
                12, None, 2, None, 1, 0.01)

    # Test that lower bands of CQT are equivalent to those of a CQT that
    # starts at the same `fmin` but has a larger `n_bins`.
    # (This is to test the effect of different resampling optimizations.)
    for n_bins_lower in [1, 11, 24, 30]:
        for n_bins in [12, 60, 72]:
            yield (__compare_lower, n_bins_lower, y, sr, 512, 100,
                   n_bins, 12, None, 2, None, 1, 0.01)
