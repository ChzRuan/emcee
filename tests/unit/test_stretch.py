# -*- coding: utf-8 -*-

from __future__ import division, print_function

import warnings

import pytest
import numpy as np

import emcee
from emcee import moves

__all__ = ["test_live_dangerously"]


def test_live_dangerously(nwalkers=32, nsteps=3000, seed=1234):
    warnings.filterwarnings("error")

    # Set up the random number generator.
    np.random.seed(seed)
    coords = np.random.randn(nwalkers, 2 * nwalkers)
    proposal = moves.StretchMove()
    sampler = emcee.EnsembleSampler(nwalkers, 1, lambda x: 0.0)

    # Test to make sure that the error is thrown if there aren't enough
    # walkers.
    with pytest.raises(RuntimeError):
        proposal.propose(coords, np.random.randn(nwalkers), None,
                         sampler, np.random)

    # Living dangerously...
    proposal.live_dangerously = True
    proposal.propose(coords, np.random.randn(nwalkers), None,
                     sampler, np.random)
