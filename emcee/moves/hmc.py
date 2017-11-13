# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .move import Move

__all__ = ["HamiltonianMove"]


class _hmc_wrapper(object):

    def __init__(self, random, grad_fn, cov, epsilon, nsteps=None):
        self.random = random
        self.grad_fn = grad_fn
        self.nsteps = nsteps
        self.epsilon = epsilon
        if len(np.atleast_1d(cov).shape) == 2:
            self.cov = _hmc_matrix(np.atleast_2d(cov))
        else:
            self.cov = _hmc_vector(np.asarray(cov))

    def __call__(self, args):
        current_q, current_p = args

        # Sample the initial momentum.
        q = current_q
        p = current_p

        # First take a half step in momentum.
        p = p + 0.5 * self.epsilon * self.grad_fn(current_q)

        # Alternate full steps in position and momentum.
        for i in range(self.nsteps):
            # First, a full step in position.
            q = q + self.epsilon * self.cov.apply(p)

            # Then a full step in momentum.
            if i < self.nsteps - 1:
                p = p + self.epsilon * self.grad_fn(q)

        # Finish with a half momentum step to synchronize with the position.
        p = p + 0.5 * self.epsilon * self.grad_fn(q)

        # Negate the momentum. This step really isn't necessary but it doesn't
        # hurt to keep it here for completeness.
        p = -p

        # Compute the acceptance probability factor.
        factor = 0.5 * np.dot(current_p, self.cov.apply(current_p))
        factor -= 0.5 * np.dot(p, self.cov.apply(p))
        return q, factor


class HamiltonianMove(Move):
    """A Hamiltonian Monte Carlo move.

    This implementation is based on the algorithm in Figure 2 of Neal (2012;
    http://arxiv.org/abs/1206.1901). By default, gradients of your model are
    computed numerically but this is unlikely to be efficient so it's best if
    you compute the gradients yourself using the
    :func:`Model.compute_grad_log_prior` and
    :func:`Model.compute_grad_log_likelihood` methods.

    Args:
        nsteps (int or (2,)): The number of leapfrog steps to take when
            integrating the dynamics. If an integer is provided, the number of
            steps will be constant. Instead, you can also provide a tuple with
            two integers and these will be treated as lower and upper limits
            on the number of steps and the used value will be uniformly
            sampled within that range.
        epsilon (float or (2,)): The step size used in the integration. Like
            ``nsteps`` a float can be given for a constant step size or a
            range can be given and the final value will be uniformly sampled.
        cov (Optional): An estimate of the parameter covariances. The inverse
            of ``cov`` is used as a mass matrix in the integration. (default:
            ``1.0``)

    """

    _wrapper = _hmc_wrapper

    def __init__(self, nsteps, epsilon, cov=1.0):
        self.nsteps = nsteps
        self.epsilon = epsilon
        self.cov = cov

    def get_args(self, random):
        try:
            eps = float(self.epsilon)
        except TypeError:
            eps = random.uniform(self.epsilon[0], self.epsilon[1])

        # Randomize the number of steps.
        try:
            L = int(self.nsteps)
        except TypeError:
            L = random.randint(self.nsteps[0], self.nsteps[1])

        return eps, L

    def propose(self, coords, log_probs, blobs, sampler, random):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

        """
        nwalkers, ndim = coords.shape

        # Set up the integrator and sample the initial momenta.
        integrator = self._wrapper(random, sampler.log_prob_fn.grad,
                                   self.cov, *(self.get_args(random)))
        momenta = integrator.cov.sample(random, nwalkers, ndim)

        if sampler.pool is None:
            M = map
        else:
            M = sampler.pool.map

        # Integrate the dynamics in parallel.
        q, factors = map(np.array, zip(*(M(integrator, zip(coords, momenta)))))
        new_log_probs, new_blobs = sampler.compute_log_prob(q)

        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - log_probs + factors
        accepted = np.log(random.rand(nwalkers)) < lnpdiff

        # Update the parameters
        coords, log_probs, blobs = self.update(
            coords, log_probs, blobs,
            q, new_log_probs, new_blobs,
            accepted)

        return coords, log_probs, blobs, accepted


class _hmc_vector(object):

    def __init__(self, cov):
        self.cov = cov
        self.inv_cov = 1.0 / cov

    def sample(self, random, *shape):
        return random.randn(*shape) * np.sqrt(self.inv_cov)

    def apply(self, x):
        return self.cov * x


class _hmc_matrix(object):

    def __init__(self, cov):
        self.cov = cov
        self.inv_cov = np.linalg.inv(self.cov)

    def sample(self, random, *shape):
        return random.multivariate_normal(np.zeros(shape[-1]),
                                          self.inv_cov,
                                          *(shape[:-1]))

    def apply(self, x):
        return np.dot(self.cov, x)
