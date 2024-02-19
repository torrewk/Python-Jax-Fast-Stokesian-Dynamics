# Copyright 2020 Google LLC.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import functools
from jax import random
from jax import lax
import jax.numpy as jnp
import jax.ops
import jax.scipy as jsp
from jax.tree_util import Partial
import scipy.sparse.linalg

def _identity(x):
  return x

def _inner(v, q):
  h_jk = q.conj() @ v
  v = v - h_jk * q
  return (v, h_jk)

def _outer(A, M, Q, k):
  q = Q[:, k]
  v = A(M(q))
  # TODO: maybe better to use a masked dot-product rather than scan?
  v, h_col = lax.scan(_inner, v, Q.T)
  v_norm = jnp.linalg.norm(v)
  Q = Q.at[:, k+1].set(v / v_norm)
  h_col = h_col.at[k+1].set(v_norm)
  return Q, h_col

def arnoldi_iteration(A, b, n, M=None):
  # https://en.wikipedia.org/wiki/Arnoldi_iteration#The_Arnoldi_iteration
  if M is None:
    M = _identity
  m = b.shape[0]
  q = b / jnp.linalg.norm(b)
  Q = jnp.concatenate([q[:, jnp.newaxis], jnp.zeros((m, n))], axis=1)
  Q, h = lax.scan(functools.partial(_outer, A, M), Q, np.arange(n))
  return Q, h.T

@jax.jit
def lstsq(a, b):
  # return jsp.linalg.solve(a.T @ a, a.T @ b, sym_pos=True)
  return jsp.linalg.solve(a.T @ a, a.T @ b)

def _gmres(A, b, x0, n, M):
  # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf
  Q, H = arnoldi_iteration(A, b, n, M)
  beta = jnp.linalg.norm(b - A(x0))
  e1 = jnp.concatenate([jnp.ones((1,)), jnp.zeros((n,))])
  y = lstsq(H, beta * e1)
  x = x0 + M(Q[:, :-1] @ y)
  return x

def gmres(A, b, x0=None, n=5, M=None):
  if x0 is None:
    x0 = jnp.zeros_like(b)
  if M is None:
    M = _identity
  return _gmres(A, b, x0, n, M)