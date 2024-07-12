import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

def lanczos_alg(
    matrix_vector_product: ArrayLike,
    dim: int,
    order: int,
    init_vec: ArrayLike) -> tuple:

    """Perform a Lanczos factorization of a matrix-vector product M*x = (V T V^t) * x
    M  is the input matrix (N x N),
    T is a tridiagonal matrix (n x n), 
    V is the transformation matrix (N x n) 
    
    Parameters
    ----------
    matrix_vector_product:
        Input M*x
    dim:
        Dimension of matrix M
    order:
        Order of the decompostion, and therefore dimension of matrix T
    init_vec:
        Input vector x

    Returns
    -------
    (tridiag, vecs)
    
    """ 

    def update(
            args: tuple, 
            i: int) -> tuple:
        """Perform a Lanczos factorization of a matrix-vector product M*x = (V T V^t) * x
        M  is the input matrix (N x N),
        T is a tridiagonal matrix (n x n), 
        V is the transformation matrix (N x n) 
        
        Parameters
        ----------
        args:
            Contains beta, the vectors forming V and T at the current step
        i:
            Iteration index

        Returns
        -------
        (beta, vecs, tridiag), (beta, vecs, tridiag)
        
        """ 

        beta, vecs, tridiag = args
        v = vecs[i, :].reshape((dim))
        w = matrix_vector_product(v)
        w = w - beta * jnp.where(i == 0, jnp.zeros(dim),
                               vecs[i - 1, :].reshape((dim)))

        alpha = jnp.dot(w, v)
        tridiag = tridiag.at[i, i].add(alpha)
        w = w - alpha * v

        beta = jnp.linalg.norm(w)
        tridiag = tridiag.at[i, i+1].add(beta)
        tridiag = tridiag.at[i+1, i].add(beta)
        # vecs = vecs.at[i+1, :].add(w/beta)
        vecs = vecs.at[i+1, :].add(jnp.where(beta>1e-8, w/beta, 0.))

        return (beta, vecs, tridiag), (beta, vecs, tridiag)

    tridiag = jnp.zeros((order, order))
    vecs = jnp.zeros((order, dim))

    init_vec = init_vec / jnp.linalg.norm(init_vec)
    vecs = vecs.at[0, :].add(init_vec)

    beta = 0
    (beta, vecs, tridiag), _ = lax.scan(update,  (beta, vecs, tridiag), jnp.arange(order))

    return (tridiag, vecs)
