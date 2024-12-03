import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike


def lanczos_alg(
    matrix_vector_product: ArrayLike, dim: int, order: int, init_vec: ArrayLike
) -> tuple[Array, Array]:
    """Perform a Lanczos factorization of a matrix-vector product M*x = (V T V^t) * x.

    Where T is a tridiagonal matrix (order x order) and V is a transformation matrix (dim x order).

    Parameters
    ----------
    matrix_vector_product: (float)
        Array (,dim) containing input matrix-vector produc M(x)
    dim: (int)
        Dimension of matrix M
    order: (int)
        Order of the decomposition and dimension of the spanned Krylov subspace
    init_vec: (float)
        Array (,dim) containing input vector x

    Returns
    -------
    (tridiag, vecs)

    """

    def update(
        args: tuple[ArrayLike, ArrayLike, ArrayLike], i: int
    ) -> tuple[tuple[ArrayLike, ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """Perform one iteration of the Lanczos decomposition.

        This results in the increase of the dimensione of the Krylov subspace by one.

        Parameters
        ----------
        args: (float)
            Tuple containing beta, array (dim,order) V and array T (order,order), at the current step
        i: (int)
            Iteration index

        Returns
        -------
        (beta, vecs, tridiag), (beta, vecs, tridiag)

        """
        beta, vecs, tridiag = args
        v = vecs[i, :].reshape((dim))
        w = matrix_vector_product(v)
        w = w - beta * jnp.where(i == 0, jnp.zeros(dim), vecs[i - 1, :].reshape((dim)))

        alpha = jnp.dot(w, v)
        tridiag = tridiag.at[i, i].add(alpha)
        w = w - alpha * v

        beta = jnp.linalg.norm(w)
        tridiag = tridiag.at[i, i + 1].add(beta)
        tridiag = tridiag.at[i + 1, i].add(beta)
        # vecs = vecs.at[i+1, :].add(w/beta)
        vecs = vecs.at[i + 1, :].add(jnp.where(beta > 1e-8, w / beta, 0.0))

        return (beta, vecs, tridiag), (beta, vecs, tridiag)

    tridiag = jnp.zeros((order, order))
    vecs = jnp.zeros((order, dim))

    init_vec = init_vec / jnp.linalg.norm(init_vec)
    vecs = vecs.at[0, :].add(init_vec)

    beta = 0
    (beta, vecs, tridiag), _ = lax.scan(update, (beta, vecs, tridiag), jnp.arange(order))

    return (tridiag, vecs)
