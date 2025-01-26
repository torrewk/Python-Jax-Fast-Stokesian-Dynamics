import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

def lanczos_algorithm(
    matrix_vector_product: ArrayLike, dimension: int, order: int, initial_vector: ArrayLike
) -> tuple[Array, Array]:
    """Perform a Lanczos factorization of a matrix-vector product M*x = (V T V^T) * x.

    Where T is a tridiagonal matrix (order x order) and V is a transformation matrix (dimension x order).

    Parameters
    ----------
    matrix_vector_product : ArrayLike
        Function computing the matrix-vector product M(x).
    dimension : int
        Dimension of matrix M.
    order : int
        Order of the decomposition and dimension of the spanned Krylov subspace.
    initial_vector : ArrayLike
        Initial vector x (shape: (dimension,)).

    Returns
    -------
    tuple[Array, Array]
        - Tridiagonal matrix T of shape (order, order).
        - Transformation matrix V of shape (order, dimension).
    """
    def update(
        args: tuple[ArrayLike, ArrayLike, ArrayLike], i: int
    ) -> tuple[tuple[ArrayLike, ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """Perform one iteration of the Lanczos decomposition.

        This results in the increase of the dimension of the Krylov subspace by one.

        Parameters
        ----------
        args : tuple[ArrayLike, ArrayLike, ArrayLike]
            Tuple containing beta, vector matrix V (order, dimension), and tridiagonal matrix T (order, order).
        i : int
            Iteration index.

        Returns
        -------
        tuple[tuple[ArrayLike, ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike, ArrayLike]]
            Updated beta, vector matrix V, and tridiagonal matrix T.
        """
        beta, vectors, tridiagonal = args
        v = vectors[i, :].reshape((dimension,))
        w = matrix_vector_product(v)
        w = w - beta * jnp.where(i == 0, jnp.zeros(dimension), vectors[i - 1, :].reshape((dimension,)))

        alpha = jnp.dot(w, v)
        tridiagonal = tridiagonal.at[i, i].add(alpha)
        w = w - alpha * v

        beta = jnp.linalg.norm(w)
        tridiagonal = tridiagonal.at[i, i + 1].add(beta)
        tridiagonal = tridiagonal.at[i + 1, i].add(beta)
        vectors = vectors.at[i + 1, :].add(jnp.where(beta > 1e-8, w / beta, 0.0))

        return (beta, vectors, tridiagonal), (beta, vectors, tridiagonal)

    tridiagonal = jnp.zeros((order, order))
    vectors = jnp.zeros((order, dimension))

    initial_vector = initial_vector / jnp.linalg.norm(initial_vector)
    vectors = vectors.at[0, :].add(initial_vector)

    beta = 0
    (beta, vectors, tridiagonal), _ = lax.scan(update, (beta, vectors, tridiagonal), jnp.arange(order))

    return tridiagonal, vectors

