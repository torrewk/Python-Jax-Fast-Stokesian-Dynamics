import jax.numpy as jnp
from jax import lax

# Perform Lanczos factorization of a matrix-vector product M*x = (V T V^t) * x
# M (N x N) is the input,
# V (n x n) is a tridiagonal matrix
# V (N x n) is the transformation matrix


def lanczos_alg(
    matrix_vector_product,  # input M*x
    dim,  # dimension of matrix M
    order,  # order of the decompostion, and therefore dimension of matrix T
    init_vec,  # input vector x
):

    def update(args, i):

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
        vecs = vecs.at[i + 1, :].add(w / beta)

        return (beta, vecs, tridiag), (beta, vecs, tridiag)

    tridiag = jnp.zeros((order, order))
    vecs = jnp.zeros((order, dim))

    init_vec = init_vec / jnp.linalg.norm(init_vec)
    vecs = vecs.at[0, :].add(init_vec)

    beta = 0
    (beta, vecs, tridiag), _ = lax.scan(
        update, (beta, vecs, tridiag), jnp.arange(order)
    )

    return (tridiag, vecs)
