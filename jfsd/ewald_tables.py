import math
from decimal import Decimal

import numpy as np
from jax.typing import ArrayLike


def compute_real_space_ewald_table(num_entries: int, radius: float, xi: float) -> ArrayLike:
    """Construct the table containing mobility scalar function values as functions of discrete distance.

    These will be used to linearly interpolate values for any distance.
    Due to the high complexity of the mobility functions, calculations are performed in 'extended' precision
    and then truncated to single precision.

    Parameters
    ----------
    num_entries : int
        Number of entries in the tabulation, for each scalar mobility function.
    radius : float
        Particle radius.
    xi : float
        Ewald splitting parameter.

    Returns
    -------
    np.ndarray
        Array containing the computed Ewald coefficients.
    """

    # Table discretization in extended precision (80-bit)
    dr_string = "0.00100000000000000000000000000000"  # Pass value as a string with arbitrary precision
    dr_decimal = Decimal(dr_string)  # Convert to float with arbitrary precision
    dr = np.longfloat(dr_decimal)  # Convert to numpy long float (truncated to 64/80/128-bit, depending on platform)

    ident_minus_rr = np.zeros(num_entries)
    rr = np.zeros(num_entries)
    scalar_g1 = np.zeros(num_entries)
    scalar_g2 = np.zeros(num_entries)
    scalar_h1 = np.zeros(num_entries)
    scalar_h2 = np.zeros(num_entries)
    scalar_h3 = np.zeros(num_entries)

    xi_longfloat = np.longfloat(xi)
    radius_longfloat = np.longfloat(radius)
    PI = np.longfloat(np.pi)
    index_array = np.arange(num_entries, dtype=np.longdouble)
    r_array = index_array * dr + dr

    # Expressions have been simplified assuming no overlap, touching, and overlap
    for i in range(num_entries):
        r = r_array[i]

        if r > 2 * radius_longfloat:
            ident_minus_rr[i] = (
                -math.pow(radius_longfloat, -1)
                + (math.pow(radius_longfloat, 2) * math.pow(r, -3)) / 2.0
                + (3 * math.pow(r, -1)) / 4.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -2)
                    * math.pow(r, -3)
                    * (-12 * math.pow(r, 4) + math.pow(xi_longfloat, -4))
                )
                / 128
                + math.pow(radius_longfloat, -2)
                * ((9 * r) / 32.0 - (3 * math.pow(r, -3) * math.pow(xi_longfloat, -4)) / 128.0)
                + (
                    math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * (
                        128 * math.pow(radius_longfloat, -1)
                        + 64 * math.pow(radius_longfloat, 2) * math.pow(r, -3)
                        + 96 * math.pow(r, -1)
                        + math.pow(radius_longfloat, -2) * (36 * r - 3 * math.pow(r, -3) * math.pow(xi_longfloat, -4))
                    )
                )
                / 256.0
                + (
                    math.erfc(2 * radius_longfloat * xi_longfloat - r * xi_longfloat)
                    * (
                        128 * math.pow(radius_longfloat, -1)
                        - 64 * math.pow(radius_longfloat, 2) * math.pow(r, -3)
                        - 96 * math.pow(r, -1)
                        + math.pow(radius_longfloat, -2) * (-36 * r + 3 * math.pow(r, -3) * math.pow(xi_longfloat, -4))
                    )
                )
                / 256.0
                + (
                    3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -2)
                    * math.pow(xi_longfloat, -3)
                    * (1 + 6 * math.pow(r, 2) * math.pow(xi_longfloat, 2))
                )
                / 64.0
                + (
                    math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -3)
                    * (
                        8 * r * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2)
                        - 16 * math.pow(radius_longfloat, 3) * math.pow(xi_longfloat, 2)
                        + radius_longfloat * (2 - 28 * math.pow(r, 2) * math.pow(xi_longfloat, 2))
                        - 3 * (r + 6 * math.pow(r, 3) * math.pow(xi_longfloat, 2))
                    )
                )
                / 128.0
                + (
                    math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -3)
                    * (
                        8 * r * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2)
                        + 16 * math.pow(radius_longfloat, 3) * math.pow(xi_longfloat, 2)
                        + radius_longfloat * (-2 + 28 * math.pow(r, 2) * math.pow(xi_longfloat, 2))
                        - 3 * (r + 6 * math.pow(r, 3) * math.pow(xi_longfloat, 2))
                    )
                )
                / 128.0
            )

            rr[i] = (
                -math.pow(radius_longfloat, -1)
                - math.pow(radius_longfloat, 2) * math.pow(r, -3)
                + (3 * math.pow(r, -1)) / 2.0
                + (
                    3
                    * math.pow(radius_longfloat, -2)
                    * math.pow(r, -3)
                    * (4 * math.pow(r, 4) + math.pow(xi_longfloat, -4))
                )
                / 64.0
                + (
                    math.erfc(2 * radius_longfloat * xi_longfloat - r * xi_longfloat)
                    * (
                        64 * math.pow(radius_longfloat, -1)
                        + 64 * math.pow(radius_longfloat, 2) * math.pow(r, -3)
                        - 96 * math.pow(r, -1)
                        + math.pow(radius_longfloat, -2) * (-12 * r - 3 * math.pow(r, -3) * math.pow(xi_longfloat, -4))
                    )
                )
                / 128.0
                + (
                    math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * (
                        64 * math.pow(radius_longfloat, -1)
                        - 64 * math.pow(radius_longfloat, 2) * math.pow(r, -3)
                        + 96 * math.pow(r, -1)
                        + math.pow(radius_longfloat, -2) * (12 * r + 3 * math.pow(r, -3) * math.pow(xi_longfloat, -4))
                    )
                )
                / 128.0
                + (
                    3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -2)
                    * math.pow(xi_longfloat, -3)
                    * (-1 + 2 * math.pow(r, 2) * math.pow(xi_longfloat, 2))
                )
                / 32.0
                - (
                    (2 * radius_longfloat + 3 * r)
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -3)
                    * (
                        -1
                        - 8 * radius_longfloat * r * math.pow(xi_longfloat, 2)
                        + 8 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2)
                        + 2 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                    )
                )
                / 64.0
                + (
                    (2 * radius_longfloat - 3 * r)
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -3)
                    * (
                        -1
                        + 8 * radius_longfloat * r * math.pow(xi_longfloat, 2)
                        + 8 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2)
                        + 2 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                    )
                )
                / 64.0
                - (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -2)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -4)
                    * (1 + 4 * math.pow(r, 4) * math.pow(xi_longfloat, 4))
                )
                / 64.0
            )

            scalar_g1[i] = (
                (
                    math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -5)
                    * (
                        9
                        + 15 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        - 30 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                    )
                )
                / 64.0
                + (
                    math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        18 * radius_longfloat
                        - 45 * r
                        - 3
                        * (2 * radius_longfloat + r)
                        * (-16 * radius_longfloat * r + 8 * math.pow(radius_longfloat, 2) + 25 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        + 6
                        * (2 * radius_longfloat + r)
                        * (
                            -32 * r * math.pow(radius_longfloat, 3)
                            + 32 * math.pow(radius_longfloat, 4)
                            + 44 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 36 * radius_longfloat * math.pow(r, 3)
                            + 25 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        -9 * (2 * radius_longfloat + 5 * r)
                        + 3
                        * (2 * radius_longfloat - r)
                        * (16 * radius_longfloat * r + 8 * math.pow(radius_longfloat, 2) + 25 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        - 6
                        * (2 * radius_longfloat - r)
                        * (
                            32 * r * math.pow(radius_longfloat, 3)
                            + 32 * math.pow(radius_longfloat, 4)
                            + 44 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 36 * radius_longfloat * math.pow(r, 3)
                            + 25 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        3
                        + 3 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 20 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 128.0
                - (
                    3
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        15
                        + 5
                        * math.pow(r, 2)
                        * math.pow(xi_longfloat, 2)
                        * (3 + 64 * math.pow(radius_longfloat, 4) * math.pow(xi_longfloat, 4))
                        + 512 * math.pow(radius_longfloat, 6) * math.pow(xi_longfloat, 6)
                        - 256 * radius_longfloat * math.pow(r, 5) * math.pow(xi_longfloat, 6)
                        + 100 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
                - (
                    3
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        15
                        + 5
                        * math.pow(r, 2)
                        * math.pow(xi_longfloat, 2)
                        * (3 + 64 * math.pow(radius_longfloat, 4) * math.pow(xi_longfloat, 4))
                        + 512 * math.pow(radius_longfloat, 6) * math.pow(xi_longfloat, 6)
                        + 256 * radius_longfloat * math.pow(r, 5) * math.pow(xi_longfloat, 6)
                        + 100 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
            )

            scalar_g2[i] = (
                (
                    -3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -5)
                    * (
                        3
                        - math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 2 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                    )
                )
                / 64.0
                + (
                    math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        18 * radius_longfloat
                        + 45 * r
                        - 3
                        * (
                            24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            + 5 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        + 6
                        * (
                            24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            + 5 * math.pow(r, 3)
                        )
                        * math.pow(-2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        -18 * radius_longfloat
                        + 45 * r
                        + 3
                        * (
                            -24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            - 5 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        - 6
                        * (
                            -24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            - 5 * math.pow(r, 3)
                        )
                        * math.pow(2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    3
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        15
                        - 15 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 4
                        * (
                            128 * math.pow(radius_longfloat, 6)
                            - 80 * math.pow(radius_longfloat, 4) * math.pow(r, 2)
                            + 16 * radius_longfloat * math.pow(r, 5)
                            - 5 * math.pow(r, 6)
                        )
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        -3
                        + 3 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 4 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 128.0
                - (
                    3
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        -15
                        + 15 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 4
                        * (
                            -128 * math.pow(radius_longfloat, 6)
                            + 80 * math.pow(radius_longfloat, 4) * math.pow(r, 2)
                            + 16 * radius_longfloat * math.pow(r, 5)
                            + 5 * math.pow(r, 6)
                        )
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
            )

            scalar_h1[i] = (
                (
                    3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -7)
                    * (
                        27
                        - 2
                        * math.pow(xi_longfloat, 2)
                        * (
                            15 * math.pow(r, 2)
                            + 2 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 4 * math.pow(r, 6) * math.pow(xi_longfloat, 4)
                            + 48
                            * math.pow(radius_longfloat, 2)
                            * (
                                3
                                - math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 2 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                            )
                        )
                    )
                )
                / 4096.0
                + (
                    3
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        270 * radius_longfloat
                        - 135 * r
                        + 6
                        * (2 * radius_longfloat + 5 * r)
                        * (12 * math.pow(radius_longfloat, 2) + 5 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        - 4
                        * (
                            144 * r * math.pow(radius_longfloat, 4)
                            + 96 * math.pow(radius_longfloat, 5)
                            + 64 * math.pow(radius_longfloat, 3) * math.pow(r, 2)
                            - 30 * radius_longfloat * math.pow(r, 4)
                            - 5 * math.pow(r, 5)
                        )
                        * math.pow(xi_longfloat, 4)
                        + 8
                        * math.pow(2 * radius_longfloat - r, 3)
                        * (
                            96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 40960.0
                + (
                    3
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        -135 * (2 * radius_longfloat + r)
                        - 6
                        * (2 * radius_longfloat - 5 * r)
                        * (12 * math.pow(radius_longfloat, 2) + 5 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        + 4
                        * (
                            -144 * r * math.pow(radius_longfloat, 4)
                            + 96 * math.pow(radius_longfloat, 5)
                            + 64 * math.pow(radius_longfloat, 3) * math.pow(r, 2)
                            - 30 * radius_longfloat * math.pow(r, 4)
                            + 5 * math.pow(r, 5)
                        )
                        * math.pow(xi_longfloat, 4)
                        - 8
                        * (
                            -96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(2 * radius_longfloat + r, 3)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 40960.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        27
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            -6 * math.pow(r, 2)
                            + 9 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 2 * math.pow(r, 8) * math.pow(xi_longfloat, 6)
                            + 12
                            * math.pow(radius_longfloat, 2)
                            * (
                                -3
                                + 3 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 4 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                            )
                        )
                    )
                )
                / 8192.0
                + (
                    3
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -135
                        + 240 * (6 * math.pow(radius_longfloat, 2) + math.pow(r, 2)) * math.pow(xi_longfloat, 2)
                        - 360
                        * math.pow(r, 2)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(xi_longfloat, 4)
                        + 16
                        * (
                            96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(-2 * radius_longfloat + r, 4)
                        * math.pow(xi_longfloat, 8)
                    )
                )
                / 81920.0
                + (
                    3
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -135
                        + 240 * (6 * math.pow(radius_longfloat, 2) + math.pow(r, 2)) * math.pow(xi_longfloat, 2)
                        - 360
                        * math.pow(r, 2)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(xi_longfloat, 4)
                        + 16
                        * (
                            -96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(2 * radius_longfloat + r, 4)
                        * math.pow(xi_longfloat, 8)
                    )
                )
                / 81920.0
            )

            scalar_h2[i] = (
                (
                    9
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -7)
                    * (
                        -45
                        - 78 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 28 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        + 32
                        * math.pow(radius_longfloat, 2)
                        * math.pow(xi_longfloat, 2)
                        * (
                            15
                            + 19 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                            + 10 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        )
                        - 56 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 4096.0
                + (
                    9
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (2 * radius_longfloat + r)
                        + 6
                        * (
                            -20 * r * math.pow(radius_longfloat, 2)
                            + 8 * math.pow(radius_longfloat, 3)
                            + 46 * radius_longfloat * math.pow(r, 2)
                            + 13 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        - 4
                        * (2 * radius_longfloat + r)
                        * (
                            -32 * r * math.pow(radius_longfloat, 3)
                            + 16 * math.pow(radius_longfloat, 4)
                            + 48 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 56 * radius_longfloat * math.pow(r, 3)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                        + 8
                        * (2 * radius_longfloat + r)
                        * (
                            16 * math.pow(radius_longfloat, 4)
                            + 16 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(-2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                + (
                    9
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (-2 * radius_longfloat + r)
                        - 6
                        * (
                            20 * r * math.pow(radius_longfloat, 2)
                            + 8 * math.pow(radius_longfloat, 3)
                            + 46 * radius_longfloat * math.pow(r, 2)
                            - 13 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        + 4
                        * (2 * radius_longfloat - r)
                        * (
                            32 * r * math.pow(radius_longfloat, 3)
                            + 16 * math.pow(radius_longfloat, 4)
                            + 48 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 56 * radius_longfloat * math.pow(r, 3)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                        - 8
                        * (2 * radius_longfloat - r)
                        * (
                            16 * math.pow(radius_longfloat, 4)
                            + 16 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                - (
                    9
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            - 6 * math.pow(r, 2)
                            + 9
                            * math.pow(r, 2)
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (
                                256 * math.pow(radius_longfloat, 8)
                                + 128 * math.pow(radius_longfloat, 6) * math.pow(r, 2)
                                - 40 * math.pow(radius_longfloat, 2) * math.pow(r, 6)
                                + 7 * math.pow(r, 8)
                            )
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                - (
                    9
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            - 6 * math.pow(r, 2)
                            + 9
                            * math.pow(r, 2)
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (
                                256 * math.pow(radius_longfloat, 8)
                                + 128 * math.pow(radius_longfloat, 6) * math.pow(r, 2)
                                - 40 * math.pow(radius_longfloat, 2) * math.pow(r, 6)
                                + 7 * math.pow(r, 8)
                            )
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                - (
                    9
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            6 * math.pow(r, 2)
                            - 9 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 14 * math.pow(r, 8) * math.pow(xi_longfloat, 6)
                            + 4
                            * math.pow(radius_longfloat, 2)
                            * (
                                -15
                                - 9 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 20 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                            )
                        )
                    )
                )
                / 8192.0
            )

            scalar_h3[i] = (
                (
                    9
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -7)
                    * (
                        -45
                        + 18 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        - 4 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        + 32
                        * math.pow(radius_longfloat, 2)
                        * math.pow(xi_longfloat, 2)
                        * (
                            15
                            + math.pow(r, 2) * math.pow(xi_longfloat, 2)
                            - 2 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        )
                        + 8 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 4096.0
                + (
                    9
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (2 * radius_longfloat + r)
                        + 6 * (2 * radius_longfloat - 3 * r) * math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)
                        - 4
                        * math.pow(2 * radius_longfloat - r, 3)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(xi_longfloat, 4)
                        + 8
                        * math.pow(2 * radius_longfloat - r, 3)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                + (
                    9
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (-2 * radius_longfloat + r)
                        - 6 * (2 * radius_longfloat + 3 * r) * math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)
                        + 4
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(2 * radius_longfloat + r, 3)
                        * math.pow(xi_longfloat, 4)
                        - 8
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(-2 * radius_longfloat + r, 2)
                        * math.pow(2 * radius_longfloat + r, 3)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                - (
                    9
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            + 6 * math.pow(r, 2)
                            - 3
                            * math.pow(r, 2)
                            * (12 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(4 * math.pow(radius_longfloat, 2) - math.pow(r, 2), 3)
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                - (
                    9
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            + 6 * math.pow(r, 2)
                            - 3
                            * math.pow(r, 2)
                            * (12 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(4 * math.pow(radius_longfloat, 2) - math.pow(r, 2), 3)
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                + (
                    9
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            6 * math.pow(r, 2)
                            - 3 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 2 * math.pow(r, 8) * math.pow(xi_longfloat, 6)
                            + 4
                            * math.pow(radius_longfloat, 2)
                            * (
                                15
                                - 9 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 4 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                            )
                        )
                    )
                )
                / 8192.0
            )

        elif r == 2 * radius_longfloat:
            ident_minus_rr[i] = -(
                (
                    math.pow(radius_longfloat, -5)
                    * (3 + 16 * radius_longfloat * xi_longfloat * math.pow(PI, -0.5))
                    * math.pow(xi_longfloat, -4)
                )
                / 2048.0
                + (
                    3
                    * math.erfc(2 * radius_longfloat * xi_longfloat)
                    * math.pow(radius_longfloat, -5)
                    * (-192 * math.pow(radius_longfloat, 4) + math.pow(xi_longfloat, -4))
                )
                / 1024.0
                + math.erfc(4 * radius_longfloat * xi_longfloat)
                * (math.pow(radius_longfloat, -1) - (3 * math.pow(radius_longfloat, -5) * math.pow(xi_longfloat, -4)) / 2048.0)
                + (
                    math.exp(-16 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(xi_longfloat, -3)
                    * (-1 - 64 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2))
                )
                / 256.0
                + (
                    3
                    * math.exp(-4 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(xi_longfloat, -3)
                    * (1 + 24 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2))
                )
                / 256.0
            )

            rr[i] = (
                (
                    math.pow(radius_longfloat, -5)
                    * (3 + 16 * radius_longfloat * xi_longfloat * math.pow(PI, -0.5))
                    * math.pow(xi_longfloat, -4)
                )
                / 1024.0
                + math.erfc(2 * radius_longfloat * xi_longfloat)
                * (
                    (-3 * math.pow(radius_longfloat, -1)) / 8.0
                    - (3 * math.pow(radius_longfloat, -5) * math.pow(xi_longfloat, -4)) / 512.0
                )
                + math.erfc(4 * radius_longfloat * xi_longfloat)
                * (math.pow(radius_longfloat, -1) + (3 * math.pow(radius_longfloat, -5) * math.pow(xi_longfloat, -4)) / 1024.0)
                + (
                    math.exp(-16 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(xi_longfloat, -3)
                    * (1 - 32 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2))
                )
                / 128.0
                + (
                    3
                    * math.exp(-4 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(xi_longfloat, -3)
                    * (-1 + 8 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2))
                )
                / 128.0
            )

            scalar_g1[i] = (
                (
                    math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -5)
                    * (
                        9
                        + 15 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        - 30 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                    )
                )
                / 64.0
                + (
                    math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        18 * radius_longfloat
                        - 45 * r
                        - 3
                        * (2 * radius_longfloat + r)
                        * (-16 * radius_longfloat * r + 8 * math.pow(radius_longfloat, 2) + 25 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        + 6
                        * (2 * radius_longfloat + r)
                        * (
                            -32 * r * math.pow(radius_longfloat, 3)
                            + 32 * math.pow(radius_longfloat, 4)
                            + 44 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 36 * radius_longfloat * math.pow(r, 3)
                            + 25 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        -9 * (2 * radius_longfloat + 5 * r)
                        + 3
                        * (2 * radius_longfloat - r)
                        * (16 * radius_longfloat * r + 8 * math.pow(radius_longfloat, 2) + 25 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        - 6
                        * (2 * radius_longfloat - r)
                        * (
                            32 * r * math.pow(radius_longfloat, 3)
                            + 32 * math.pow(radius_longfloat, 4)
                            + 44 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 36 * radius_longfloat * math.pow(r, 3)
                            + 25 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        3
                        + 3 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 20 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 128.0
                - (
                    3
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        15
                        + 5
                        * math.pow(r, 2)
                        * math.pow(xi_longfloat, 2)
                        * (3 + 64 * math.pow(radius_longfloat, 4) * math.pow(xi_longfloat, 4))
                        + 512 * math.pow(radius_longfloat, 6) * math.pow(xi_longfloat, 6)
                        - 256 * radius_longfloat * math.pow(r, 5) * math.pow(xi_longfloat, 6)
                        + 100 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
                - (
                    3
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        15
                        + 5
                        * math.pow(r, 2)
                        * math.pow(xi_longfloat, 2)
                        * (3 + 64 * math.pow(radius_longfloat, 4) * math.pow(xi_longfloat, 4))
                        + 512 * math.pow(radius_longfloat, 6) * math.pow(xi_longfloat, 6)
                        + 256 * radius_longfloat * math.pow(r, 5) * math.pow(xi_longfloat, 6)
                        + 100 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
            )

            scalar_g2[i] = (
                (
                    -3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -5)
                    * (
                        3
                        - math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 2 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                    )
                )
                / 64.0
                + (
                    math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        18 * radius_longfloat
                        + 45 * r
                        - 3
                        * (
                            24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            + 5 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        + 6
                        * (
                            24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            + 5 * math.pow(r, 3)
                        )
                        * math.pow(-2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        -18 * radius_longfloat
                        + 45 * r
                        + 3
                        * (
                            -24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            - 5 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        - 6
                        * (
                            -24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            - 5 * math.pow(r, 3)
                        )
                        * math.pow(2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    3
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        15
                        - 15 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 4
                        * (
                            128 * math.pow(radius_longfloat, 6)
                            - 80 * math.pow(radius_longfloat, 4) * math.pow(r, 2)
                            + 16 * radius_longfloat * math.pow(r, 5)
                            - 5 * math.pow(r, 6)
                        )
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        -3
                        + 3 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 4 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 128.0
                - (
                    3
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        -15
                        + 15 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 4
                        * (
                            -128 * math.pow(radius_longfloat, 6)
                            + 80 * math.pow(radius_longfloat, 4) * math.pow(r, 2)
                            + 16 * radius_longfloat * math.pow(r, 5)
                            + 5 * math.pow(r, 6)
                        )
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
            )

            scalar_h1[i] = (
                (
                    3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -7)
                    * (
                        27
                        - 2
                        * math.pow(xi_longfloat, 2)
                        * (
                            15 * math.pow(r, 2)
                            + 2 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 4 * math.pow(r, 6) * math.pow(xi_longfloat, 4)
                            + 48
                            * math.pow(radius_longfloat, 2)
                            * (
                                3
                                - math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 2 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                            )
                        )
                    )
                )
                / 4096.0
                + (
                    3
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        270 * radius_longfloat
                        - 135 * r
                        + 6
                        * (2 * radius_longfloat + 5 * r)
                        * (12 * math.pow(radius_longfloat, 2) + 5 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        - 4
                        * (
                            144 * r * math.pow(radius_longfloat, 4)
                            + 96 * math.pow(radius_longfloat, 5)
                            + 64 * math.pow(radius_longfloat, 3) * math.pow(r, 2)
                            - 30 * radius_longfloat * math.pow(r, 4)
                            - 5 * math.pow(r, 5)
                        )
                        * math.pow(xi_longfloat, 4)
                        + 8
                        * math.pow(2 * radius_longfloat - r, 3)
                        * (
                            96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 40960.0
                + (
                    3
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        -135 * (2 * radius_longfloat + r)
                        - 6
                        * (2 * radius_longfloat - 5 * r)
                        * (12 * math.pow(radius_longfloat, 2) + 5 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        + 4
                        * (
                            -144 * r * math.pow(radius_longfloat, 4)
                            + 96 * math.pow(radius_longfloat, 5)
                            + 64 * math.pow(radius_longfloat, 3) * math.pow(r, 2)
                            - 30 * radius_longfloat * math.pow(r, 4)
                            + 5 * math.pow(r, 5)
                        )
                        * math.pow(xi_longfloat, 4)
                        - 8
                        * (
                            -96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(2 * radius_longfloat + r, 3)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 40960.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        27
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            -6 * math.pow(r, 2)
                            + 9 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 2 * math.pow(r, 8) * math.pow(xi_longfloat, 6)
                            + 12
                            * math.pow(radius_longfloat, 2)
                            * (
                                -3
                                + 3 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 4 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                            )
                        )
                    )
                )
                / 8192.0
                + (
                    3
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -135
                        + 240 * (6 * math.pow(radius_longfloat, 2) + math.pow(r, 2)) * math.pow(xi_longfloat, 2)
                        - 360
                        * math.pow(r, 2)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(xi_longfloat, 4)
                        + 16
                        * (
                            96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(-2 * radius_longfloat + r, 4)
                        * math.pow(xi_longfloat, 8)
                    )
                )
                / 81920.0
                + (
                    3
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -135
                        + 240 * (6 * math.pow(radius_longfloat, 2) + math.pow(r, 2)) * math.pow(xi_longfloat, 2)
                        - 360
                        * math.pow(r, 2)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(xi_longfloat, 4)
                        + 16
                        * (
                            -96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(2 * radius_longfloat + r, 4)
                        * math.pow(xi_longfloat, 8)
                    )
                )
                / 81920.0
            )

            scalar_h2[i] = (
                (
                    9
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -7)
                    * (
                        -45
                        - 78 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 28 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        + 32
                        * math.pow(radius_longfloat, 2)
                        * math.pow(xi_longfloat, 2)
                        * (
                            15
                            + 19 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                            + 10 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        )
                        - 56 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 4096.0
                + (
                    9
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (2 * radius_longfloat + r)
                        + 6
                        * (
                            -20 * r * math.pow(radius_longfloat, 2)
                            + 8 * math.pow(radius_longfloat, 3)
                            + 46 * radius_longfloat * math.pow(r, 2)
                            + 13 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        - 4
                        * (2 * radius_longfloat + r)
                        * (
                            -32 * r * math.pow(radius_longfloat, 3)
                            + 16 * math.pow(radius_longfloat, 4)
                            + 48 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 56 * radius_longfloat * math.pow(r, 3)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                        + 8
                        * (2 * radius_longfloat + r)
                        * (
                            16 * math.pow(radius_longfloat, 4)
                            + 16 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(-2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                + (
                    9
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (-2 * radius_longfloat + r)
                        - 6
                        * (
                            20 * r * math.pow(radius_longfloat, 2)
                            + 8 * math.pow(radius_longfloat, 3)
                            + 46 * radius_longfloat * math.pow(r, 2)
                            - 13 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        + 4
                        * (2 * radius_longfloat - r)
                        * (
                            32 * r * math.pow(radius_longfloat, 3)
                            + 16 * math.pow(radius_longfloat, 4)
                            + 48 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 56 * radius_longfloat * math.pow(r, 3)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                        - 8
                        * (2 * radius_longfloat - r)
                        * (
                            16 * math.pow(radius_longfloat, 4)
                            + 16 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                - (
                    9
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            - 6 * math.pow(r, 2)
                            + 9
                            * math.pow(r, 2)
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (
                                256 * math.pow(radius_longfloat, 8)
                                + 128 * math.pow(radius_longfloat, 6) * math.pow(r, 2)
                                - 40 * math.pow(radius_longfloat, 2) * math.pow(r, 6)
                                + 7 * math.pow(r, 8)
                            )
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                - (
                    9
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            - 6 * math.pow(r, 2)
                            + 9
                            * math.pow(r, 2)
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (
                                256 * math.pow(radius_longfloat, 8)
                                + 128 * math.pow(radius_longfloat, 6) * math.pow(r, 2)
                                - 40 * math.pow(radius_longfloat, 2) * math.pow(r, 6)
                                + 7 * math.pow(r, 8)
                            )
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                - (
                    9
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            6 * math.pow(r, 2)
                            - 9 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 14 * math.pow(r, 8) * math.pow(xi_longfloat, 6)
                            + 4
                            * math.pow(radius_longfloat, 2)
                            * (
                                -15
                                - 9 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 20 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                            )
                        )
                    )
                )
                / 8192.0
            )

            scalar_h3[i] = (
                (
                    9
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -7)
                    * (
                        -45
                        + 18 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        - 4 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        + 32
                        * math.pow(radius_longfloat, 2)
                        * math.pow(xi_longfloat, 2)
                        * (
                            15
                            + math.pow(r, 2) * math.pow(xi_longfloat, 2)
                            - 2 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        )
                        + 8 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 4096.0
                + (
                    9
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (2 * radius_longfloat + r)
                        + 6 * (2 * radius_longfloat - 3 * r) * math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)
                        - 4
                        * math.pow(2 * radius_longfloat - r, 3)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(xi_longfloat, 4)
                        + 8
                        * math.pow(2 * radius_longfloat - r, 3)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                + (
                    9
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (-2 * radius_longfloat + r)
                        - 6 * (2 * radius_longfloat + 3 * r) * math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)
                        + 4
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(2 * radius_longfloat + r, 3)
                        * math.pow(xi_longfloat, 4)
                        - 8
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(-2 * radius_longfloat + r, 2)
                        * math.pow(2 * radius_longfloat + r, 3)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                - (
                    9
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            + 6 * math.pow(r, 2)
                            - 3
                            * math.pow(r, 2)
                            * (12 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(4 * math.pow(radius_longfloat, 2) - math.pow(r, 2), 3)
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                - (
                    9
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            + 6 * math.pow(r, 2)
                            - 3
                            * math.pow(r, 2)
                            * (12 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(4 * math.pow(radius_longfloat, 2) - math.pow(r, 2), 3)
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                + (
                    9
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            6 * math.pow(r, 2)
                            - 3 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 2 * math.pow(r, 8) * math.pow(xi_longfloat, 6)
                            + 4
                            * math.pow(radius_longfloat, 2)
                            * (
                                15
                                - 9 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 4 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                            )
                        )
                    )
                )
                / 8192.0
            )

        elif r < 2 * radius_longfloat:
            ident_minus_rr[i] = (
                (-9 * r * math.pow(radius_longfloat, -2)) / 32
                + math.pow(radius_longfloat, -1)
                - (math.pow(radius_longfloat, 2) * math.pow(r, -3)) / 2
                - (3 * math.pow(r, -1)) / 4
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -2)
                    * math.pow(r, -3)
                    * (-12 * math.pow(r, 4) + math.pow(xi_longfloat, -4))
                )
                / 128
                + (
                    math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * (
                        -128 * math.pow(radius_longfloat, -1)
                        + 64 * math.pow(radius_longfloat, 2) * math.pow(r, -3)
                        + 96 * math.pow(r, -1)
                        + math.pow(radius_longfloat, -2) * (36 * r - 3 * math.pow(r, -3) * math.pow(xi_longfloat, -4))
                    )
                )
                / 256
                + (
                    math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * (
                        128 * math.pow(radius_longfloat, -1)
                        + 64 * math.pow(radius_longfloat, 2) * math.pow(r, -3)
                        + 96 * math.pow(r, -1)
                        + math.pow(radius_longfloat, -2) * (36 * r - 3 * math.pow(r, -3) * math.pow(xi_longfloat, -4))
                    )
                )
                / 256
                + (
                    3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -2)
                    * math.pow(xi_longfloat, -3)
                    * (1 + 6 * math.pow(r, 2) * math.pow(xi_longfloat, 2))
                )
                / 64
                + (
                    math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -3)
                    * (
                        8 * r * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2)
                        - 16 * math.pow(radius_longfloat, 3) * math.pow(xi_longfloat, 2)
                        + radius_longfloat * (2 - 28 * math.pow(r, 2) * math.pow(xi_longfloat, 2))
                        - 3 * (r + 6 * math.pow(r, 3) * math.pow(xi_longfloat, 2))
                    )
                )
                / 128
                + (
                    math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -3)
                    * (
                        8 * r * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2)
                        + 16 * math.pow(radius_longfloat, 3) * math.pow(xi_longfloat, 2)
                        + radius_longfloat * (-2 + 28 * math.pow(r, 2) * math.pow(xi_longfloat, 2))
                        - 3 * (r + 6 * math.pow(r, 3) * math.pow(xi_longfloat, 2))
                    )
                )
                / 128
            )

            rr[i] = (
                (
                    (2 * radius_longfloat + 3 * r)
                    * math.pow(radius_longfloat, -2)
                    * math.pow(2 * radius_longfloat - r, 3)
                    * math.pow(r, -3)
                )
                / 16.0
                + (
                    math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * (
                        -64 * math.pow(radius_longfloat, -1)
                        - 64 * math.pow(radius_longfloat, 2) * math.pow(r, -3)
                        + 96 * math.pow(r, -1)
                        + math.pow(radius_longfloat, -2) * (12 * r + 3 * math.pow(r, -3) * math.pow(xi_longfloat, -4))
                    )
                )
                / 128.0
                + (
                    math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * (
                        64 * math.pow(radius_longfloat, -1)
                        - 64 * math.pow(radius_longfloat, 2) * math.pow(r, -3)
                        + 96 * math.pow(r, -1)
                        + math.pow(radius_longfloat, -2) * (12 * r + 3 * math.pow(r, -3) * math.pow(xi_longfloat, -4))
                    )
                )
                / 128.0
                + (
                    3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -2)
                    * math.pow(xi_longfloat, -3)
                    * (-1 + 2 * math.pow(r, 2) * math.pow(xi_longfloat, 2))
                )
                / 32.0
                - (
                    (2 * radius_longfloat + 3 * r)
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -3)
                    * (
                        -1
                        - 8 * radius_longfloat * r * math.pow(xi_longfloat, 2)
                        + 8 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2)
                        + 2 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                    )
                )
                / 64.0
                + (
                    (2 * radius_longfloat - 3 * r)
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -2)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -3)
                    * (
                        -1
                        + 8 * radius_longfloat * r * math.pow(xi_longfloat, 2)
                        + 8 * math.pow(radius_longfloat, 2) * math.pow(xi_longfloat, 2)
                        + 2 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                    )
                )
                / 64.0
                - (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -2)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -4)
                    * (1 + 4 * math.pow(r, 4) * math.pow(xi_longfloat, 4))
                )
                / 64.0
            )

            scalar_g1[i] = (
                (-9 * math.pow(radius_longfloat, -4) * math.pow(r, -4) * math.pow(xi_longfloat, -6)) / 128.0
                - (9 * math.pow(radius_longfloat, -4) * math.pow(r, -2) * math.pow(xi_longfloat, -4)) / 128.0
                + (
                    math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -5)
                    * (
                        9
                        + 15 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        - 30 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                    )
                )
                / 64.0
                + (
                    math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        18 * radius_longfloat
                        - 45 * r
                        - 3
                        * (2 * radius_longfloat + r)
                        * (-16 * radius_longfloat * r + 8 * math.pow(radius_longfloat, 2) + 25 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        + 6
                        * (2 * radius_longfloat + r)
                        * (
                            -32 * r * math.pow(radius_longfloat, 3)
                            + 32 * math.pow(radius_longfloat, 4)
                            + 44 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 36 * radius_longfloat * math.pow(r, 3)
                            + 25 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        -9 * (2 * radius_longfloat + 5 * r)
                        + 3
                        * (2 * radius_longfloat - r)
                        * (16 * radius_longfloat * r + 8 * math.pow(radius_longfloat, 2) + 25 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        - 6
                        * (2 * radius_longfloat - r)
                        * (
                            32 * r * math.pow(radius_longfloat, 3)
                            + 32 * math.pow(radius_longfloat, 4)
                            + 44 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 36 * radius_longfloat * math.pow(r, 3)
                            + 25 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        3
                        + 3 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 20 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 128.0
                + (
                    3
                    * math.erfc(2 * radius_longfloat * xi_longfloat - r * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        15
                        + 5
                        * math.pow(r, 2)
                        * math.pow(xi_longfloat, 2)
                        * (3 + 64 * math.pow(radius_longfloat, 4) * math.pow(xi_longfloat, 4))
                        + 512 * math.pow(radius_longfloat, 6) * math.pow(xi_longfloat, 6)
                        - 256 * radius_longfloat * math.pow(r, 5) * math.pow(xi_longfloat, 6)
                        + 100 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
                - (
                    3
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        15
                        + 5
                        * math.pow(r, 2)
                        * math.pow(xi_longfloat, 2)
                        * (3 + 64 * math.pow(radius_longfloat, 4) * math.pow(xi_longfloat, 4))
                        + 512 * math.pow(radius_longfloat, 6) * math.pow(xi_longfloat, 6)
                        + 256 * radius_longfloat * math.pow(r, 5) * math.pow(xi_longfloat, 6)
                        + 100 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
            )

            scalar_g2[i] = (
                (-3 * r * math.pow(radius_longfloat, -3)) / 10.0
                - (12 * math.pow(radius_longfloat, 2) * math.pow(r, -4)) / 5.0
                + (3 * math.pow(r, -2)) / 2.0
                + (3 * math.pow(radius_longfloat, -4) * math.pow(r, 2)) / 32.0
                - (
                    3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -5)
                    * (
                        3
                        - math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 2 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                    )
                )
                / 64.0
                + (
                    math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        18 * radius_longfloat
                        + 45 * r
                        - 3
                        * (
                            24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            + 5 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        + 6
                        * (
                            24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            + 5 * math.pow(r, 3)
                        )
                        * math.pow(-2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -4)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -5)
                    * (
                        -18 * radius_longfloat
                        + 45 * r
                        + 3
                        * (
                            -24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            - 5 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        - 6
                        * (
                            -24 * r * math.pow(radius_longfloat, 2)
                            + 16 * math.pow(radius_longfloat, 3)
                            + 14 * radius_longfloat * math.pow(r, 2)
                            - 5 * math.pow(r, 3)
                        )
                        * math.pow(2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 4)
                    )
                )
                / 640.0
                + (
                    3
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        15
                        - 15 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 4
                        * (
                            128 * math.pow(radius_longfloat, 6)
                            - 80 * math.pow(radius_longfloat, 4) * math.pow(r, 2)
                            + 16 * radius_longfloat * math.pow(r, 5)
                            - 5 * math.pow(r, 6)
                        )
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        -3
                        + 3 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 4 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 128.0
                - (
                    3
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -4)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -6)
                    * (
                        -15
                        + 15 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 4
                        * (
                            -128 * math.pow(radius_longfloat, 6)
                            + 80 * math.pow(radius_longfloat, 4) * math.pow(r, 2)
                            + 16 * radius_longfloat * math.pow(r, 5)
                            + 5 * math.pow(r, 6)
                        )
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 1280.0
            )

            scalar_h1[i] = (
                (9 * r * math.pow(radius_longfloat, -4)) / 64.0
                - (3 * math.pow(radius_longfloat, -3)) / 10.0
                - (9 * math.pow(radius_longfloat, 2) * math.pow(r, -5)) / 10.0
                + (3 * math.pow(r, -3)) / 4.0
                - (3 * math.pow(radius_longfloat, -6) * math.pow(r, 3)) / 512.0
                + (
                    3
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -7)
                    * (
                        27
                        - 2
                        * math.pow(xi_longfloat, 2)
                        * (
                            15 * math.pow(r, 2)
                            + 2 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 4 * math.pow(r, 6) * math.pow(xi_longfloat, 4)
                            + 48
                            * math.pow(radius_longfloat, 2)
                            * (
                                3
                                - math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 2 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                            )
                        )
                    )
                )
                / 4096.0
                + (
                    3
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        270 * radius_longfloat
                        - 135 * r
                        + 6
                        * (2 * radius_longfloat + 5 * r)
                        * (12 * math.pow(radius_longfloat, 2) + 5 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        - 4
                        * (
                            144 * r * math.pow(radius_longfloat, 4)
                            + 96 * math.pow(radius_longfloat, 5)
                            + 64 * math.pow(radius_longfloat, 3) * math.pow(r, 2)
                            - 30 * radius_longfloat * math.pow(r, 4)
                            - 5 * math.pow(r, 5)
                        )
                        * math.pow(xi_longfloat, 4)
                        + 8
                        * math.pow(2 * radius_longfloat - r, 3)
                        * (
                            96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 40960.0
                + (
                    3
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        -135 * (2 * radius_longfloat + r)
                        - 6
                        * (2 * radius_longfloat - 5 * r)
                        * (12 * math.pow(radius_longfloat, 2) + 5 * math.pow(r, 2))
                        * math.pow(xi_longfloat, 2)
                        + 4
                        * (
                            -144 * r * math.pow(radius_longfloat, 4)
                            + 96 * math.pow(radius_longfloat, 5)
                            + 64 * math.pow(radius_longfloat, 3) * math.pow(r, 2)
                            - 30 * radius_longfloat * math.pow(r, 4)
                            + 5 * math.pow(r, 5)
                        )
                        * math.pow(xi_longfloat, 4)
                        - 8
                        * (
                            -96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(2 * radius_longfloat + r, 3)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 40960.0
                + (
                    3
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        27
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            -6 * math.pow(r, 2)
                            + 9 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 2 * math.pow(r, 8) * math.pow(xi_longfloat, 6)
                            + 12
                            * math.pow(radius_longfloat, 2)
                            * (
                                -3
                                + 3 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 4 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                            )
                        )
                    )
                )
                / 8192.0
                + (
                    3
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -135
                        + 240 * (6 * math.pow(radius_longfloat, 2) + math.pow(r, 2)) * math.pow(xi_longfloat, 2)
                        - 360
                        * math.pow(r, 2)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(xi_longfloat, 4)
                        + 16
                        * (
                            96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(-2 * radius_longfloat + r, 4)
                        * math.pow(xi_longfloat, 8)
                    )
                )
                / 81920.0
                + (
                    3
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -135
                        + 240 * (6 * math.pow(radius_longfloat, 2) + math.pow(r, 2)) * math.pow(xi_longfloat, 2)
                        - 360
                        * math.pow(r, 2)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(xi_longfloat, 4)
                        + 16
                        * (
                            -96 * r * math.pow(radius_longfloat, 3)
                            + 48 * math.pow(radius_longfloat, 4)
                            + 80 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 40 * radius_longfloat * math.pow(r, 3)
                            + 5 * math.pow(r, 4)
                        )
                        * math.pow(2 * radius_longfloat + r, 4)
                        * math.pow(xi_longfloat, 8)
                    )
                )
                / 81920.0
            )

            scalar_h2[i] = (
                (63 * r * math.pow(radius_longfloat, -4)) / 64.0
                - (3 * math.pow(radius_longfloat, -3)) / 2.0
                + (9 * math.pow(radius_longfloat, 2) * math.pow(r, -5)) / 2.0
                - (3 * math.pow(r, -3)) / 4.0
                - (33 * math.pow(radius_longfloat, -6) * math.pow(r, 3)) / 512.0
                + (9 * math.pow(radius_longfloat, -6) * math.pow(r, -3) * math.pow(xi_longfloat, -6)) / 128.0
                - (27 * math.pow(radius_longfloat, -4) * math.pow(r, -3) * math.pow(xi_longfloat, -4)) / 64.0
                + (
                    9
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -7)
                    * (
                        -45
                        - 78 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        + 28 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        + 32
                        * math.pow(radius_longfloat, 2)
                        * math.pow(xi_longfloat, 2)
                        * (
                            15
                            + 19 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                            + 10 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        )
                        - 56 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 4096.0
                + (
                    3
                    * math.erfc(2 * radius_longfloat * xi_longfloat - r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -3)
                    * math.pow(xi_longfloat, -6)
                    * (
                        -3
                        + 18
                        * math.pow(radius_longfloat, 2)
                        * math.pow(xi_longfloat, 2)
                        * (1 - 4 * math.pow(r, 4) * math.pow(xi_longfloat, 4))
                        + 128 * math.pow(radius_longfloat, 6) * math.pow(xi_longfloat, 6)
                        + 64 * math.pow(radius_longfloat, 3) * math.pow(r, 3) * math.pow(xi_longfloat, 6)
                        + 8 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 256.0
                + (
                    9
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (2 * radius_longfloat + r)
                        + 6
                        * (
                            -20 * r * math.pow(radius_longfloat, 2)
                            + 8 * math.pow(radius_longfloat, 3)
                            + 46 * radius_longfloat * math.pow(r, 2)
                            + 13 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        - 4
                        * (2 * radius_longfloat + r)
                        * (
                            -32 * r * math.pow(radius_longfloat, 3)
                            + 16 * math.pow(radius_longfloat, 4)
                            + 48 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            - 56 * radius_longfloat * math.pow(r, 3)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                        + 8
                        * (2 * radius_longfloat + r)
                        * (
                            16 * math.pow(radius_longfloat, 4)
                            + 16 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(-2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                + (
                    9
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (-2 * radius_longfloat + r)
                        - 6
                        * (
                            20 * r * math.pow(radius_longfloat, 2)
                            + 8 * math.pow(radius_longfloat, 3)
                            + 46 * radius_longfloat * math.pow(r, 2)
                            - 13 * math.pow(r, 3)
                        )
                        * math.pow(xi_longfloat, 2)
                        + 4
                        * (2 * radius_longfloat - r)
                        * (
                            32 * r * math.pow(radius_longfloat, 3)
                            + 16 * math.pow(radius_longfloat, 4)
                            + 48 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 56 * radius_longfloat * math.pow(r, 3)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(xi_longfloat, 4)
                        - 8
                        * (2 * radius_longfloat - r)
                        * (
                            16 * math.pow(radius_longfloat, 4)
                            + 16 * math.pow(radius_longfloat, 2) * math.pow(r, 2)
                            + 7 * math.pow(r, 4)
                        )
                        * math.pow(2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                - (
                    9
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            - 6 * math.pow(r, 2)
                            + 9
                            * math.pow(r, 2)
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (
                                256 * math.pow(radius_longfloat, 8)
                                + 128 * math.pow(radius_longfloat, 6) * math.pow(r, 2)
                                - 40 * math.pow(radius_longfloat, 2) * math.pow(r, 6)
                                + 7 * math.pow(r, 8)
                            )
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                + (
                    3
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        135
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            -6 * (30 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            + 9
                            * (4 * math.pow(radius_longfloat, 2) - 3 * math.pow(r, 2))
                            * math.pow(r, 2)
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (
                                -768 * math.pow(radius_longfloat, 8)
                                + 128 * math.pow(radius_longfloat, 6) * math.pow(r, 2)
                                + 256 * math.pow(radius_longfloat, 3) * math.pow(r, 5)
                                - 168 * math.pow(radius_longfloat, 2) * math.pow(r, 6)
                                + 11 * math.pow(r, 8)
                            )
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                - (
                    9
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            6 * math.pow(r, 2)
                            - 9 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 14 * math.pow(r, 8) * math.pow(xi_longfloat, 6)
                            + 4
                            * math.pow(radius_longfloat, 2)
                            * (
                                -15
                                - 9 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 20 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                            )
                        )
                    )
                )
                / 8192.0
            )

            scalar_h3[i] = (
                (9 * r * math.pow(radius_longfloat, -4)) / 64.0
                + (9 * math.pow(radius_longfloat, 2) * math.pow(r, -5)) / 2.0
                - (9 * math.pow(r, -3)) / 4.0
                - (9 * math.pow(radius_longfloat, -6) * math.pow(r, 3)) / 512.0
                + (
                    9
                    * math.exp(-(math.pow(r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -4)
                    * math.pow(xi_longfloat, -7)
                    * (
                        -45
                        + 18 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                        - 4 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        + 32
                        * math.pow(radius_longfloat, 2)
                        * math.pow(xi_longfloat, 2)
                        * (
                            15
                            + math.pow(r, 2) * math.pow(xi_longfloat, 2)
                            - 2 * math.pow(r, 4) * math.pow(xi_longfloat, 4)
                        )
                        + 8 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                    )
                )
                / 4096.0
                + (
                    9
                    * math.exp(-(math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (2 * radius_longfloat + r)
                        + 6 * (2 * radius_longfloat - 3 * r) * math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)
                        - 4
                        * math.pow(2 * radius_longfloat - r, 3)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(xi_longfloat, 4)
                        + 8
                        * math.pow(2 * radius_longfloat - r, 3)
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(2 * radius_longfloat + r, 2)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                + (
                    9
                    * math.exp(-(math.pow(-2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)))
                    * math.pow(radius_longfloat, -6)
                    * math.pow(PI, -0.5)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -7)
                    * (
                        45 * (-2 * radius_longfloat + r)
                        - 6 * (2 * radius_longfloat + 3 * r) * math.pow(2 * radius_longfloat + r, 2) * math.pow(xi_longfloat, 2)
                        + 4
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(2 * radius_longfloat + r, 3)
                        * math.pow(xi_longfloat, 4)
                        - 8
                        * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                        * math.pow(-2 * radius_longfloat + r, 2)
                        * math.pow(2 * radius_longfloat + r, 3)
                        * math.pow(xi_longfloat, 6)
                    )
                )
                / 8192.0
                - (
                    9
                    * math.erfc((-2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            + 6 * math.pow(r, 2)
                            - 3
                            * math.pow(r, 2)
                            * (12 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(4 * math.pow(radius_longfloat, 2) - math.pow(r, 2), 3)
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                - (
                    9
                    * math.erfc((2 * radius_longfloat + r) * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            60 * math.pow(radius_longfloat, 2)
                            + 6 * math.pow(r, 2)
                            - 3
                            * math.pow(r, 2)
                            * (12 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(xi_longfloat, 2)
                            + 2
                            * (4 * math.pow(radius_longfloat, 2) + math.pow(r, 2))
                            * math.pow(4 * math.pow(radius_longfloat, 2) - math.pow(r, 2), 3)
                            * math.pow(xi_longfloat, 6)
                        )
                    )
                )
                / 16384.0
                + (
                    9
                    * math.erfc(r * xi_longfloat)
                    * math.pow(radius_longfloat, -6)
                    * math.pow(r, -5)
                    * math.pow(xi_longfloat, -8)
                    * (
                        -45
                        + 8
                        * math.pow(xi_longfloat, 2)
                        * (
                            6 * math.pow(r, 2)
                            - 3 * math.pow(r, 4) * math.pow(xi_longfloat, 2)
                            - 2 * math.pow(r, 8) * math.pow(xi_longfloat, 6)
                            + 4
                            * math.pow(radius_longfloat, 2)
                            * (
                                15
                                - 9 * math.pow(r, 2) * math.pow(xi_longfloat, 2)
                                + 4 * math.pow(r, 6) * math.pow(xi_longfloat, 6)
                            )
                        )
                    )
                )
                / 8192.0
            )


    ewald_coefficients = np.zeros((2 * num_entries, 4))
    ewald_coefficients[0::2, 0] = ident_minus_rr  # UF1
    ewald_coefficients[0::2, 1] = rr  # UF2
    ewald_coefficients[0::2, 2] = scalar_g1 / 2  # UC1
    ewald_coefficients[0::2, 3] = -scalar_g2 / 2  # UC2
    ewald_coefficients[1::2, 0] = scalar_h1  # DC1
    ewald_coefficients[1::2, 1] = scalar_h2  # DC2
    ewald_coefficients[1::2, 2] = scalar_h3  # DC3
    
    return ewald_coefficients
