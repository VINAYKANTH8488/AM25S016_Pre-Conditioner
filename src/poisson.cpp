
#include "../include/poisson.h"

Eigen::MatrixXd generateSPDMatrix(int nx, int ny)
{
    int N = nx * ny;

    Eigen::MatrixXd A =
        Eigen::MatrixXd::Zero(N, N);

    auto idx = [nx](int i, int j)
    {
        return i + j * nx;
    };

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int p = idx(i, j);

            A(p, p) = 4.0;

            // West
            if (i > 0)
                A(p, idx(i - 1, j)) = -1.0;

            // East
            if (i < nx - 1)
                A(p, idx(i + 1, j)) = -1.0;

            // South
            if (j > 0)
                A(p, idx(i, j - 1)) = -1.0;

            // North
            if (j < ny - 1)
                A(p, idx(i, j + 1)) = -1.0;
        }
    }

    return A;
}
