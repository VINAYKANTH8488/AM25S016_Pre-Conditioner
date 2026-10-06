#ifndef DIRECT_H
#define DIRECT_H

#include <Eigen/Dense>

struct DirectResult
{
    Eigen::VectorXd x;

    int iterations;

    double residual;

    bool converged;
};

DirectResult solveDirect(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b
);

DirectResult solveLU(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b
);

DirectResult solveCholesky(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b
);

DirectResult solveIC0(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double tol,
    int maxIter
);

DirectResult solveILU0(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double tol,
    int maxIter
);

#endif