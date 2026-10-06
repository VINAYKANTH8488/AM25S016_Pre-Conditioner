#ifndef ITERATIVE_H
#define ITERATIVE_H

#include <Eigen/Dense>

struct IterativeResult
{
    Eigen::VectorXd x;

    int iterations;

    double residual;

    bool converged;
};

IterativeResult jacobi(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double tol,
    int maxiter
);

IterativeResult gaussSeidel(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double tol,
    int maxiter
);

IterativeResult symmetricGaussSeidel(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double tol,
    int maxiter
);

IterativeResult sor(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double omega,
    double tol,
    int maxiter
);

#endif