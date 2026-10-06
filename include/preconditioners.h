#ifndef PRECONDITIONERS_H
#define PRECONDITIONERS_H

#include <Eigen/Dense>

struct LeftPCResult
{
    Eigen::MatrixXd Anew;
    Eigen::VectorXd bnew;
};

struct RightPCResult
{
    Eigen::MatrixXd Anew;
    Eigen::VectorXd bnew;
    Eigen::MatrixXd Minv;
};

struct SplitPCResult
{
    Eigen::MatrixXd Anew;
    Eigen::VectorXd bnew;
    Eigen::MatrixXd M2inv;
};

LeftPCResult leftPrecondition(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& M
);

RightPCResult rightPrecondition(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& M
);

SplitPCResult splitPrecondition(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& M1,
    const Eigen::MatrixXd& M2
);

#endif