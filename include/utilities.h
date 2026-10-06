#ifndef UTILITIES_H
#define UTILITIES_H

#include <Eigen/Dense>
#include <string>
#include <vector>

struct SolverResult
{
    std::string name;

    int iterations;

    double residual;

    double elapsedTime;

    Eigen::VectorXd solution;

    bool converged;
};

double computeResidual(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& b
);

void printSummary(
    const std::vector<SolverResult>& results
);

void printVectorSolutions(
    const std::vector<SolverResult>& results
);

void printFieldSolutions(
    const std::vector<SolverResult>& results,
    int nx,
    int ny
);

#endif