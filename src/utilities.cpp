#include "../include/utilities.h"

#include <iostream>
#include <iomanip>

using namespace Eigen;
using namespace std;

//////////////////////////////////////////////////////////////////////////////
// Residual
//
// Python:
//
// np.linalg.norm(b - A @ x)
//////////////////////////////////////////////////////////////////////////////

double computeResidual(
    const MatrixXd& A,
    const VectorXd& x,
    const VectorXd& b)
{
    return (b - A * x).norm();
}

//////////////////////////////////////////////////////////////////////////////
// Summary Table
//////////////////////////////////////////////////////////////////////////////

void printSummary(
    const vector<SolverResult>& results)
{
    cout << "\n";

    cout << string(85, '=') << endl;
    cout << "SUMMARY" << endl;
    cout << string(85, '=') << endl;

    cout
        << left
        << setw(15) << "Method"
        << setw(12) << "Iterations"
        << setw(15) << "Residual"
        << setw(15) << "Time(s)"
        << setw(12) << "Converged"
        << endl;

    for (const auto& r : results)
    {
        cout
            << left
            << setw(15) << r.name
            << setw(12) << r.iterations
            << setw(15) << scientific << setprecision(4)
            << r.residual
            << setw(15) << fixed << setprecision(6)
            << r.elapsedTime
            << setw(12)
            << (r.converged ? "YES" : "NO")
            << endl;
    }

    cout << string(85, '=') << endl;
}

//////////////////////////////////////////////////////////////////////////////
// Print Vector
//////////////////////////////////////////////////////////////////////////////

void printVectorSolutions(
    const vector<SolverResult>& results)
{
    for (const auto& r : results)
    {
        cout << "\n"
             << r.name
             << "\n";

        cout << r.solution << endl;
    }
}

//////////////////////////////////////////////////////////////////////////////
// Print 2D Field
//////////////////////////////////////////////////////////////////////////////

void printFieldSolutions(
    const vector<SolverResult>& results,
    int nx,
    int ny)
{
    for (const auto& r : results)
    {
        cout << "\n"
             << r.name
             << "\n";

        MatrixXd phi(ny, nx);

        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                phi(j, i) =
                    r.solution(i + j * nx);
            }
        }

        cout
            << phi.format(
                IOFormat(
                    4,
                    0,
                    " ",
                    "\n",
                    "[",
                    "]"
                )
            )
            << endl;
    }
}