#include "../include/poisson.h"
#include "../include/preconditioners.h"
#include "../include/iterative.h"
#include "../include/direct.h"
#include "../include/utilities.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <functional>

using namespace Eigen;
using namespace std;

//////////////////////////////////////////////////////////////////////////////
// RUN ITERATIVE SOLVER
//////////////////////////////////////////////////////////////////////////////

SolverResult runIterativeSolver(
    const string& name,
    function<IterativeResult()> solver)
{
    auto start =
        chrono::high_resolution_clock::now();

    IterativeResult result =
        solver();

    auto end =
        chrono::high_resolution_clock::now();

    double elapsed =
        chrono::duration<double>(end-start).count();

    return {
        name,
        result.iterations,
        result.residual,
        elapsed,
        result.x,
        result.converged
    };
}

//////////////////////////////////////////////////////////////////////////////
// RUN DIRECT SOLVER
//////////////////////////////////////////////////////////////////////////////

SolverResult runDirectSolver(
    const string& name,
    function<DirectResult()> solver)
{
    auto start =
        chrono::high_resolution_clock::now();

    DirectResult result =
        solver();

    auto end =
        chrono::high_resolution_clock::now();

    double elapsed =
        chrono::duration<double>(end-start).count();

    return {
        name,
        result.iterations,
        result.residual,
        elapsed,
        result.x,
        result.converged
    };
}

//////////////////////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////////////////////

int main()
{
    cout << "\n====================================\n";
    cout << " CFD SOLVER & PRECONDITIONER TOOL\n";
    cout << "====================================\n";

    int nx;
    int ny;

    double tol;

    int maxiter;

    double omega;

    cout << "\nEnter nx : ";
    cin >> nx;

    cout << "Enter ny : ";
    cin >> ny;

    cout << "Enter tolerance : ";
    cin >> tol;

    cout << "Enter maximum iterations : ";
    cin >> maxiter;

    cout << "Enter SOR omega : ";
    cin >> omega;

    /////////////////////////////////////////////////////////////////////////
    // MATRIX
    /////////////////////////////////////////////////////////////////////////

    MatrixXd A =
        generateSPDMatrix(nx,ny);

    VectorXd b =
        VectorXd::Ones(nx*ny);

    /////////////////////////////////////////////////////////////////////////
    // PRECONDITIONING
    /////////////////////////////////////////////////////////////////////////

    cout << "\nPreconditioning Option\n";
    cout << "1. None\n";
    cout << "2. Left\n";
    cout << "3. Right\n";
    cout << "4. Split\n";

    int pc;

    cout << "Choice : ";
    cin >> pc;

    MatrixXd Ause;
    VectorXd buse;

    if(pc==1)
    {
        Ause=A;
        buse=b;
    }
    else if(pc==2)
    {
        MatrixXd D =
            A.diagonal().asDiagonal();

        auto result =
            leftPrecondition(A,b,D);

        Ause=result.Anew;
        buse=result.bnew;
    }
    else if(pc==3)
    {
        MatrixXd D =
            A.diagonal().asDiagonal();

        auto result =
            rightPrecondition(A,b,D);

        Ause=result.Anew;
        buse=result.bnew;
    }
    else if(pc==4)
    {
        MatrixXd D =
            A.diagonal().asDiagonal();

        MatrixXd M1 =
            D.cwiseSqrt();

        MatrixXd M2 =
            D.cwiseSqrt();

        auto result =
            splitPrecondition(
                A,
                b,
                M1,
                M2
            );

        Ause=result.Anew;
        buse=result.bnew;
    }
    else
    {
        cout << "Invalid choice\n";
        return 0;
    }

    /////////////////////////////////////////////////////////////////////////
    // RESULTS
    /////////////////////////////////////////////////////////////////////////

    vector<SolverResult> results;

    /////////////////////////////////////////////////////////////////////////
    // DIRECT
    /////////////////////////////////////////////////////////////////////////

    results.push_back(
        runDirectSolver(
            "Direct",
            [&]()
            {
                return solveDirect(
                    Ause,
                    buse
                );
            }
        )
    );

    /////////////////////////////////////////////////////////////////////////
    // JACOBI
    /////////////////////////////////////////////////////////////////////////

    results.push_back(
        runIterativeSolver(
            "Jacobi",
            [&]()
            {
                return jacobi(
                    Ause,
                    buse,
                    tol,
                    maxiter
                );
            }
        )
    );

    /////////////////////////////////////////////////////////////////////////
    // GS
    /////////////////////////////////////////////////////////////////////////

    results.push_back(
        runIterativeSolver(
            "Gauss-Seidel",
            [&]()
            {
                return gaussSeidel(
                    Ause,
                    buse,
                    tol,
                    maxiter
                );
            }
        )
    );

    /////////////////////////////////////////////////////////////////////////
    // SGS
    /////////////////////////////////////////////////////////////////////////

    results.push_back(
        runIterativeSolver(
            "SGS",
            [&]()
            {
                return symmetricGaussSeidel(
                    Ause,
                    buse,
                    tol,
                    maxiter
                );
            }
        )
    );

    /////////////////////////////////////////////////////////////////////////
    // SOR
    /////////////////////////////////////////////////////////////////////////

    results.push_back(
        runIterativeSolver(
            "SOR",
            [&]()
            {
                return sor(
                    Ause,
                    buse,
                    omega,
                    tol,
                    maxiter
                );
            }
        )
    );

    /////////////////////////////////////////////////////////////////////////
    // LU
    /////////////////////////////////////////////////////////////////////////

    results.push_back(
        runDirectSolver(
            "LU",
            [&]()
            {
                return solveLU(
                    Ause,
                    buse
                );
            }
        )
    );

    /////////////////////////////////////////////////////////////////////////
    // CHOLESKY
    /////////////////////////////////////////////////////////////////////////

    results.push_back(
        runDirectSolver(
            "Cholesky",
            [&]()
            {
                return solveCholesky(
                    Ause,
                    buse
                );
            }
        )
    );

    /////////////////////////////////////////////////////////////////////////
    // IC0
    /////////////////////////////////////////////////////////////////////////

    results.push_back(
        runDirectSolver(
            "IC(0)",
            [&]()
            {
                return solveIC0(
                    Ause,
                    buse,
                    tol,
                    maxiter
                );
            }
        )
    );

    /////////////////////////////////////////////////////////////////////////
    // ILU0
    /////////////////////////////////////////////////////////////////////////

    results.push_back(
        runDirectSolver(
            "ILU(0)",
            [&]()
            {
                return solveILU0(
                    Ause,
                    buse,
                    tol,
                    maxiter
                );
            }
        )
    );

    /////////////////////////////////////////////////////////////////////////
    // SUMMARY
    /////////////////////////////////////////////////////////////////////////

    printSummary(results);

    /////////////////////////////////////////////////////////////////////////
    // VIEW
    /////////////////////////////////////////////////////////////////////////

    cout << "\n1. Print vector\n";
    cout << "2. Print 2D field\n";

    string view;

    cout << "Choice : ";
    cin >> view;

    if(view=="1")
    {
        printVectorSolutions(results);
    }
    else if(view=="2")
    {
        printFieldSolutions(
            results,
            nx,
            ny
        );
    }

    return 0;
}