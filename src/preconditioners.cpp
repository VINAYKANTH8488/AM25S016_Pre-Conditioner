#include "../include/preconditioners.h"

using namespace Eigen;

//////////////////////////////////////////////////////////////////////////////
// LEFT PRECONDITIONING
//
// Python:
//
// Anew = np.linalg.solve(M,A)
// bnew = np.linalg.solve(M,b)
//////////////////////////////////////////////////////////////////////////////

LeftPCResult leftPrecondition(
    const MatrixXd& A,
    const VectorXd& b,
    const MatrixXd& M)
{
    LeftPCResult result;

    PartialPivLU<MatrixXd> solver(M);

    result.Anew = solver.solve(A);
    result.bnew = solver.solve(b);

    return result;
}

//////////////////////////////////////////////////////////////////////////////
// RIGHT PRECONDITIONING
//
// Python:
//
// Minv=np.linalg.inv(M)
// Anew=A@Minv
//////////////////////////////////////////////////////////////////////////////

RightPCResult rightPrecondition(
    const MatrixXd& A,
    const VectorXd& b,
    const MatrixXd& M)
{
    RightPCResult result;

    result.Minv = M.inverse();

    result.Anew = A * result.Minv;

    result.bnew = b;

    return result;
}

//////////////////////////////////////////////////////////////////////////////
// SPLIT PRECONDITIONING
//
// Python:
//
// M1inv=np.linalg.inv(M1)
// M2inv=np.linalg.inv(M2)
//
// Anew=M1inv @ A @ M2inv
// bnew=M1inv @ b
//////////////////////////////////////////////////////////////////////////////

SplitPCResult splitPrecondition(
    const MatrixXd& A,
    const VectorXd& b,
    const MatrixXd& M1,
    const MatrixXd& M2)
{
    SplitPCResult result;

    MatrixXd M1inv = M1.inverse();

    result.M2inv = M2.inverse();

    result.Anew =
        M1inv *
        A *
        result.M2inv;

    result.bnew =
        M1inv *
        b;

    return result;
}