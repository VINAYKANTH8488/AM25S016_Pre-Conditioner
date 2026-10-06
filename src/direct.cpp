#include "../include/direct.h"

using namespace Eigen;

//////////////////////////////////////////////////////////////////////////////
// DIRECT
//////////////////////////////////////////////////////////////////////////////

DirectResult solveDirect(
    const MatrixXd& A,
    const VectorXd& b)
{
    VectorXd x =
        A.fullPivLu().solve(b);

    double res =
        (b - A*x).norm();

    return {
        x,
        1,
        res,
        true
    };
}

//////////////////////////////////////////////////////////////////////////////
// LU
//////////////////////////////////////////////////////////////////////////////

DirectResult solveLU(
    const MatrixXd& A,
    const VectorXd& b)
{
    PartialPivLU<MatrixXd> lu(A);

    VectorXd x =
        lu.solve(b);

    double res =
        (b - A*x).norm();

    return {
        x,
        1,
        res,
        true
    };
}

//////////////////////////////////////////////////////////////////////////////
// CHOLESKY
//////////////////////////////////////////////////////////////////////////////

DirectResult solveCholesky(
    const MatrixXd& A,
    const VectorXd& b)
{
    LLT<MatrixXd> chol(A);

    VectorXd x =
        chol.solve(b);

    double res =
        (b - A*x).norm();

    return {
        x,
        1,
        res,
        true
    };
}

//////////////////////////////////////////////////////////////////////////////
// IC(0)
//////////////////////////////////////////////////////////////////////////////

DirectResult solveIC0(
    const MatrixXd& A,
    const VectorXd& b,
    double tol,
    int maxIter)
{
    int n = A.rows();

    MatrixXd L =
        MatrixXd::Zero(n,n);

    //////////////////////////////////////////////////////////////
    // IC0 FACTORIZATION
    //////////////////////////////////////////////////////////////

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<=i;j++)
        {
            if(A(i,j)==0.0)
                continue;

            double s=0.0;

            for(int k=0;k<j;k++)
                s += L(i,k)*L(j,k);

            if(i==j)
            {
                L(i,j)=sqrt(A(i,i)-s);
            }
            else
            {
                L(i,j)=
                    (A(i,j)-s)
                    /
                    L(j,j);
            }
        }
    }

    //////////////////////////////////////////////////////////////
    // ITERATIVE REFINEMENT
    //////////////////////////////////////////////////////////////

    VectorXd x =
        VectorXd::Zero(n);

    bool converged=false;

    int iterations=0;

    double res=0.0;

    for(int k=0;k<maxIter;k++)
    {
        VectorXd r =
            b - A*x;

        res = r.norm();

        if(res < tol)
        {
            converged=true;
            break;
        }

        iterations++;

        VectorXd y =
            L.triangularView<Lower>().solve(r);

        VectorXd z =
            L.transpose()
             .triangularView<Upper>()
             .solve(y);

        x += z;
    }

    if(!converged)
    {
        res =
            (b - A*x).norm();
    }

    return {
        x,
        iterations,
        res,
        converged
    };
}

//////////////////////////////////////////////////////////////////////////////
// ILU(0)
//////////////////////////////////////////////////////////////////////////////

DirectResult solveILU0(
    const MatrixXd& A,
    const VectorXd& b,
    double tol,
    int maxIter)
{
    int n=A.rows();

    MatrixXd L =
        MatrixXd::Identity(n,n);

    MatrixXd U =
        MatrixXd::Zero(n,n);

    //////////////////////////////////////////////////////////////
    // ILU0 FACTORIZATION
    //////////////////////////////////////////////////////////////

    for(int i=0;i<n;i++)
    {
        // U

        for(int j=i;j<n;j++)
        {
            if(A(i,j)==0.0)
                continue;

            double s=0.0;

            for(int k=0;k<i;k++)
                s += L(i,k)*U(k,j);

            U(i,j)=A(i,j)-s;
        }

        // L

        for(int j=i+1;j<n;j++)
        {
            if(A(j,i)==0.0)
                continue;

            double s=0.0;

            for(int k=0;k<i;k++)
                s += L(j,k)*U(k,i);

            L(j,i)=
                (A(j,i)-s)
                /
                U(i,i);
        }
    }

    //////////////////////////////////////////////////////////////
    // ITERATIVE REFINEMENT
    //////////////////////////////////////////////////////////////

    VectorXd x =
        VectorXd::Zero(n);

    bool converged=false;

    int iterations=0;

    double res=0.0;

    for(int k=0;k<maxIter;k++)
    {
        VectorXd r =
            b - A*x;

        res =
            r.norm();

        if(res < tol)
        {
            converged=true;
            break;
        }

        iterations++;

        VectorXd y =
            L.triangularView<Lower>().solve(r);

        VectorXd z =
            U.triangularView<Upper>().solve(y);

        x += z;
    }

    if(!converged)
    {
        res =
            (b - A*x).norm();
    }

    return {
        x,
        iterations,
        res,
        converged
    };
}