#include "../include/iterative.h"

using namespace Eigen;

//////////////////////////////////////////////////////////////////////////////
// JACOBI
//////////////////////////////////////////////////////////////////////////////

IterativeResult jacobi(
    const MatrixXd& A,
    const VectorXd& b,
    double tol,
    int maxiter)
{
    int n = b.size();

    VectorXd x =
        VectorXd::Zero(n);

    VectorXd D =
        A.diagonal();

    MatrixXd R =
        A - D.asDiagonal().toDenseMatrix();

    double res = 0.0;

    for(int k=0;k<maxiter;k++)
    {
        VectorXd xnew =
            (b - R*x).cwiseQuotient(D);

        res =
            (b - A*xnew).norm();

        if(res < tol)
        {
            return {
                xnew,
                k+1,
                res,
                true
            };
        }

        x = xnew;
    }

    return {
        x,
        maxiter,
        res,
        false
    };
}

//////////////////////////////////////////////////////////////////////////////
// GAUSS SEIDEL
//////////////////////////////////////////////////////////////////////////////

IterativeResult gaussSeidel(
    const MatrixXd& A,
    const VectorXd& b,
    double tol,
    int maxiter)
{
    int n = b.size();

    VectorXd x =
        VectorXd::Zero(n);

    double res = 0.0;

    for(int k=0;k<maxiter;k++)
    {
        VectorXd xold = x;

        for(int i=0;i<n;i++)
        {
            double s1 = 0.0;
            double s2 = 0.0;

            for(int j=0;j<i;j++)
                s1 += A(i,j)*x(j);

            for(int j=i+1;j<n;j++)
                s2 += A(i,j)*xold(j);

            x(i) =
                (b(i)-s1-s2)/A(i,i);
        }

        res =
            (b - A*x).norm();

        if(res < tol)
        {
            return {
                x,
                k+1,
                res,
                true
            };
        }
    }

    return {
        x,
        maxiter,
        res,
        false
    };
}

//////////////////////////////////////////////////////////////////////////////
// SYMMETRIC GAUSS SEIDEL
//////////////////////////////////////////////////////////////////////////////

IterativeResult symmetricGaussSeidel(
    const MatrixXd& A,
    const VectorXd& b,
    double tol,
    int maxiter)
{
    int n = b.size();

    VectorXd x =
        VectorXd::Zero(n);

    double res = 0.0;

    for(int k=0;k<maxiter;k++)
    {
        VectorXd xold = x;

        // Forward sweep

        for(int i=0;i<n;i++)
        {
            double s1 = 0.0;
            double s2 = 0.0;

            for(int j=0;j<i;j++)
                s1 += A(i,j)*x(j);

            for(int j=i+1;j<n;j++)
                s2 += A(i,j)*xold(j);

            x(i) =
                (b(i)-s1-s2)/A(i,i);
        }

        // Backward sweep

        for(int i=n-1;i>=0;i--)
        {
            double s1 = 0.0;
            double s2 = 0.0;

            for(int j=0;j<i;j++)
                s1 += A(i,j)*x(j);

            for(int j=i+1;j<n;j++)
                s2 += A(i,j)*x(j);

            x(i) =
                (b(i)-s1-s2)/A(i,i);
        }

        res =
            (b - A*x).norm();

        if(res < tol)
        {
            return {
                x,
                k+1,
                res,
                true
            };
        }
    }

    return {
        x,
        maxiter,
        res,
        false
    };
}

//////////////////////////////////////////////////////////////////////////////
// SOR
//////////////////////////////////////////////////////////////////////////////

IterativeResult sor(
    const MatrixXd& A,
    const VectorXd& b,
    double omega,
    double tol,
    int maxiter)
{
    int n = b.size();

    VectorXd x =
        VectorXd::Zero(n);

    double res = 0.0;

    for(int k=0;k<maxiter;k++)
    {
        VectorXd xold = x;

        for(int i=0;i<n;i++)
        {
            double s1 = 0.0;
            double s2 = 0.0;

            for(int j=0;j<i;j++)
                s1 += A(i,j)*x(j);

            for(int j=i+1;j<n;j++)
                s2 += A(i,j)*xold(j);

            x(i) =
                (1.0-omega)*xold(i)
                +
                omega*
                (b(i)-s1-s2)
                /A(i,i);
        }

        res =
            (b - A*x).norm();

        if(res < tol)
        {
            return {
                x,
                k+1,
                res,
                true
            };
        }
    }

    return {
        x,
        maxiter,
        res,
        false
    };
}