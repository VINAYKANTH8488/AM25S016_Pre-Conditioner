
#ifndef POISSON_H
#define POISSON_H

#include <Eigen/Dense>

/*
 * Generate the SPD matrix corresponding to the
 * 2D Poisson equation using a 5-point stencil.
 *
 * Equivalent to:
 * generate_spd_matrix(nx, ny)
 * in the Python code.
 */

Eigen::MatrixXd generateSPDMatrix(int nx, int ny);

#endif