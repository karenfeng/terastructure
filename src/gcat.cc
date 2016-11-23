#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>

// Calculates deviance of a model using IRLS (iteratively-reweighted least squares)
double calcDevModel(const gsl_matrix *X, const gsl_vector *y, gsl_vector *b,
                    gsl_vector *bl, int max_iter, double tol) {
  // Deviance of model
  double dev = 0;
  // Stopping condition
  double max_rel_change;
  double rel_change;
  // Utility sizes
  long X_rows = X->size1;
  long X_cols = X->size2;
  // GSL matrices and vectors for calculations
  gsl_vector *p = gsl_vector_calloc(X_rows); // MLE
  gsl_vector *f = gsl_vector_calloc(X_cols);
  gsl_matrix *W = gsl_matrix_calloc(X_cols, X_cols);
  gsl_matrix *Wo = gsl_matrix_calloc(X_cols, X_cols); // Inverted W
  gsl_permutation *W_permut = gsl_permutation_calloc(X_cols);
  // Values for iteration
  int signum;
  long i, j, k;
  int n_iter;
  double y_i;
  double bl_i;
  double p_i, p_k;
  double w_ij;
  double f_i;
  
  for(n_iter = 0; n_iter < max_iter; n_iter++) {
    // p <- as.vector(1/(1 + exp(-X %*% b)))
    gsl_blas_dgemv(CblasNoTrans, -1.0, X, b, 0.0, p);
    for(i = 0; i < X_rows; i++) {
      gsl_vector_set(p, i, 1/(1+exp(gsl_vector_get(p, i))));
    }
    // var.b <- solve(crossprod(X, p * (1 - p) * X))
    gsl_matrix_set_zero(W);
    for(i = 0; i < X_cols; i++) {
      for(j = i; j < X_cols; j++) {
        for(k = 0; k < X_rows; k++) {
          p_k = gsl_vector_get(p, k);
          w_ij = gsl_matrix_get(W, i, j) +
            (gsl_matrix_get(X, k, i) *
            gsl_matrix_get(X, k, j) *
            p_k * (1-p_k));
          gsl_matrix_set(W, i, j, w_ij);
        }
        // Reflect (symmetry)
        if (i != j) {
          gsl_matrix_set(W, j, i, gsl_matrix_get(W, i, j));
        }
      }
    }
    gsl_linalg_LU_decomp(W, W_permut, &signum);
    gsl_linalg_LU_invert(W, W_permut, Wo);
    // b = b + Wo %*% X*(y-p)
    gsl_vector_set_zero(f);
    for(i = 0; i < X_cols; i++) {
      for(j = 0; j < X_rows; j++) {
        f_i = gsl_vector_get(f, i) + gsl_matrix_get(X, j, i) * (gsl_vector_get(y, j) - gsl_vector_get(p, j));
        gsl_vector_set(f, i, f_i);
      }
    }
    gsl_blas_dgemv(CblasNoTrans, 1.0, Wo, f, 1.0, b);
    // Stopping condition
    max_rel_change = 0;
    for(i = 0; i < X_cols; i++) {
      bl_i = gsl_vector_get(bl, i);
      rel_change = fabs(gsl_vector_get(b, i) - bl_i) / (fabs(bl_i) + 0.01*tol);
      if (rel_change > max_rel_change) {
        max_rel_change = rel_change;
      }
    }
    if (max_rel_change < tol) {
      break;
    }
    gsl_vector_memcpy(bl, b);
  }
  // Calculate deviance
  for(i = 0; i < X_rows; i++) {
    y_i = gsl_vector_get(y, i);
    p_i = gsl_vector_get(p, i);
    dev += (y_i * log(p_i)) + ((1-y_i)*log(1-p_i));
  }

  // Clean up
  gsl_vector_free(p);
  gsl_vector_free(f);
  gsl_matrix_free(W);
  gsl_matrix_free(Wo);
  gsl_permutation_free(W_permut);

  return dev;
}

// TODO: allow for user-inputted covariates (in X)
// Calculates log odds 
double getDiffDev(gsl_vector *y, gsl_vector *pi, gsl_vector *trait, uint32_t n, int max_iter, double tol) {
  // Difference of deviance
  double diff_dev;
  // Deviance of null and alt models for log-likelihood
  double dev_null, dev_alt;
  // Iteration values
  int i, j;
  int y_i; // Number of major alleles
  double trait_i;
  // GSL matrices and vectors for IRLS
  gsl_matrix *X_null = gsl_matrix_calloc(2*tot_indiv, 1); // Intercept
  gsl_matrix *X_alt = gsl_matrix_calloc(2*tot_indiv, 2); // Intercept and trait
  gsl_vector *y_dbl = gsl_vector_calloc(2*tot_indiv); // Doubled genotypes
  gsl_vector *b_null = gsl_vector_calloc(1); // Beta for null
  gsl_vector *bl_null = gsl_vector_calloc(1); // Beta last for null
  gsl_vector *b_alt = gsl_vector_malloc(2); // Beta for alt
  gsl_vector *bl_alt = gsl_vector_malloc(2); // Beta last for alt
  
  // Set values of X
  for(i = 0; i < tot_indiv; i++) {
    gsl_matrix_set(X_null, i, 0, 1);
    gsl_matrix_set(X_null, i+tot_indiv, 0, 1);
    gsl_matrix_set(X_alt, i, 0, 1);
    gsl_matrix_set(X_alt, i+tot_indiv, 0, 1);
    trait_i = gsl_vector_get(trait, i);
    gsl_matrix_set(X_alt, i, 1, trait_i);
    gsl_matrix_set(X_alt, i+tot_indiv, 1, trait_i);
  }

  // Double genotype and subtract population structure offset
  for(i = 0; i < n; i++) {
    y_i = gsl_vector_get(y, i);
    pi_i = gsl_vector_get(pi, i);
    if (y_i == 2) {
      gsl_vector_set(y_dbl, j, 1.0 - pi_i);
      gsl_vector_set(y_dbl, j+tot_indiv, 1.0 - pi_i);
    } else if(y_i == 1) {
      gsl_vector_set(y_dbl, j, 1.0 - pi_i);
      gsl_vector_set(y_dbl, j+tot_indiv, 0.0 - pi_i);
    } else if(y_i == 0) {
      gsl_vector_set(y_dbl, j, 0.0 - pi_i);
      gsl_vector_set(y_dbl, j+tot_indiv, 0.0 - pi_i);
    }
  }

  // Calculate deviance for null model
  dev_null = calcDevModel(X_null, y_dbl, b_null, bl_null, max_iter, tol);

  // Set b_alt
  gsl_vector_set(b_alt, 0, gsl_vector_get(b_null, 0));
  gsl_vector_set(bl_alt, 0, gsl_vector_get(bl_null, 0));
  gsl_vector_set(b_alt, 1, 0);
  gsl_vector_set(bl_alt, 1, 0);

  // Calculate dev for alt model
  dev_alt = calcDevModel(X, y_dbl, b_alt, bl_alt, max_iter, tol);
  diff_dev = -2*(dev_null - dev_alt);

  // Clean up
  gsl_matrix_free(X_alt);
  gsl_matrix_free(X_null);
  gsl_vector_free(y_dbl);
  gsl_vector_free(b_null);
  gsl_vector_free(bl_null);
  gsl_vector_free(b_alt);
  gsl_vector_free(bl_alt);
  
  return diff_dev;
}

// Runs GCAT on one SNP location
double gcat(const double *y, const double *pi, const double *trait, uint32_t n) {
  // Convert matrices/vectors into GSL types
  gsl_vector *trait_gsl = gsl_vector_malloc(n);
  gsl_vector *y_gsl = gsl_vector_malloc(n);
  gsl_vector *pi_gsl = gsl_vector_malloc(n);

  for(uint32_t i = 0; i < n; i++) {
    gsl_vector_set(trait_gsl, i, trait[i]);
    gsl_vector_set(y_gsl, i, y[i]);
    gsl_vector_set(pi_gsl, i, pi[i])
  }

  // Algorithmic constants - should be built into env
  int max_iter = 10;
  double tol = 1e-6;
  // Result from calculations
  double diff_dev = getDiffDev(y_gsl, pi_gsl, trait_gsl, n, max_iter, tol);

  // Clean up
  gsl_vector_free(trait_gsl);
  gsl_vector_free(y_gsl);
  gsl_vector_free(pi_gsl);

  return diff_dev;
}