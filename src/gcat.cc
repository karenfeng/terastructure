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

gsl_matrix *getMatrix(std::string file_name, long M_rows, long M_cols) {
  double tmp_ele;
  gsl_matrix *M = gsl_matrix_calloc(M_rows, M_cols);
  std::ifstream inFile(file_name);
  
  for(int i = 0; i < M_rows; i++) {
    for(int j = 0; j < M_cols; j++) {
      inFile >> tmp_ele;
      gsl_matrix_set(M, i, j, tmp_ele);
    }
  }
  return M;
}

// Reads GSL vector from file
gsl_vector *getVector(std::string file_name, long V_eles) {
  double tmp_ele;
  gsl_vector *V = gsl_vector_calloc(V_eles);
  std::ifstream inFile(file_name);
  
  for(int i = 0; i < V_eles; i++) {
    inFile >> tmp_ele;
    gsl_vector_set(V, i, tmp_ele);
  }
  return V;
}

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
    //printf("%f %f\n", y_i, p_i);
    dev += (y_i * log(p_i)) + ((1-y_i)*log(1-p_i));
  }
  gsl_vector_free(p);
  gsl_vector_free(f);
  gsl_matrix_free(W);
  gsl_matrix_free(Wo);
  gsl_permutation_free(W_permut);
  return dev;
}

double *getDiffDev(std::string lf_file, std::string geno_file, std::string trait_file, int tot_snp, int tot_indiv, int tot_lf, int max_iter, double tol) {
  // Difference of deviance
  double *diff_dev = (double*) malloc(sizeof(double)*tot_snp);
  // Deviance of null and alt models for log-likelihood
  double dev_null, dev_alt;
  // Iteration values
  int i, j;
  int y_ij; // Number of major alleles
  double X_ij;
  // GSL matrices and vectors for IRLS
  gsl_matrix *LF = getMatrix(lf_file, tot_indiv, tot_lf);
  gsl_vector *trait = getVector(trait_file, tot_indiv);
  gsl_matrix *X = gsl_matrix_calloc(2*tot_indiv, tot_lf+1); // Stacked LFs and trait
  gsl_matrix *X_null = gsl_matrix_calloc(2*tot_indiv, tot_lf); // Stacked LFs
  gsl_matrix *Y = getMatrix(geno_file, tot_snp, tot_indiv);
  gsl_vector *y = gsl_vector_calloc(2*tot_indiv); // Doubled Y
  gsl_vector *b_null = gsl_vector_calloc(tot_lf); // Beta for null
  gsl_vector *bl_null = gsl_vector_calloc(tot_lf); // Beta last for null
  gsl_vector *b_alt = gsl_vector_calloc(tot_lf+1); // Beta for alt
  gsl_vector *bl_alt = gsl_vector_calloc(tot_lf+1); // Beta last for alt
  
  // Stack 2 copies of traits and LFs to get X
  for(i = 0; i < tot_indiv; i++) {
    for(j = 0; j < tot_lf; j++) {
      X_ij = gsl_matrix_get(LF, i, j);
      // Set for X
      gsl_matrix_set(X, i, j, X_ij);
      gsl_matrix_set(X, i+tot_indiv, j, X_ij);
      // Set for X_null
      gsl_matrix_set(X_null, i, j, X_ij);
      gsl_matrix_set(X_null, i+tot_indiv, j, X_ij);
    }
    X_ij = gsl_vector_get(trait, i);
    gsl_matrix_set(X, i, tot_lf, X_ij);
    gsl_matrix_set(X, i+tot_indiv, tot_lf, X_ij);
  }
  for(i = 0; i < tot_snp; i++) {
    // Init doubled y for this SNP
    for(j = 0; j < tot_indiv; j++) {
      y_ij = gsl_matrix_get(Y, i, j);
      if (y_ij == 2) {
        gsl_vector_set(y, j, 1.0);
        gsl_vector_set(y, j+tot_indiv, 1.0);
      } else if(y_ij == 1) {
        gsl_vector_set(y, j, 1.0);
        gsl_vector_set(y, j+tot_indiv, 0.0);
      } else if(y_ij == 0) {
        gsl_vector_set(y, j, 0.0);
        gsl_vector_set(y, j+tot_indiv, 0.0);
      }
    }
    // Reset b_nulls to 0
    gsl_vector_set_zero(b_null);
    gsl_vector_set_zero(bl_null);
    // Calculate dev for null hypothesis
    dev_null = calcDevModel(X_null, y, b_null, bl_null, max_iter, tol);
    // Set b_alt
    for(j = 0; j < tot_lf; j++) {
      gsl_vector_set(b_alt, j, gsl_vector_get(b_null, j));
      gsl_vector_set(bl_alt, j, gsl_vector_get(bl_null, j));
    }
    gsl_vector_set(b_alt, tot_lf, 0);
    gsl_vector_set(bl_alt, tot_lf, 0);
    // Calculate dev for alt
    dev_alt = calcDevModel(X, y, b_alt, bl_alt, max_iter, tol);
    diff_dev[i] = -2*(dev_null - dev_alt);
  }
  // Clean up
  gsl_matrix_free(X);
  gsl_matrix_free(Y);
  gsl_vector_free(y);
  gsl_vector_free(bl_null);
  gsl_vector_free(b_null);
  gsl_vector_free(bl_alt);
  gsl_vector_free(b_alt);
  
  return diff_dev;
}

NumericVector assoc() {
  // Algorithmic constants
  int max_iter = 10;
  double tol = 1e-6;
  // Constants based on data set
  int tot_snp = 10000;
  int tot_indiv = 1000;
  int tot_lf = 3;
  // Result from calculations
  double *diffDev = log_reg::getDiffDev("data/LF.txt",
                                        "data/geno.txt",
                                        "data/trait.txt",
                                        tot_snp, tot_indiv, tot_lf, max_iter, tol);
  // Iterative constants
  int i = 0;
  NumericVector assoc(tot_snp);
  for(i = 0; i < tot_snp; i++) {
    assoc[i] = diffDev[i];
  }
  return assoc;
}