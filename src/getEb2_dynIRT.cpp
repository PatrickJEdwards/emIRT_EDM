// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>

using namespace Rcpp;

// // [[Rcpp::export()]]
void getEb2_dynIRT(arma::mat &Eb2,
                   const arma::mat &Eystar,      // N x J
                   const arma::mat &Ex,          // N x T
                   const arma::cube &Vb2,        // 2 x 2 x T
                   const arma::mat &bill_session,// J x 1
                   const arma::mat &mubeta,      // 2 x 1
                   const arma::mat &sigmabeta,   // 2 x 2
                   const arma::mat &ones_col,    // N x T (1 if in service at t, else 0)
                   const int J,
                   const arma::mat &Ep           // NEW: N x T, E[p_it]
                 ) {

  // In the written implementation, the sufficient stats are:
  //    Ea = sum_i y^dagger_{ijt}         (intercept part)
  //    Eb = sum_i x_{it} y^dagger_{ijt}  (slope part)
  // where y^dagger_{ijt} = E[y*_{ijt}] - E[p_{it}].

  int t, j;
  arma::mat Ex2;      // [ones, x_it] — i.e., \tilde{x}_{it}
  arma::vec ydag;     // NEW: y^\dagger_{·jt} for session t

#pragma omp parallel for private(j,t,Ex2)	
  for (j = 0; j < J; j++) {

    t = bill_session(j,0);                 // session index for item j

    // Column for x_it at time t
    Ex2 = Ex.col(t);

    // We cannot just use ones here, as we have to zero out legislators not present
    //    Prepend the service mask as the intercept column (1 for in-service, 0 otherwise)
    //    so trans(Ex2) * ydag produces [ sum_i ydag_{ijt},  sum_i x_it * ydag_{ijt} ].
    Ex2.insert_cols(0, ones_col.col(t));
    
    // NEW: construct y^dagger_{·jt} = E[y*_{·jt}] - E[p_{·t}], masking out-of-service cells
    //    Masking ensures out-of-service i do not contribute (same as your ones_col logic).
    ydag = Eystar.col(j) - ( Ep.col(t) % ones_col.col(t) );
    
    // Posterior mean for (\alpha_{jt}, \beta_{jt}) using the usual linear-Gaussian update:
    //    Eb2[j,·] = Vb2_t * ( Σ^{-1}_β μ_β + sum_i \tilde{x}_{it} y^dagger_{ijt} )
    Eb2.row(j) = trans(Vb2.slice(t) * (inv_sympd(sigmabeta) * mubeta + trans(Ex2) * ydag));
  }

  return;
    
}
