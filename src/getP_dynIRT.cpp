// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>
//#include <RcppTN.h>
#include "getP_dynIRT.h"

using namespace Rcpp;

// // [[Rcpp::export()]]



// Updates Ep (means) and Vp (variances) for propensities p_it.
// Conjugate Normal update per (i,t), then enforce sum_i p_it = 0 within each t.
void getP_dynIRT(arma::mat &Ep,                 // N x T (updated)
                 arma::mat &Vp,                 // N x T (updated)
                 const arma::mat &Eystar,       // N x J
                 const arma::mat &alpha,        // J x 1
                 const arma::mat &beta,         // J x 1
                 const arma::mat &x,            // N x T
                 const arma::mat &bill_session, // J x 1
                 const arma::mat &startlegis,   // N x 1
                 const arma::mat &endlegis,     // N x 1
                 const double pmu,              // prior mean
                 const double psigma,           // prior variance
                 const unsigned int T,
                 const unsigned int N,
                 const unsigned int J) {
  
  // Prior precision (psigma is a variance)
  const double prior_prec = 1.0 / psigma;
  
  // Unconstrained per-(i,t) updates
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int t = 0; t < T; ++t) {

      // Skip out-of-service cells
      if (t < (unsigned) startlegis(i,0) || t > (unsigned) endlegis(i,0)) {
        Ep(i,t) = 0.0;
        Vp(i,t) = 0.0;
        continue;
      }

      double sum_r = 0.0;
      unsigned int n_it = 0;

      // Accumulate residuals r_ijt over items in session t
      for (unsigned int j = 0; j < J; ++j) {
        if (bill_session(j,0) == static_cast<double>(t)) {
          // r_ijt = E[y*_ijt] - E[alpha_jt] - E[beta_jt] * E[x_it]
          const double r_ijt = Eystar(i,j) - alpha(j,0) - beta(j,0) * x(i,t);
          sum_r += r_ijt;
          ++n_it;
        }
      }

      // With your data, n_it >= 1 for all in-service (i,t)
      const double post_var  = 1.0 / (prior_prec + static_cast<double>(n_it));
      const double post_mean = post_var * (prior_prec * pmu + sum_r);
      Ep(i,t) = post_mean;
      Vp(i,t) = post_var;
    }
  }
  
  
  
  
  // Enforce sum_i p_it = 0 within each time period (means only; Method A)
  for (unsigned int t = 0; t < T; ++t) {
    double sum_m = 0.0;
    unsigned int n_serv = 0;
    
    // count in-service legislators and accumulate means
    for (unsigned int i = 0; i < N; ++i) {
      if (t >= (unsigned) startlegis(i,0) && t <= (unsigned) endlegis(i,0)) {
        sum_m += Ep(i,t);
        ++n_serv;
      }
    }
    
    // REQUIRE at least one legislator; otherwise fail loudly
    if (n_serv == 0) {
      Rcpp::stop("getP_dynIRT: no serving legislators in period t=%u; "
                   "cannot enforce sum-to-zero centering.", t);
    }
    
    const double mean_t = sum_m / (double) n_serv;
    
    
    
    // subtract within-period mean; keep out-of-service cells at 0
    for (unsigned int i = 0; i < N; ++i) {
      if (t >= (unsigned) startlegis(i,0) && t <= (unsigned) endlegis(i,0)) {
        Ep(i,t) -= mean_t;
      } else {
        Ep(i,t) = 0.0;
        Vp(i,t) = 0.0;
      }
    }
  }
  
  return;
}