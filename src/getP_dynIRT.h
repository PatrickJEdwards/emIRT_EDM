// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#ifndef GETP_DYNIRT_H
#define GETP_DYNIRT_H

#include <RcppArmadillo.h>

// Updates propensity means Ep (N x T) and variances Vp (N x T) in place.
// Uses residuals r_ijt = E[y*_ijt] - E[alpha_jt] - E[beta_jt] * E[x_it],
// a N(pmu, psigma) prior, and enforces sum_i p_it = 0 within each time t.
void getP_dynIRT(arma::mat &Ep,                 // N x T  (means)   [updated]
                 arma::mat &Vp,                 // N x T  (variances) [updated]
                 const arma::mat &Eystar,       // N x J  (E[y*])
                 const arma::mat &alpha,        // J x 1  (alpha)
                 const arma::mat &beta,         // J x 1  (beta)
                 const arma::mat &x,            // N x T  (x)
                 const arma::mat &bill_session, // J x 1  (0..T-1)
                 const arma::mat &startlegis,   // N x 1
                 const arma::mat &endlegis,     // N x 1
                 const double pmu,              // prior mean
                 const double psigma,           // prior variance
                 const unsigned int T,
                 const unsigned int N,
                 const unsigned int J);

#endif