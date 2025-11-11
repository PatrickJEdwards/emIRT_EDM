// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#ifndef GETYESTAR_DYNIRT_H
#define GETYESTAR_DYNIRT_H

#include <RcppArmadillo.h>

void getEystar_dynIRT(arma::mat &Eystar,           // N x J (filled)
					          const arma::mat &alpha,        // J x 1  (alpha)
                    const arma::mat &beta,         // J x 1  (beta)
                    const arma::mat &x,            // N x T  (x)
                    const arma::mat &p,            // N x T  (p)   <-- NEW
                    const arma::mat &y,            // N x J  (observed {-1,0,1})
                    const arma::mat &bill_session, // J x 1  (0..T-1)
                    const arma::mat &startlegis,   // N x 1
                    const arma::mat &endlegis,     // N x 1
                    const int N, 
                    const int J
                    );

#endif
