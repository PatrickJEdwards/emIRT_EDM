// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>
//#include <RcppTN.h>
#include "etn1.h"

using namespace Rcpp;

// // [[Rcpp::export()]]
void getEystar_dynIRT(arma::mat &Eystar,
					          const arma::mat &alpha,        // J x 1
                    const arma::mat &beta,         // J x 1
                    const arma::mat &x,            // N x T
                    const arma::mat &p,            // N x T  (p)   <-- NEW
                    const arma::mat &y,            // N x J  (1 / -1 / 0)
                    const arma::mat &bill_session, // J x 1 (0..T-1)
                    const arma::mat &startlegis,   // N x 1
                    const arma::mat &endlegis,     // N x 1
                    const int N,
                    const int J
                    ) {

	double q1;
	signed int i, j;

    // Main Calculation
#pragma omp parallel for private(i,j,q1)
  	for(i=0; i < N; i++){

		for(j=0; j < J; j++){

			if( (bill_session(j,0) <= endlegis(i,0)) && (bill_session(j,0) >= startlegis(i,0)) ){

			    // ONLY CHANGE: add p(i, bill_session(j,0)) to the latent mean
		    	q1 = p(i, bill_session(j,0)) + alpha(j,0) + x(i,bill_session(j,0)) * beta(j,0);

//			    if(y(i,j)==1)     Eystar(i,j) = RcppTN::etn1(q1, 1.0, 0.0, R_PosInf);
//			    if(y(i,j)==-1)    Eystar(i,j) = RcppTN::etn1(q1, 1.0, R_NegInf, 0.0);
//			    if(y(i,j)==0)     Eystar(i,j) = RcppTN::etn1(q1, 1.0, R_NegInf, R_PosInf);

			    if(y(i,j)==1)     Eystar(i,j) = etn1(q1, 1.0, 0.0, R_PosInf);
			    if(y(i,j)==-1)    Eystar(i,j) = etn1(q1, 1.0, R_NegInf, 0.0);
			    if(y(i,j)==0)     Eystar(i,j) = etn1(q1, 1.0, R_NegInf, R_PosInf);

				// Note: Taking etn() of extreme truncated normals is a problem			    
			    // etn(-9.49378,1,0,1000) gives Inf for Eystar, which crashes everything
			    // In these cases, we should ignore the vote
			    if( !(std::isfinite(Eystar(i,j))) ) Eystar(i,j) = q1;
		    
			} // end if( (bill_session(j,0) <= endlegis(i,0)) 

		}  // end for(j=0; j < J; j++)
  	}  // end for(i=0; i < N; i++)

    return; 
} 
