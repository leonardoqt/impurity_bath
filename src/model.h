#ifndef __IMP_MODEL__
#define __IMP_MODEL__

#include<armadillo>

class model;

class model
{
private:
	const int sz = 3;
	const int nb = 1200;
	const int Tmax = nb;
	const int nT = 100;
	const double dep=3.0;
	int has_run_exact=0;
	void run_exact_first();
	arma::cx_cube dt_da;
	arma::mat N0_a;
public:
	//
	double mu, T, beta, v2gamma;
	arma::mat Hs;
	arma::vec Vsb;
	arma::vec Eb;
	arma::mat uo;
	//
	arma::vec T_grid;
	arma::mat ndt;
	arma::mat npt;
	arma::mat N0;
	arma::mat rho0;
	//
	void init(double Mu, double Temp);
	void load_system(arma::mat Hs, arma::vec Vsb);
	void assign_u(arma::mat U);
	void run_exact();
	void run_redfield();
	double get_err();
	// 
	arma::mat gen_u1(); // full adiabatic
	arma::mat gen_u2(); // one talk, then adiabatic
	arma::mat gen_u3(); // one talk, then max the other talk
	arma::mat gen_u4(); // diag the above to get two talk
	//
	arma::mat gen_u_ang(double); // a 2x2 rotation
};

#endif
