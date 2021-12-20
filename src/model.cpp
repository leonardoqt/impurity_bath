#include "model.h"

using namespace arma;

void model::init(double Mu, double Temp)
{
	mu = Mu;
	T = Temp;
	beta = 1/T;
	v2gamma = datum::pi*nb/dep;
	//
	Eb = linspace(-dep,dep,nb);
	uo = eye(sz,sz);
	//
	T_grid = linspace(0,Tmax,nT);
	ndt = mat(nT,sz);
	npt = mat(nT,sz);
	//
	vec N_diag = 1/(1+exp(beta*(Eb-mu)));
	N_diag = join_vert(vec(sz,fill::zeros),N_diag);
	N0 = diagmat(N_diag);
	//
	rho0 = mat(8,8,fill::zeros);
	rho0(0,0) = 1;
}

void model::load_system(mat HS, vec VSB)
{
	Hs = HS;
	Vsb = VSB;
	has_run_exact = 0;
}

void model::assign_u(mat U)
{
	uo = U;
}

void model::run_exact_first()
{
	mat H_tot = diagmat(join_vert(Hs.diag(),Eb));
	H_tot(span(0,sz-1),span(0,sz-1)) = Hs;
	for(int t1=sz; t1<sz+nb; t1++)
	{
		H_tot(span(0,sz-1),t1) = Vsb;
		H_tot(t1,span(0,sz-1)) = Vsb.t();
	}
	//
	H_tot = (H_tot + H_tot.t())/2;
	mat U_a;
	vec E_a;
	eig_sym(E_a,U_a,H_tot);
	//
	mat d0_da;
	d0_da = U_a.t()*eye(sz+nb,sz);
	N0_a = U_a.t()*N0*U_a;
	//
	dt_da = cx_cube(sz+nb,sz,nT);
	cx_mat dt_oa(sz+nb,sz);
	cx_vec phase;
	cx_double ii(0,1);
	for (int t1=0; t1<nT; t1++)
	{
		phase = exp(ii*E_a*T_grid(t1));
		for(int t2=0; t2<sz; t2++)
			dt_da.slice(t1).col(t2) = phase % d0_da.col(t2);
		dt_oa = dt_da.slice(t1) * uo;
		ndt.row(t1) = diagvec( real(dt_oa.t() * N0_a * dt_oa) ).t();
	}
}

void model::run_exact()
{
	if(! has_run_exact)
	{
		has_run_exact = 1;
		run_exact_first();
	}
	else
	{
		cx_mat dt_oa;
		for (int t1=0; t1<nT; t1++)
		{
			dt_oa = dt_da.slice(t1) * uo;
			ndt.row(t1) = diagvec( real(dt_oa.t() * N0_a * dt_oa) ).t();
		}
	}
}

void model::run_redfield()
{
	mat Hs_o = uo.t() * Hs * uo;
	vec Vsb_o = uo.t() * Vsb;
	mat HH(8,8,fill::zeros);
	HH(span(1,3),span(1,3)) = Hs_o;
	HH(span(4,6),span(4,6)) = eye(3,3)*trace(Hs_o)-Hs_o;
	HH(7,7) = trace(Hs_o);
	//
	vec gamma = ( Vsb_o % Vsb_o )*v2gamma;
	vec ff = 1 /( 1 + exp( beta*(diagvec(Hs_o)-mu) ) );
	//
	cube L0(8,8,12,fill::zeros);
	//
	L0(0,1, 0) = L0(4,7, 1) = L0(3,5, 2) = L0(2,6, 3) = 1;
	L0(0,2, 4) = L0(5,7, 5) = L0(3,4, 6) = L0(1,6, 7) = 1;
	L0(0,3, 8) = L0(6,7, 9) = L0(2,4,10) = L0(1,5,11) = 1;
	//
	mat II = eye(8,8);
	mat LL(64,64,fill::zeros);
	mat Lt;
	for (int t1=0; t1<3; t1++)
		for (int t2=0; t2<4; t2++)
		{
			Lt = L0.slice(t1*4+t2);
			LL += gamma(t1)*( 1-ff(t1) )*( kron(Lt,Lt)-0.5*(kron(II,Lt.t()*Lt)+kron(Lt.t()*Lt,II)) );
			Lt = Lt.t();
			LL += gamma(t1)*(   ff(t1) )*( kron(Lt,Lt)-0.5*(kron(II,Lt.t()*Lt)+kron(Lt.t()*Lt,II)) );
		}
	cx_double ii(0,1);
	cx_mat HL = -ii*(kron(II,HH)-kron(HH,II)) + LL;
	//
	cx_mat rhot;
	vec ot;
	for(int t1=0; t1<nT; t1++)
	{
		rhot = reshape(expmat(HL*T_grid(t1))*reshape(rho0,64,1),8,8);
		ot = diagvec(real(rhot));
		npt(t1,0) = ot(1) + ot(5) + ot(6) + ot(7);
		npt(t1,1) = ot(2) + ot(4) + ot(6) + ot(7);
		npt(t1,2) = ot(3) + ot(4) + ot(5) + ot(7);
	}
}

double model::get_err()
{
	//return sqrt( trace( (npt-ndt).t()*(npt-ndt) ) / sz / nT );
	return max( max( abs(npt-ndt) ) );
}

mat model::gen_u1()
{
	mat U;
	vec val;
	eig_sym(val,U,Hs);
	return U;
}

mat model::gen_u2()
{
	mat U,V,tmpx;
	vec val;
	//
	tmpx = mat(3,3,fill::zeros);
	tmpx.col(0) = Vsb;
	svd(U,val,V,tmpx);
	//
	mat H2 = U.t() * Hs * U;
	mat H1212 = H2(span(1,2),span(1,2));
	mat u2;
	eig_sym(val,u2,H1212);
	mat U2 = eye(3,3);
	U2(span(1,2),span(1,2)) = u2;
	U = U * U2;
	return U;
}

mat model::gen_u3()
{
	mat U,V,tmpx;
	vec val;
	//
	tmpx = mat(3,3,fill::zeros);
	tmpx.col(0) = Vsb;
	svd(U,val,V,tmpx);
	//
	mat H2 = U.t() * Hs * U;
	tmpx = mat(2,2,fill::zeros);
	tmpx.col(0) = H2(span(1,2),span(0,0));
	mat u2;
	svd(u2,val,V,tmpx);
	mat U2 = eye(3,3);
	U2(span(1,2),span(1,2)) = u2;
	U = U * U2;
	return U;
}

mat model::gen_u4()
{
	mat U,V,tmpx;
	vec val;
	//
	tmpx = mat(3,3,fill::zeros);
	tmpx.col(0) = Vsb;
	svd(U,val,V,tmpx);
	//
	mat H2 = U.t() * Hs * U;
	tmpx = mat(2,2,fill::zeros);
	tmpx.col(0) = H2(span(1,2),span(0,0));
	mat u2;
	svd(u2,val,V,tmpx);
	mat U2 = eye(3,3);
	U2(span(1,2),span(1,2)) = u2;
	U = U * U2;
	H2 = U.t() * Hs * U;
	tmpx = H2(span(0,1),span(0,1));
	eig_sym(val,u2,tmpx);
	U2 = eye(3,3);
	U2(span(0,1),span(0,1)) = u2;
	U = U * U2;
	return U;
}

mat model::gen_u_ang(double theta)
{
	mat U = eye(3,3);
	U(1,1) = cos(theta);
	U(1,2) = sin(theta);
	U(2,1) =-sin(theta);
	U(2,2) = cos(theta);
	return U;
}
