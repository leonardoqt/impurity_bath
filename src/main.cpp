#include <iostream>
#include <armadillo>
#include <mpi.h>
#include "model.h"

using namespace std;
using namespace arma;

int main()
{
	MPI_Init(NULL,NULL);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	//
	model m1;
	double e0,e1,e2,v01,v02,v12;
	if (rank == 0)
		cin>>e0>>e1>>e2>>v01>>v02>>v12;
	MPI_Bcast(&e0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&e1,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&e2,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&v01,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&v02,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&v12,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	mat H={{e0,v01,v02},{v01,e1,v12},{v02,v12,e2}};
	vec V={0.0035,0,0};
	m1.init(0,0.1);
	m1.load_system(H,V);
	/*
	double amin = 0, amax = 2 * datum::pi, bmin = amin, bmax = amax, cmin = amin, cmax = amax;
	int npoint = 50;
	double dt = amax / (npoint-1);
	vec an,bn,cn;
	double min_err=1e15, min_a, min_b, min_c;
	double* send=new double[4];
	double* recv=new double[4*size];
	for(int t1=0; t1<3; t1++)
	{
		an = linspace(amin,amax,npoint);
		bn = linspace(bmin,bmax,npoint);
		cn = linspace(cmin,cmax,npoint);
		for(int t2=0; t2<npoint; t2++)
		for(int t3=0; t3<npoint; t3++)
		for(int t4=0; t4<npoint; t4++)
		{
			int ind = t2*npoint*npoint+t3*npoint+t4;
			if(ind%size == rank)
			{
				double cc = cos(an(t2)), ss = sin(an(t2));
				mat r1 = {{cc,ss,0},{-ss,cc,0},{0,0,1}};
				cc = cos(bn(t3)); ss = sin(bn(t3));
				mat r2 = {{1,0,0},{0,cc,ss},{0,-ss,cc}};
				cc = cos(cn(t4)); ss = sin(cn(t4));
				mat r3 = {{cc,0,-ss},{0,1,0},{ss,0,cc}};
				m1.assign_u(r3*r2*r1);
				m1.run_exact();
				m1.run_redfield();
				if(m1.get_err() < min_err)
				{
					min_err = m1.get_err();
					min_a = an(t2);
					min_b = bn(t3);
					min_c = cn(t4);
				}
			}
		}
		send[0] = min_err; send[1] = min_a; send[2] = min_b; send[3] = min_c;
		MPI_Gather(send,4,MPI_DOUBLE,recv,4,MPI_DOUBLE,0,MPI_COMM_WORLD);
		if(rank == 0)
		{
			mat tmp_res(4*size,1,fill::zeros);
			for(int t2=0; t2<4*size; t2++)
				tmp_res(t2,0) = recv[t2];
			tmp_res = reshape(tmp_res,4,size);
			int ind = index_min(tmp_res.row(0).t());
			amin = tmp_res(1,ind) - dt; amax = tmp_res(1,ind) + dt;
			bmin = tmp_res(2,ind) - dt; bmax = tmp_res(2,ind) + dt;
			cmin = tmp_res(3,ind) - dt; cmax = tmp_res(3,ind) + dt;
		}
		MPI_Bcast(&amin,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&amax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&bmin,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&bmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&cmin,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&cmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		dt = 2 * dt / (npoint-1);
	}
	//
	if (rank == 0)
	{
		double cc = cos((amin+amax)/2), ss = sin((amin+amax)/2);
		mat r1 = {{cc,ss,0},{-ss,cc,0},{0,0,1}};
		cc = cos((bmin+bmax)/2); ss = sin((bmin+bmax)/2);
		mat r2 = {{1,0,0},{0,cc,ss},{0,-ss,cc}};
		cc = cos((cmin+cmax)/2); ss = sin((cmin+cmax)/2);
		mat r3 = {{cc,0,-ss},{0,1,0},{ss,0,cc}};
		m1.assign_u(r3*r2*r1);
		m1.run_exact();
		m1.run_redfield();
		//
		cout<<"Hamiltonian:"<<endl;
		m1.Hs.print();
		cout<<"Gamma:"<<endl;
		((m1.Vsb%m1.Vsb)*m1.v2gamma).print();
		cout<<"Best basis:"<<endl;
		m1.uo.print();
		cout<<"New Hamiltonian:"<<endl;
		(m1.uo.t()*m1.Hs*m1.uo).print();
		cout<<"New Gamma:"<<endl;
		(((m1.uo.t()*m1.Vsb)%(m1.uo.t()*m1.Vsb))*m1.v2gamma).print();
		cout<<"Best error:"<<endl;
		cout<<m1.get_err()<<endl;
	}
	*/
	mat aHs(3,12,fill::zeros), aG(3,4,fill::zeros), aU(3,12,fill::zeros), aHn(3,12,fill::zeros),aGn(3,4,fill::zeros);
	vec aerr(4,fill::zeros);
	mat UU;
	//1
	UU = m1.gen_u1();
	m1.assign_u(UU);
	m1.run_exact();
	m1.run_redfield();
	aG.col(0) = (m1.Vsb%m1.Vsb)*m1.v2gamma;
	aGn.col(0) = ((m1.uo.t()*m1.Vsb)%(m1.uo.t()*m1.Vsb))*m1.v2gamma;
	aU.cols(0,2) = m1.uo;
	aHs.cols(0,2) = m1.Hs;
	aHn.cols(0,2) = m1.uo.t() * m1.Hs * m1.uo;
	aerr(0) = m1.get_err();
	//2
	UU = m1.gen_u2();
	m1.assign_u(UU);
	m1.run_exact();
	m1.run_redfield();
	aG.col(1) = (m1.Vsb%m1.Vsb)*m1.v2gamma;
	aGn.col(1) = ((m1.uo.t()*m1.Vsb)%(m1.uo.t()*m1.Vsb))*m1.v2gamma;
	aU.cols(3,5) = m1.uo;
	aHs.cols(3,5) = m1.Hs;
	aHn.cols(3,5) = m1.uo.t() * m1.Hs * m1.uo;
	aerr(1) = m1.get_err();
	//3
	UU = m1.gen_u3();
	m1.assign_u(UU);
	m1.run_exact();
	m1.run_redfield();
	aG.col(2) = (m1.Vsb%m1.Vsb)*m1.v2gamma;
	aGn.col(2) = ((m1.uo.t()*m1.Vsb)%(m1.uo.t()*m1.Vsb))*m1.v2gamma;
	aU.cols(6,8) = m1.uo;
	aHs.cols(6,8) = m1.Hs;
	aHn.cols(6,8) = m1.uo.t() * m1.Hs * m1.uo;
	aerr(2) = m1.get_err();
	//4
	UU = m1.gen_u4();
	m1.assign_u(UU);
	m1.run_exact();
	m1.run_redfield();
	aG.col(3) = (m1.Vsb%m1.Vsb)*m1.v2gamma;
	aGn.col(3) = ((m1.uo.t()*m1.Vsb)%(m1.uo.t()*m1.Vsb))*m1.v2gamma;
	aU.cols(9,11) = m1.uo;
	aHs.cols(9,11) = m1.Hs;
	aHn.cols(9,11) = m1.uo.t() * m1.Hs * m1.uo;
	aerr(3) = m1.get_err();
	//
	cout<<"Hamiltonian:"<<endl;
	aHs.print();
	cout<<"Gamma:"<<endl;
	aG.print();
	cout<<"Best basis:"<<endl;
	aU.print();
	cout<<"New Hamiltonian:"<<endl;
	aHn.print();
	cout<<"New Gamma:"<<endl;
	aGn.print();
	cout<<"Best error:"<<endl;
	aerr.t().print();
	MPI_Finalize();
	return 0;
}
