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
	mat H={{-0.1,0.0047,-0.0138},{0.0047,-0.1183,0},{-0.0138,0,-0.0927}};
	vec V={0.0035,0,0};
	m1.init(0,0.1);
	m1.load_system(H,V);
	double amin = 0, amax = datum::pi, bmin = amin, bmax = amax, cmin = amin, cmax = amax;
	double dt = datum::pi / 4;
	vec an,bn,cn;
	double min_err=1e15, min_a, min_b, min_c;
	double* send=new double[4];
	double* recv=new double[4*size];
	for(int t1=0; t1<10; t1++)
	{
		an = linspace(amin,amax,5);
		bn = linspace(bmin,bmax,5);
		cn = linspace(cmin,cmax,5);
		for(int t2=0; t2<5; t2++)
		for(int t3=0; t3<5; t3++)
		for(int t4=0; t4<5; t4++)
		{
			int ind = t2*25+t3*5+t4;
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
		dt /= 2;
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
	MPI_Finalize();
	return 0;
}
