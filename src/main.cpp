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
	int nnn = 36;
	vec de = linspace(0,0.012,nnn);
	vec dv = linspace(0,0.012,nnn);
	mat H;
	vec V={0.0035,0,0};
	m1.init(0,0.1);
	//
	double *err_send, *err_recv, *ang_send, *ang_recv;
	err_send = new double[nnn*nnn];
	err_recv = new double[nnn*nnn];
	ang_send = new double[nnn*nnn];
	ang_recv = new double[nnn*nnn];
	for (int t1 = 0; t1<nnn*nnn; t1++)
		err_send[t1] = err_recv[t1] = ang_send[t1] = ang_recv[t1] = 0.0;
	
	int ind;
	for(int t1=0; t1<nnn; t1++)
		for(int t2=0; t2<nnn; t2++)
		{
			ind = t1*nnn + t2;
			if (ind%size == rank)
			{
				H = {{-0.1,dv(t2),0.012-dv(t2)},{dv(t2),-0.1,0.0},{0.012-dv(t2),0.0,-0.1-de(t1)}};
				m1.load_system(H,V);
				vec theta = linspace(0,datum::pi/2,90);
				err_send[ind] = 1e6;
				double err_tmp;
				for(auto ang:theta)
				{
					m1.assign_u(m1.gen_u_ang(ang));
					m1.run_exact();
					m1.run_redfield();
					err_tmp = m1.get_err();
					if (err_tmp < err_send[ind])
					{
						err_send[ind] = err_tmp;
						ang_send[ind] = ang;
					}
				}
			}
		}
	//
	MPI_Allreduce(err_send,err_recv,nnn*nnn,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(ang_send,ang_recv,nnn*nnn,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	//
	if (rank == 0)
	{
		mat err_out(nnn*nnn,1,fill::zeros), ang_out(nnn*nnn,1,fill::zeros);
		for(int t1=0; t1<nnn*nnn; t1++)
		{
			err_out(t1,0) = err_recv[t1];
			ang_out(t1,0) = ang_recv[t1];
		}
		err_out = reshape(err_out,nnn,nnn);
		ang_out = reshape(ang_out,nnn,nnn);
		cout<<"error matrix:"<<endl;
		err_out.print();
		cout<<"angle matrix:"<<endl;
		ang_out.print();
	}
	MPI_Finalize();
	return 0;
}
