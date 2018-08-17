/*3D softrobtic phase expansion code*/
/*by Yao*/
/*5.2017*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <ctime>
//#include <omp.h>
#include <float.h>
#include <limits.h>

using namespace std;

#define M 10
#define N 10
#define L 10

#define ndim 3

#define nparticle M*N*L
#define nneighbors 18
#define boxlength 0.01


int nparticle1;
//double cy_trans[nparticle]; //the transformation cycle from ini to propa

double Radius = boxlength/(2.0*M);//assume M=N=L
double rho = 4.43E2;//in 2d is 4.43E3, in order to keep the same mass so that acceleration won't be too large
double sigY = 2E5;
double E0 = 1E6;
double mu0 = 0.3;
double E1 = 2E9;
double mu1 = 0.3;
double E2 = 1E6;
double mu2 = 0.3;
double Mass, Kn01, Kn02, Tv0, Kn11, Kn12, Tv1;//, damping;
double Kn21, Kn22, Tv2;
double MaxF1;
double t_end, t_start, t_step;
int steps, damage;

double Positions_t[nparticle][ndim]={};
double Positions[nparticle][ndim]={};
double OrinPositions[nparticle][ndim]={};//for reset the configuration or displacement calculation
double bondsign[nparticle][nneighbors];//denote the state of the bond (failure or not)
double nedge[2*ndim]={};//the number of particle for each surface

int neighbors[nparticle][nneighbors]={};//the neighbor index for each particle
double origindistance[nparticle][nneighbors]={};//original distance for calculation of dL
double nsign[nparticle][nneighbors]={};//the nearest neighbor level(e.g., first nearest neighbor)
double Stretch[nparticle][nneighbors]={};
double Kn[nparticle][nneighbors]={};
double Tv[nparticle][nneighbors]={};
int NB[nparticle]={};//the real number of neighbor of each particle

double Lvel[nparticle][ndim]={};//velocity of time t
double Lvel_t[nparticle][ndim]={};//velocity of time t/2

double dL[nparticle][nneighbors]={};//delta distance after stretch
double dL_total[nparticle][2]={};// the same as TdL
double TdL_total[nparticle][2]={};//0 is the first-nearest neighbor and the 1 is the second-nearest neighbor
double distance1[nparticle][nneighbors]={};//the current distance
double Lacc[nparticle][ndim]={};

double netF[nparticle][ndim]={};//the surface force


//int crack_time;
//int damage_total_number=0;


int n1Tn2[nparticle] = {};
int n2Tn1[nparticle] = {};

int Top[M*L]={};
int Bottom[M*L]={};
int Left[N*L]={};
int Right[N*L]={};
int Front[M*N]={};
int Back[M*N]={};

int Phase01[nparticle]={};

double Ur[M][2];


bool IsNan (double val)
{
if (val==val)
return false;
else
return true;
}


bool Iszero (int nn)
{
if (nn==0)
return true;
else
return false;
}



void Initialization()
{
double x, y, z;
int i,j,k,Le,Ri,To,Bo,Ba,Fr,n1,n2;
//int i,j,q,l,k,m,n,n1,n2;

//ofstream fout("edge_check.dat");
ifstream fin("w3_attached_100x100x100.vtk");


	for(i=0;i<nparticle;i++)
	{
		fin>>Phase01[i];
		Phase01[i]=0;

	}
	fin.close();


	n1=0;

	for(k=0;k<L;k++)
	{
        z = Radius + k*2*Radius;
	    for(j=0;j<M;j++)
	    {
	    	y = Radius + j*2*Radius;
	    	for(i=0; i<N; i++)
	    	{
	    		x = Radius + i*2*Radius;
	    		if(n1<nparticle)
	    		{
	    		Positions_t[n1][0]=x;
	    		Positions_t[n1][1]=y;
	    		Positions_t[n1][2]=z;
	    		//fout<<n1<<"  "<<Positions_t[n1][0]<<"  "<<Positions_t[n1][1]<<"  "<<Positions_t[n1][2]<<endl;
	    		}
	    		n1=n1+1;
	    	}
	    }
	}
//cin.get();
	//cout<<"n1:"<<n1<<endl;

	n2=0;

	Le=-1;
	Ri=-1;
	To=-1;
	Bo=-1;
	Ba=-1;
	Fr=-1;

	//int exist = 0;
	//int noexist = 0;
	for(j=0;j<n1;j++)
	{
		if(Phase01[j]!=55)
		{
			//exist++;
			Positions[n2][0]=Positions_t[j][0];
			Positions[n2][1]=Positions_t[j][1];
			Positions[n2][2]=Positions_t[j][2];

			OrinPositions[n2][0]=Positions_t[j][0];
			OrinPositions[n2][1]=Positions_t[j][1];
			OrinPositions[n2][2]=Positions_t[j][2];

			if(Positions[n2][0]<Radius+2*Radius)
			{
			//Phase01[j]=88;
			Le++; Left[Le]=n2;
			//fout<<j<<"\t"<<n2<<"\t"<<Le<<"\t"<<Left[Le]<<endl;
			}


			if(Positions[n2][0]>(0.01-2*Radius))
			{
			//Phase01[j]=99;
			Ri++; Right[Ri]=n2;
			}

			if(Positions[n2][1]>(0.01-2*Radius))
			{
			//Phase01[j]=99;
			To++; Top[To]=n2;
			}

			if(Positions[n2][1]<(2*Radius))
			{
			//Phase01[j]=99;
			Bo++; Bottom[Bo]=n2;
			}

			if(Positions[n2][2]>(0.01*(L+0.0)/(M+0.0)-2*Radius))
			{
			//Phase01[j]=99;
			Ba++; Back[Ba]=n2;
			}

			if(Positions[n2][2]<(2*Radius))
			{
			//Phase01[j]=99;
			Fr++; Front[Fr]=n2;
			}



			n1Tn2[j] = n2;
			n2Tn1[n2] = j;




			n2 = n2+1;
		}

	}


	nparticle1=n2;



	nedge[0] = Le+1;
	nedge[1] = Ri+1;
	nedge[2] = To+1;
	nedge[3] = Bo+1;
	nedge[4] = Ba+1;
	nedge[5] = Fr+1;
	cout<<nedge[0]<<"\t"<<nedge[1]<<"\t"<<nedge[2]<<"\t"<<nedge[3]<<"\t"<<nedge[4]<<"\t"<<nedge[5]<<endl;
	//cin.get();

/*
for(i=0;i<n1;i++)
{
	int temp = 0;
	if(Phase01[i]!=55)
	{temp =1;
		for(j=0;j<nedge[5];j++)
		{
			if(n2Tn1[Front[j]]==i)
			{temp=2;break;}
		}
	fout<<temp<<endl;
	}
	else fout<<temp<<endl;
}
*/

	for(i=0;i<nparticle1;i++)
	for(j=0;j<nneighbors;j++)
	bondsign[i][j]=0;

cout<<"***********************initialization compeleted***************************"<<endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////


void Searchneighbor()
{
double dis;
int i,j,index;

	//#pragma omp parallel for private(index,dis,j)
	for(i=0;i<nparticle1;i++)
	{
		//cout<<"neighbor:"<<i<<endl;
		index = 0;
		for(j=0;j<nparticle1;j++)
		{//cout<<i<<"\t"<<j<<endl;cin.get();
			dis = sqrt((Positions[j][0]-Positions[i][0])*(Positions[j][0]-Positions[i][0])+(Positions[j][1]-Positions[i][1])*(Positions[j][1]-Positions[i][1])+(Positions[j][2]-Positions[i][2])*(Positions[j][2]-Positions[i][2]));
			if((j!=i)&&(dis<2.01*Radius))
			{
				neighbors[i][index] = j;
				nsign[i][index]=1;
				origindistance[i][index]=dis;

				if((Phase01[n2Tn1[i]] == 0) && (Phase01[n2Tn1[j]] ==0))
				{
				Kn[i][index] = Kn01;
				Tv[i][index] = Tv0;
				Stretch[i][index] = (120*sigY/(2.0*Kn01) + 1.0)*dis;
				}
				else if((Phase01[n2Tn1[i]] == 1) && (Phase01[n2Tn1[j]] ==1))
				{
				Kn[i][index] = Kn11;
				Tv[i][index] = Tv1;
				Stretch[i][index] = (120*sigY/(2.0*Kn11) + 1.0)*dis;
				//cout<<"stop"<<endl;
				}

				else if((Phase01[n2Tn1[i]] == 2) && (Phase01[n2Tn1[j]] ==2))
				{
				Kn[i][index] = Kn21;
				Tv[i][index] = Tv2;
				Stretch[i][index] = (120*sigY/(2.0*Kn21) + 1.0)*dis;
				}

				else if(((Phase01[n2Tn1[i]] == 0) && (Phase01[n2Tn1[j]] ==1))||((Phase01[n2Tn1[i]] == 1) && (Phase01[n2Tn1[j]] ==0)))
				{
				Kn[i][index] = 2*(Kn01*Kn11/(Kn01+Kn11));
				if(Tv0 ==0&&Tv1==0)
				Tv[i][index] = 0;
				else
				Tv[i][index] = 2*(Tv0*Tv1/(Tv0+Tv1));

				Stretch[i][index] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11))) + 1.0)*dis;
				}

				else if(((Phase01[n2Tn1[i]] == 0) && (Phase01[n2Tn1[j]] ==2))||((Phase01[n2Tn1[i]] == 2) && (Phase01[n2Tn1[j]] ==0)))
				{
				//Kn[i][index] = 2*(Kn01*Kn21/(Kn01+Kn21));
				Kn[i][index] = 1e6;
				if(Tv0 ==0&&Tv2==0)
				Tv[i][index] = 0;
				else
				//Tv[i][index] = 2*(Tv0*Tv2/(Tv0+Tv2));
				Tv[i][index] = 1e6;

				//Stretch[i][index] = (1*sigY/(4.0*(Kn01*Kn21/(Kn01+Kn21))) + 1.0)*dis;
				Stretch[i][index] = 0.000102594;
				}

				else if(((Phase01[n2Tn1[i]] == 2) && (Phase01[n2Tn1[j]] ==1))||((Phase01[n2Tn1[i]] == 1) && (Phase01[n2Tn1[j]] ==2)))
				{
				//Kn[i][index] = 2*(Kn21*Kn11/(Kn21+Kn11));
				Kn[i][index] = 1e6;
				if(Tv2 ==0&&Tv1==0)
				Tv[i][index] = 0;
				else
				//Tv[i][index] = 2*(Tv2*Tv1/(Tv2+Tv1));
				Tv[i][index] = 1e6;

				//Stretch[i][index] = (1*sigY/(4.0*(Kn21*Kn11/(Kn21+Kn11))) + 1.0)*dis;
				Stretch[i][index] = 0.000102594;
				}

			index++;
			}
			else if((dis<2.01*sqrt(2.0)*Radius)&&(dis>2.01*Radius))
			{
				neighbors[i][index]=j;
				nsign[i][index]=2;
				origindistance[i][index]=dis;

				if((Phase01[n2Tn1[i]] == 0) && (Phase01[n2Tn1[j]] ==0))
				{
				Kn[i][index] = Kn02;
				Tv[i][index] = Tv0;
				Stretch[i][index] = (120*sigY/(2.0*Kn01)/sqrt(2.0) + 1.0)*dis;
				}
				else if((Phase01[n2Tn1[i]] == 1) && (Phase01[n2Tn1[j]] ==1))
				{
				Kn[i][index] = Kn12;
				Tv[i][index] = Tv1;
				Stretch[i][index] = (120*sigY/(2.0*Kn11)/sqrt(2.0) + 1.0)*dis;
				}

				else if((Phase01[n2Tn1[i]] == 2) && (Phase01[n2Tn1[j]] ==2))
				{
				Kn[i][index] = Kn22;
				Tv[i][index] = Tv2;
				Stretch[i][index] = (120*sigY/(2.0*Kn21)/sqrt(2.0) + 1.0)*dis;
				}

				else if(((Phase01[n2Tn1[i]] == 0) && (Phase01[n2Tn1[j]] ==1))||((Phase01[n2Tn1[i]] == 1) && (Phase01[n2Tn1[j]] ==0)))
				{
				Kn[i][index] = 2.0*(Kn02*Kn12/(Kn02+Kn12));
				if(Tv0 ==0&&Tv1==0)
				Tv[i][index] = 0;
				else
				Tv[i][index] = 2.0*(Tv0*Tv1/(Tv0+Tv1));

				Stretch[i][index] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11)))/sqrt(2.0) + 1.0)*dis;
				}

				else if(((Phase01[n2Tn1[i]] == 0) && (Phase01[n2Tn1[j]] ==2))||((Phase01[n2Tn1[i]] == 2) && (Phase01[n2Tn1[j]] ==0)))
				{
				//Kn[i][index] = 2.0*(Kn02*Kn22/(Kn02+Kn22));
				Kn[i][index] = 1e6;
				if(Tv0 ==0&&Tv2==0)
				Tv[i][index] = 0;
				else
				//Tv[i][index] = 2.0*(Tv0*Tv2/(Tv0+Tv2));
				Tv[i][index] = 1e6;

				//Stretch[i][index] = (1*sigY/(4.0*(Kn01*Kn21/(Kn01+Kn21)))/sqrt(2.0) + 1.0)*dis;
				Stretch[i][index] = 0.00014509;
				}
				else if(((Phase01[n2Tn1[i]] == 2) && (Phase01[n2Tn1[j]] ==1))||((Phase01[n2Tn1[i]] == 1) && (Phase01[n2Tn1[j]] ==2)))
				{
				//Kn[i][index] = 2.0*(Kn22*Kn12/(Kn22+Kn12));
				Kn[i][index] = 1e6;
				if(Tv2 ==0&&Tv1==0)
				Tv[i][index] = 0;
				else
				//Tv[i][index] = 2.0*(Tv2*Tv1/(Tv2+Tv1));
				Tv[i][index] = 1e6;

				//Stretch[i][index] = (1*sigY/(4.0*(Kn21*Kn11/(Kn21+Kn11)))/sqrt(2.0) + 1.0)*dis;
				Stretch[i][index] = 0.00014509;
				}




			index++;
			}

			//cout<<i<<"\t"<<j<<"\t"<<index-1<<"\t"<<Phase01[n2Tn1[i]]<<"\t"<<Phase01[n2Tn1[j]]<<"\t"<<Kn[i][index-1]<<"\t"<<Tv[i][index-1]<<"yo"<<endl;
			//cin.get();
		}
	NB[i]=index;
	}

//	for(int i=0; i<nparticle; i++)
//	for(int j=0; j<NB[i];j++)
//	cout<<i<<"  "<<j<<"  "<<neighbors[i][j]<<endl;




/*
	for(i=0;i<nparticle1;i++)
	for(j=0;j<NB[i];j++)
	bondsign[i][j]=1;
*/

	ifstream fin("bondsign3498");
	for(i=0;i<nparticle1;i++)
	for(j=0;j<NB[i];j++)
	{
	fin>>bondsign[i][j];
	bondsign[i][j]=1;
	}


cout<<"***********************search neighbor completed***************************"<<endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////


void Update()
{
int i,j;

for(i=0;i<nparticle;i++)
for(j=0;j<nneighbors;j++)
dL[i][j]=0;


for(i=0;i<nparticle;i++)
for(j=0;j<ndim;j++)
dL_total[i][j]=0;

for(i=0;i<nparticle;i++)
for(j=0;j<ndim;j++)
TdL_total[i][j]=0;

for(i=0;i<nparticle;i++)
for(j=0;j<nneighbors;j++)
distance1[i][j]=0;


	//#pragma omp parallel for private(j)
	for(i=0;i<nparticle1;i++)
	{
		//#pragma omp parallel for
		for(j=0;j<NB[i];j++)
		{
			distance1[i][j]=sqrt((Positions[i][0]-Positions[neighbors[i][j]][0])*(Positions[i][0]-Positions[neighbors[i][j]][0])+(Positions[i][1]-Positions[neighbors[i][j]][1])*(Positions[i][1]-Positions[neighbors[i][j]][1])+(Positions[i][2]-Positions[neighbors[i][j]][2])*(Positions[i][2]-Positions[neighbors[i][j]][2]));
			//cout<<setprecision(15)<<distance1[i][j]<<"  "<<i<<"  "<<j<<endl;

			if(nsign[i][j]==1)
			{
				if(bondsign[i][j]==1)
				{
					if((Phase01[i]!=Phase01[neighbors[i][j]])&&(Phase01[i]==2||Phase01[neighbors[i][j]]==2))
					{
						//if(distance1[i][j]<=Stretch[i][j])
						dL[i][j] = distance1[i][j]-origindistance[i][j];
						//else if(distance1[i][j]>Stretch[i][j]&&distance1[i][j]<=2.0*Stretch[i][j])
						//dL[i][j] = Stretch[i][j]-origindistance[i][j];
						//else
						//bondsign[i][j]=0;
					}
					else
					{
						dL[i][j] = distance1[i][j]-origindistance[i][j];
					}

				}
				else if(bondsign[i][j] == 0 && distance1[i][j] < origindistance[i][j])
				{
				dL[i][j] = distance1[i][j]-origindistance[i][j];
				}
				dL_total[i][0] = dL_total[i][0] + dL[i][j];
				TdL_total[i][0] = TdL_total[i][0] + dL[i][j]*Tv[i][j];
			}
			else if(nsign[i][j] ==2)
			{
				if(bondsign[i][j]==1)
				{
					if((Phase01[i]!=Phase01[neighbors[i][j]])&&(Phase01[i]==2||Phase01[neighbors[i][j]]==2))
					{
						//if(distance1[i][j]<=Stretch[i][j])
						dL[i][j] = distance1[i][j]-origindistance[i][j];
						//else if(distance1[i][j]>Stretch[i][j]&&distance1[i][j]<=2.0*Stretch[i][j])
						//dL[i][j] = Stretch[i][j]-origindistance[i][j];
						//else
						//bondsign[i][j]=0;
					}
					else
					{
						dL[i][j] = distance1[i][j]-origindistance[i][j];
					}

				}
				else if(bondsign[i][j] == 0 && distance1[i][j] < origindistance[i][j])
				{
				dL[i][j] = distance1[i][j]-origindistance[i][j];
				}
				dL_total[i][1] = dL_total[i][1] + dL[i][j];
				TdL_total[i][1] = TdL_total[i][1] + dL[i][j]*Tv[i][j];
			}
		}
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////

void Netinteraction ()
{
	double dx,dy,dz,f;
	int i,j;

	for(i=0;i<nparticle;i++)
	for(j=0;j<ndim;j++)
	netF[i][j]=0;

	//#pragma omp parallel for private(dx,dy,dz,f,j)
	for(i=0;i<nparticle1;i++)
	{
		//#pragma omp parallel for
		for(j=0;j<NB[i];j++)
		{
			dx = Positions[neighbors[i][j]][0] - Positions[i][0];
			dy = Positions[neighbors[i][j]][1] - Positions[i][1];
			dz = Positions[neighbors[i][j]][2] - Positions[i][2];

			if(nsign[i][j]==1)
			{
				if(bondsign[i][j]==0&&dL[i][j]>=0)
				f=0;
				else
				f = 2.0*Kn[i][j]*dL[i][j] + 1.0/2.0*Tv[i][j]*(dL_total[i][0]+dL_total[neighbors[i][j]][0]) + 1.0/2.0*(TdL_total[i][0]+TdL_total[neighbors[i][j]][0]);
			}
			else if(nsign[i][j]==2)
			{
				if(bondsign[i][j]==0&&dL[i][j]>=0)
				f=0;
				else
				f = 2.0*Kn[i][j]*dL[i][j] + 1.0/2.0*Tv[i][j]*(dL_total[i][1]+dL_total[neighbors[i][j]][1]) + 1.0/2.0*(TdL_total[i][1]+TdL_total[neighbors[i][j]][1]);
			}

			netF[i][0] = netF[i][0] + dx*f/distance1[i][j];
			netF[i][1] = netF[i][1] + dy*f/distance1[i][j];
			netF[i][2] = netF[i][2] + dz*f/distance1[i][j];
		}
			//cout<<i<<"  "<<netF[i][0]<<endl;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////






int output_map(int time, int ct=0)
{
	//char filename1[255];
	//char filename2[255];
	//char filename3[255];
	//char filename4[255];
	//char filename5[255];
	//char filename6[255];
	//char filename7[255];
	//char filename8[255];
	char filename9[255];
	char filename10[255];
	char filename11[255];
	char filename12[255];
	char filename13[255];
	char filename14[255];
	char filename15[255];
	char filename16[255];
	char filename17[255];

	//sprintf(filename1,"disxy_time%d.dat",time);
	//sprintf(filename2,"disyz_time%d.dat",time);
	//sprintf(filename3,"diszx_time%d.dat",time);
	//sprintf(filename4,"Ur_time%d",time);
	//sprintf(filename5,"crack_structure_time%d_ct%d.vtk",time,ct);
	//sprintf(filename6,"crack_pure_time%d_ct%d.vtk",time,ct);
	//sprintf(filename7,"stretch_time%d_ct%d.vtk",time,ct);
	//sprintf(filename8,"interface_dispField_time%d_ct%d.vtk",time,ct);
	sprintf(filename9,"Position_%d.txt",time);
	sprintf(filename10,"dL_%d.txt",time);
	sprintf(filename11,"dL_Total_%d.txt",time);
	sprintf(filename12,"TdL_Total_%d.txt",time);
	sprintf(filename13,"bondsign_new_%d.txt",time);
	sprintf(filename14,"Phase01_%d.txt",time);

	sprintf(filename15,"Lvel_%d.txt",time);
	sprintf(filename16,"Lacc_%d.txt",time);
	sprintf(filename17,"netF_%d.txt",time);

	//ofstream fout1(filename1);
	//ofstream fout2(filename2);
	//ofstream fout3(filename3);
	//ofstream fout4(filename4);
	//ofstream fout5(filename5);
	//ofstream fout6(filename6);
	//ofstream fout7(filename7);
	//ofstream fout8(filename8);
	ofstream fout9(filename9);
	ofstream fout10(filename10);
	ofstream fout11(filename11);
	ofstream fout12(filename12);
	ofstream fout13(filename13);
	ofstream fout14(filename14);

	ofstream fout15(filename15);
	ofstream fout16(filename16);
	ofstream fout17(filename17);

	//for(int i=0; i<nparticle1; i++)
	//{

	//	double displaceX = OrinPositions[i][0] - Positions[i][0];
	//	double displaceY = OrinPositions[i][1] - Positions[i][1];
	//	double displaceZ = OrinPositions[i][2] - Positions[i][2];
	//	double distotal = displaceX*displaceX+displaceY*displaceY+displaceZ*displaceZ;

	//	if(OrinPositions[i][2]>0.0033&&OrinPositions[i][2]<0.0034)
	//	{
	//		if(Phase01[i]!=1)
	//		{
	//			fout1<<sqrt(distotal)<<endl;
	//		}
	//		else
	//			fout1<<"0"<<endl;
	//	}

	//	if(OrinPositions[i][0]>0.0033&&OrinPositions[i][0]<0.0034)
	//	{
	//
	//		if(Phase01[i]!=1)
	//		{
	//			fout2<<sqrt(distotal)<<endl;
	//		}
	//		else
	//			fout2<<"0"<<endl;
	//	}

	//	if(OrinPositions[i][1]>0.0033&&OrinPositions[i][1]<0.0034)
	//	{
	//		if(Phase01[i]!=1)
	//		{
	//			fout3<<sqrt(distotal)<<endl;
	//		}
	//		else
	//			fout3<<"0"<<endl;

	//	}

	//	//for(int j=0; j<Total_cell; j++)
	//	//{
	//	//	double VecXO = OrinPositions[Cell[j]][0]-OrinPositions[i][0];
	//	//	double VecYO = OrinPositions[Cell[j]][1]-OrinPositions[i][1];
	//	//	double VecZO = OrinPositions[Cell[j]][2]-OrinPositions[i][2];
	//	//	int index = floor(sqrt(VecXO*VecXO+VecYO*VecYO+VecZO*VecZO)*10000);

	//	//	Ur[index][0] += distotal;
	//	//	Ur[index][1] += 1;
	//	//}
	//}

	//for(int i=0; i<M; i++)
	//fout4<<i<<"\t"<<Ur[i][0]/(Ur[i][1]*Total_cell)<<endl;



        //fout5<<"# vtk DataFile Version 3.0"<<endl;
        //fout5<<"2D_to_3D example"<<endl;
        //fout5<<"ASCII"<<endl;
        //fout5<<"DATASET STRUCTURED_POINTS"<<endl;
        //fout5<<"DIMENSIONS 100 100 100"<<endl;
        //fout5<<"SPACING 1 1 1"<<endl;
        //fout5<<"ORIGIN 0 0 0"<<endl;
        //fout5<<"POINT_DATA 1000000"<<endl;
        //fout5<<"SCALARS volume_scalars UNSIGNED_INT 1"<<endl;
        //fout5<<"LOOKUP_TABLE default"<<endl;

        //fout6<<"# vtk DataFile Version 3.0"<<endl;
        //fout6<<"2D_to_3D example"<<endl;
        //fout6<<"ASCII"<<endl;
        //fout6<<"DATASET STRUCTURED_POINTS"<<endl;
        //fout6<<"DIMENSIONS 100 100 100"<<endl;
        //fout6<<"SPACING 1 1 1"<<endl;
        //fout6<<"ORIGIN 0 0 0"<<endl;
        //fout6<<"POINT_DATA 1000000"<<endl;
        //fout6<<"SCALARS volume_scalars UNSIGNED_INT 1"<<endl;
        //fout6<<"LOOKUP_TABLE default"<<endl;


	//for(int i=0; i<nparticle1; i++)
	//{
	//        if (count_if(bondsign[i], bondsign[i]+NB[i], Iszero)!=0)
	//	{
	//                damage = 4;
	//	}
	//        else
	//                damage = Phase01[i];
	//
	//        fout5<<damage<<endl;

	//	if (damage==4)
	//	fout6<<damage<<endl;
	//	else
	//	fout6<<"0"<<endl;

	//	for(int j=0; j<NB[i]; j++)
	//	{
	//		if((Phase01[i]!=Phase01[neighbors[i][j]])&&(Phase01[i]==2||Phase01[neighbors[i][j]]==2))
	//		{
	//			fout7<<i<<"\t"<<j<<"\t"<<neighbors[i][j]<<"\t"<<distance1[i][j]<<"\t"<<Stretch[i][j]<<"\t"<<nsign[i][j]<<endl;
	//		}
	//
	//	}
	//}
	/*
	fout9<<"Position_x"<<"\t"<<"Position_y"<<"\t"<<"Position_z"<<endl;
	fout10<<"Particle_i"<<"\t"<<"Neighbor_j"<<endl;
	fout11<<"dL_total_x"<<"\t"<<"dL_total_y"<<endl;
	fout12<<"TdL_total_x"<<"\t"<<"TdL_total_y"<<endl;
	fout13<<"Particle_i"<<"\t"<<"Neighbor_j"<<"\t"<<"bondsign_new"<<endl;
	fout14<<"Phase01"<<endl;
	*/
	int p = 20;
	for(int i=0; i<nparticle1; i++)
	{
		//double disx = OrinPositions[i][0] - Positions[i][0];
		//double disy = OrinPositions[i][1] - Positions[i][1];
		//double disz = OrinPositions[i][2] - Positions[i][2];
		//double distot = disx*disx+disy*disy+disz*disz;
		//int indicator=0;
		//if(Phase01[i]!=2)
		fout9<<fixed<<showpoint<<std::setprecision(p)<<Positions[i][0]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<Positions[i][1]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<Positions[i][2]<<endl;
		fout11<<fixed<<showpoint<<std::setprecision(p)<<dL_total[i][0]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<dL_total[i][1]<<endl;
		fout12<<fixed<<showpoint<<std::setprecision(p)<<TdL_total[i][0]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<TdL_total[i][1]<<endl;
		fout14<<fixed<<showpoint<<std::setprecision(p)<<Phase01[i]<<endl;

		fout15<<fixed<<showpoint<<std::setprecision(p)<<Lvel[i][0]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<Lvel[i][1]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<Lvel[i][2]<<endl;
		fout15.flush();
		fout16<<fixed<<showpoint<<std::setprecision(p)<<Lacc[i][0]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<Lacc[i][1]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<Lacc[i][2]<<endl;
		fout16.flush();
		fout17<<fixed<<showpoint<<std::setprecision(p)<<netF[i][0]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<netF[i][1]<<"\t"<<fixed<<showpoint<<std::setprecision(p)<<netF[i][2]<<endl;
		fout17.flush();

		for(int j=0; j<NB[i]; j++)
		{
			fout10<<i<<"\t"<<j<<"\t"<<dL[i][j]<<endl;
			fout13<<i<<"\t"<<j<<"\t"<<bondsign[i][j]<<endl;
		}
	}

return 0;

}

/*
void phase_expansion(double speed, int time)
{
        //int NB_1st_num;
        double VecXO,VecYO,VecZO,dis;

        #pragma omp parallel for private(VecXO,VecYO,VecZO,dis)
        for(int i=0; i<nparticle1; i++)
        {
                VecXO=0;
                VecYO=0;
                VecZO=0;

                if(Phase01[i]==1)
                {
                        for(int n=0; n<NB[i]; n++)
                        {
                            if(nsign[i][n]==1&&Phase01[neighbors[i][n]]!=1)
                            {
                                VecXO += OrinPositions[neighbors[i][n]][0]-OrinPositions[i][0];
                                VecYO += OrinPositions[neighbors[i][n]][1]-OrinPositions[i][1];
                                VecZO += OrinPositions[neighbors[i][n]][2]-OrinPositions[i][2];
                            }
                        }

                        dis = sqrt(VecXO*VecXO+VecYO*VecYO+VecZO*VecZO);
                        if(dis!=0)
                        {
                        VecXO = VecXO/dis;
                        VecYO = VecYO/dis;
                        VecZO = VecZO/dis;
                        }

                        Lvel[i][0] = speed*VecXO;
                        Lvel[i][1] = speed*VecYO;
                        Lvel[i][2] = speed*VecZO;


                }
        }
}

*/




int main ()
{
//int coreNum = omp_get_num_procs();
struct timespec time1 ={0,0};
struct timespec time2 ={0,0};
clock_gettime(CLOCK_REALTIME,&time1);
clock_t start,end;

start = clock();

srand(time(NULL));

ofstream fout("reaction_force.dat");
ofstream fout1("fracture_strength.dat");



Kn01 = 2.0*E0/((1.0 + mu0));
Kn02 = 2.0*E0/((1.0 + mu0));
Tv0 = E0*(4.0*mu0 - 1.0)/((9+4*sqrt(2))*(1.0 + mu0)*(1.0 - 2.0*mu0));

Kn11 = 2.0*E1/((1.0 + mu1));
Kn12 = 2.0*E1/((1.0 + mu1));
Tv1 = E1*(4.0*mu1 - 1.0)/((9+4*sqrt(2))*(1.0 + mu1)*(1.0 - 2.0*mu1));

Kn21 = 2.0*E2/((1.0 + mu2));
Kn22 = 2.0*E2/((1.0 + mu2));
Tv2 = E2*(4.0*mu2 - 1.0)/((9+4*sqrt(2))*(1.0 + mu2)*(1.0 - 2.0*mu2));

Initialization();


Mass = boxlength*boxlength*rho/(nparticle1+0.0);
cout<<Mass<<endl;

Searchneighbor();

output_map(0);

t_end = 1.0;
t_start = 0;
t_step=1E-9;
steps = floor((t_end-t_start)/t_step);

	for(int i=0; i<nedge[2]; i++)
	Lvel[Top[i]][1] = 0;

	for(int i=0; i<nedge[3]; i++)
	Lvel[Bottom[i]][1] = 0;

	for(int i=0; i<nedge[0]; i++)
	Lvel[Left[i]][0] = 0;

	for(int i=0; i<nedge[1]; i++)
	Lvel[Right[i]][0] = 0;

	for(int i=0; i<nedge[4]; i++)
	Lvel[Back[i]][2] = 0;

	for(int i=0; i<nedge[5]; i++)
	Lvel[Front[i]][2] = 0;

MaxF1=0;
int t;

for(t=1; t<=4; t++)
{

		for(int i=0; i<nparticle1; i++)
		{
			Positions[i][0] = Positions[i][0] + Lvel[i][0]*t_step + Lacc[i][0]*t_step*t_step/2.0;
			Positions[i][1] = Positions[i][1] + Lvel[i][1]*t_step + Lacc[i][1]*t_step*t_step/2.0;
			Positions[i][2] = Positions[i][2] + Lvel[i][2]*t_step + Lacc[i][2]*t_step*t_step/2.0;

			Lvel_t[i][0] = Lvel[i][0] + Lacc[i][0]*t_step/2.0;
			Lvel_t[i][1] = Lvel[i][1] + Lacc[i][1]*t_step/2.0;
			Lvel_t[i][2] = Lvel[i][2] + Lacc[i][2]*t_step/2.0;

		}

		output_map(t);
		Update();
		output_map(10+t);


		Netinteraction();

		output_map(20+t);

		for(int i=0; i<nparticle1; i++)
		{
			Lacc[i][0] = netF[i][0]/Mass;
			Lacc[i][1] = netF[i][1]/Mass;
			Lacc[i][2] = netF[i][2]/Mass;
		}

		for(int i=0; i<nparticle1; i++)
		{
			Lvel[i][0] = Lvel_t[i][0] + Lacc[i][0]*t_step/2.0;
			Lvel[i][1] = Lvel_t[i][1] + Lacc[i][1]*t_step/2.0;
			Lvel[i][2] = Lvel_t[i][2] + Lacc[i][2]*t_step/2.0;
		}

		output_map(30+t);




        	for(int i=0; i<nedge[0]; i++)
        	{
        		Lvel[Left[i]][0] = 0;
        		Lacc[Left[i]][0] = 0;
        	}

        	for(int i=0; i<nedge[1]; i++)
        	{
        		Lvel[Right[i]][0] = 0.5;
        		Lacc[Right[i]][0] = 0;
		}

		//for(int i=0; i<nedge[2]; i++)
		//{
		//	Lvel[Top[i]][1] = 0;
		//	Lacc[Top[i]][1] = 0;
		//}

		//for(int i=0; i<nedge[3]; i++)
		//{
		//	Lvel[Bottom[i]][1] = 0;
		//	Lacc[Bottom[i]][1] = 0;
		//}

		//for(int i=0; i<nedge[4]; i++)
		//{
		//	Lvel[Back[i]][2] = 0;
		//	Lacc[Back[i]][2] = 0;
		//}

		//for(int i=0; i<nedge[5]; i++)
		//{
		//	Lvel[Front[i]][2] = 0;
		//	Lacc[Front[i]][2] = 0;
		//}




	double tmf1=0;
        for(int i=0; i<nedge[0]; i++)
        {
            tmf1 += netF[Right[i]][0];
	    if(IsNan(netF[Right[i]][0]))
	    {
		cout<<"NaN in force!!!"<<endl;
		exit(0);
	    }
        }

        fout<<t<<"\t"<<fabs(tmf1)<<endl;

        if(fabs(tmf1)>MaxF1)
        MaxF1 = fabs(tmf1);

	//if(t>50000)
	//{
        //	if(fabs(tmf1)<(MaxF1/10.0))
	//	{
	//	fout1<<MaxF1<<endl;
        //	break;
	//	}
	//}

    	cout<<t<<endl;


//       int counter=0;
//       for(int i=0; i<nparticle1; i++)
//        {
//                if (count_if(bondsign[i], bondsign[i]+NB[i], Iszero)!=0)
//                {
//                        counter++;
//                }
//        }
	//if(t%1==0)
	//output_map(t);


}


//output_map(t);








//cout<<"cpu numbers: "<<coreNum<<endl;

clock_gettime(CLOCK_REALTIME,&time2);
cout<<"real time costs: "<<(time2.tv_sec-time1.tv_sec)<<"s"<<endl;


end = clock();


cout<<"Run time:"<<(double)(end-start)/CLOCKS_PER_SEC<<" s"<<endl;

return 0;
}