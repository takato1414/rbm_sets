//----------------------------------------
// file name : RecurrentTemporalRBMBase.h
// intended use : Base class of Recurrent Temporal Restricted Boltzmann Machine
//
// creator : Takato Horii
//
// last update : 2019/09/23
//----------------------------------------

#ifndef _RECURRENT_TEMPORAL_RBMBASE_H_INCLUDED_
#define _RECURRENT_TEMPORAL_RBMBASE_H_INCLUDED_

//include files
#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

//#define EIGEN_NO_DEBUG
#include <Eigen/Dense>

#include "ConditionalRBMBase.h"
using namespace std;
using namespace Eigen;

typedef double (*CalcFP)(double x);

double sigmoid_a(double x);

double act_a(double x);

double thz_a(double x);

class RecurrentTemporalRBMBase
{
public:
	RecurrentTemporalRBMBase();
	~RecurrentTemporalRBMBase();

protected:
	ConditionalRBMParam p;

	RecurrentTemporalRBMBase* prebRBM;

	int update_cnt;
	double err_all;

	vector<vector<VectorXd>> inputdata;
	vector<vector<VectorXd>> outputdata;
	vector<vector<VectorXd>> outputprob;
	vector<vector<VectorXd>> labeldata;

	vector<VectorXd> P_v0;
	vector<VectorXd> P_v1;
	vector<VectorXd> P_h0;
	vector<VectorXd> P_h1;
	vector<VectorXd> P_hhat;
	

	vector<VectorXd> Act_v0;
	vector<VectorXd> Act_v1;
	vector<VectorXd> Act_h0;
	vector<VectorXd> Act_h1;

	vector<VectorXd> bistar;	//contributions from directed autoregressive connections
	vector<VectorXd> bjstar;	//contributions from directed visible-to-hidden connections

	MatrixXd w;
	MatrixXd pre_w;
	vector<MatrixXd> A;
	vector<MatrixXd> B;
	VectorXd b;
	VectorXd pre_b;
	VectorXd c;
	VectorXd pre_c;
	VectorXd sig;
	VectorXd z;
	VectorXd pre_z;
	VectorXd sig_h;
	VectorXd s;
	VectorXd pre_s;

	MatrixXd pre_dw;
	vector<MatrixXd> pre_dA;
	vector<MatrixXd> pre_dB;
	VectorXd pre_db;
	VectorXd pre_dc;
	VectorXd pre_dz;
	VectorXd pre_ds;

	VectorXd P_re;
	vector<VectorXd> recall_data;

	ofstream weight_file;
	ofstream weight_file_all;
	ofstream b_file_all;
	ofstream c_file_all;
	ofstream z_file_all;
	ofstream s_file_all;
	ofstream err_file;

public:
	//Init RBM
	void Initialize(ConditionalRBMParam param);
	//Set param for RBM
	void SetParameter(ConditionalRBMParam param);
	//Input data
	void InputData(vector<vector<vector<double>>> i_data);
	//RBM training
	virtual void CalcRBM() = 0;
	//Output data
	void OutputData(vector<vector<vector<double>>>* o_data);
	//Assign the output data to the input data of the next RBM
	void ConnectRBM(ConditionalRBMBase* next_rbm);

	//Data abstraction from trigger data (i.e., input)
	virtual void Abstract(vector<double> trigger_data, vector<double>& gen_data, vector<double>& gen_prob) = 0;
	//Recall visible data from trigger data (i.e., hidden data)
	virtual void Recall(vector<double> trigger_data, vector<double>& gen_data, vector<double>& gen_prob) = 0;
	//Recall visible data
	virtual void Recall() = 0;
	

	void SetLabelData(vector<vector<vector<double>>> l_data);
	//Training with BP
	void CalcRBM_BackProp();

	void LoadWeight(string filename);

	int GetUpdateCnt();
	
protected:
	//Init function
	void Init();
	//Reconstruction from s single hidden data
	virtual double Reconstruct(int batch_index, int case_index) = 0;
	//Update network weight
	virtual void UpdateWeight(int batch_index) = 0;


	double BackProp(int batch_index);

	void SaveWeight();

	void SaveWeight(int num);

	void SaveLog();


	double Sigmoid(double x);

	double Gaussian(double x, double mu=0.0, double sigma_2=1.0);

	double Gauss_Rand(double mu=0.0, double sigma_2=1.0);

};

#endif //_RECURRENT_TEMPORAL_RBMBASE_H_INCLUDED_
