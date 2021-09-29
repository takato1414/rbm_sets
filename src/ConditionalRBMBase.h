//----------------------------------------
// file name : ConditionalRBMBase.cpp
// intended use : Base class of Conditional Restricted Boltzmann Machine
//
// creator : Takato Horii
//
// last update : 2019/09/23
//----------------------------------------

#ifndef _CONDITIONAL_RBMBASE_H_INCLUDED_
#define _CONDITIONAL_RBMBASE_H_INCLUDED_

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

using namespace std;

class ConditionalRBMParam{
public:
	ConditionalRBMParam();
	~ConditionalRBMParam();
public:
	int batch_num;
	int case_num;
	int preb_window_num;

	int vis_num;
	int hid_num;

	double epsilon_w;
	double epsilon_b;
	double epsilon_c;
	double epsilon_z;
	double epsilon_s;
	double epsilon_A;
	double epsilon_B;
	double epsilon_bp
	double moment;

	double err_th;
	int maxlearn_num;
	int log_num;
	bool wlog_flag;
	string RBM_name;

	unsigned int seed;
	bool is_hidden_sparse;
};

class ConditionalRBMBase
{
public:
	ConditionalRBMBase();
	~ConditionalRBMBase();

protected:
	ConditionalRBMParam p;

	ConditionalRBMBase* prebRBM;

	int update_cnt;
	double err_all;

	vector<vector<vector<double>>> inputdata;
	vector<vector<vector<double>>> outputdata;
	vector<vector<vector<double>>> outputprob;
	vector<vector<vector<double>>> labeldata;

	vector<vector<double>> P_v0;
	vector<vector<double>> P_v1;
	vector<vector<double>> P_h0;
	vector<vector<double>> P_h1;
	vector<vector<double>> P_hhat;

	vector<vector<double>> Act_v0;
	vector<vector<double>> Act_v1;
	vector<vector<double>> Act_h0;
	vector<vector<double>> Act_h1;

	vector<vector<double>> bistar;	//contributions from directed autoregressive connections
	vector<vector<double>> bjstar;	//contributions from directed visible-to-hidden connections
	
	vector<vector<double>> w;
	vector<vector<double>> pre_w;
	vector<vector<vector<double>>> A;
	vector<vector<vector<double>>> B;
	vector<double> b;
	vector<double> c;
	vector<double> pre_c;
	vector<double> sig;
	vector<double> z;

	vector<vector<double>> pre_dw;
	vector<vector<vector<double>>> pre_dA;
	vector<vector<vector<double>>> pre_dB;
	vector<double> pre_db;
	vector<double> pre_dc;
	vector<double> pre_dz;

	vector<double> P_re;
	vector<vector<double>> recall_data;

	ofstream weight_file;
	ofstream weight_file_all;
	ofstream b_file_all;
	ofstream c_file_all;
	ofstream z_file_all;
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

	virtual vector<double> Recognize(vector<double> trigger_data) = 0;

	virtual vector<double> RecognizeProb(vector<double> trigger_data) = 0;

	virtual void Recollect() = 0;

	virtual vector<double> Recollect(vector<double> trigger_data) = 0;

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

#endif //_CONDITIONAL_RBMBASE_H_INCLUDED_
