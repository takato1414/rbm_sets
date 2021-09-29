//----------------------------------------
// file name : RBMBase.cpp
// intended use : Base class of Restricted Boltzmann Machine
//
// creator : Takato Horii
//
// last update : 2019/09/23
//----------------------------------------

#ifndef _RBMBASE_H_INCLUDED_
#define _RBMBASE_H_INCLUDED_

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

class CRBMParam{
public:
	CRBMParam();
	~CRBMParam();
public:
	int batch_num;
	int case_num;

	int vis_num;
	int hid_num;

	double epsilon_w;
	double epsilon_b;
	double epsilon_c;
	double epsilon_z;
	double epsilon_bp;
	double moment;

	double err_th;
	int maxlearn_num;
	int log_num;
	bool wlog_flag;
	string RBM_name;
};

class CRBMBase
{
public:
	CRBMBase();
	~CRBMBase();

protected:
	int batch_num;
	int case_num;

	int vis_num;
	int hid_num;

	double epsilon_w;
	double epsilon_b;
	double epsilon_c;
	double epsilon_z;
	double epsilon_bp;
	double moment;

	double err_all;
	double err_th;
	int maxlearn_num;

	int update_cnt;

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

	vector<vector<double>> w;
	vector<vector<double>> pre_w;
	vector<double> b;
	vector<double> c;
	vector<double> pre_c;
	vector<double> sig;
	vector<double> z;

	vector<vector<double>> pre_dw;
	vector<double> pre_db;
	vector<double> pre_dc;
	vector<double> pre_dz;

	vector<double> in_var;

	vector<double> P_re;
	vector<vector<double>> recall_data;

	bool wlog_flag;
	int log_num;
	string RBM_name;
	ofstream weight_file;
	ofstream weight_file_all;
	ofstream b_file_all;
	ofstream c_file_all;
	ofstream z_file_all;
	ofstream err_file;

public:
	//Init RBM
	void Initialize(string name, int b_num, int c_num, int v_num, int h_num, double ew=0.01, double eb=0.01, double ec=0.01, double ez=0.01, double ebp=0.6, double m=0.05, double e_th=0.001, int ml_num=10000, int l_num=100, bool wlog=false);
	//Init RBM
	void Initialize(CRBMParam param);
	//Input data
	void InputData(vector<vector<vector<double>>> i_data);
	//RBM training
	virtual void CalcRBM() = 0;
	//Output data
	void OutputData(vector<vector<vector<double>>>* o_data);
	//Assign the output data to the input data of the next RBM
	void ConnectRBM(CRBMBase* next_rbm);

	virtual vector<double> Recognize(vector<double> trigger_data) = 0;

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

#endif //_RBMBASE_H_INCLUDED_
