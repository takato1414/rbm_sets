//----------------------------------------
// file name : BBRTRBM.h
// intended use : Gaussian-Bernoulli Recurrent Temporal Restricted Boltzmann Machine
//
// author : Takato Horii
//
// last update : 2019/09/23
//----------------------------------------

#ifndef _GBRTRBM_H_INCLUDED_
#define _GBRTRBM_H_INCLUDED_

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

#include "RecurrentTemporalRBMBase.h"

using namespace std;

class GBRTRBM : public RecurrentTemporalRBMBase
{
public:
	GBRTRBM();
	~GBRTRBM();

public:
	//RBM training
	virtual void CalcRBM();
	//Data abstraction from trigger data (i.e., input)
	virtual void Abstract(vector<double> trigger_data, vector<double>& gen_data, vector<double>& gen_prob);
	//Recall visible data from trigger data (i.e., hidden data)
	virtual void Recall(vector<double> trigger_data, vector<double>& gen_data, vector<double>& gen_prob);
	//Recall visible data
	virtual void Recall();
	
protected:
	//Reconstruction from s single hidden data
	virtual double Reconstruct(int batch_index, int case_index);
	//Update network weight
	virtual void UpdateWeight(int batch_index);
};

#endif //_GBRTRBM_H_INCLUDED_
