//----------------------------------------
// file name : main.cpp
// intended use : Deep Belief Net for motion
//
// author : Takato Horii
//
// last update : 2019/09/23
//----------------------------------------

#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include "RBMBase.h"
#include "RBM_VbHb.h"
#include "RBM_VlHb.h"
#include "RBM_VbHl.h"
#include "RBM_VlHl.h"

#include "Const.h"
#include "FeatureData.h"
#include "StimulusData.h"

using namespace std;

//Version info
string exename = "ForMotion";
string version = "Ver.1.0.0";
string lastupdate = "2019_09_23";

//Variable
#define BATCH_NUM 1
#define CASE_NUM 30
#define WINDOW_NUM 0
#define DATA_DIM 120
#define LABEL_NUM 0

//For input data
vector<vector<vector<double>>> input_data;
vector<vector<vector<double>>> label_data;

bool InputDatafromFile(char filename);
bool InputDatafromFile(string filename);
bool InputDatawithLabelfromFile(char* filename);


//sample source
int main(int argc, char* argv[])
{
	//Version info out
	cout << "start " << exename << endl;
	cout << version << endl;
	cout << "Last up date : " << lastupdate << endl << endl;

	InputDatafromFile(argv[1]);

	RecurrentTemporalRBMBase * first_rbm = new GBRTRBM();
	RecurrentTemporalRBMBase * second_rbm = new BBRTRBM();

	string RBMName[2] = {"L1_Mot_1_GB", "L1_Mot_2_BB"};
	int node_num[4] = {DATA_DIM, 60, 30};

	//set parameter
	ConditionalRBMParam first_param;
	first_param.RBM_name = RBMName[0];
	first_param.batch_num = BATCH_NUM;
	first_param.case_num = input_data[0].size();
	first_param.preb_window_num = WINDOW_NUM;

	first_param.vis_num = node_num[0];
	first_param.hid_num = node_num[1];
	
	first_param.epsilon_w = 0.001;
	first_param.epsilon_b = 0.001;
	first_param.epsilon_c = 0.001;
	first_param.epsilon_z = 0.001;
	first_param.epsilon_s = 0.001;
	first_param.epsilon_bp = 0.1;
	first_param.epsilon_A = 0.01;
	first_param.epsilon_B = 0.01;
	first_param.moment = 0.0005;
	
	first_param.maxlearn_num = 10000;
	first_param.err_th = 0.0005;

	first_param.seed = 0;
	first_param.is_hidden_sparse = false;
	
	first_param.log_num = 10;
	first_param.wlog_flag = true;

	cout << "Init" << endl;
	first_rbm->Initialize(first_param);


	ConditionalRBMParam second_param;
	second_param.RBM_name = RBMName[1];
	second_param.batch_num = BATCH_NUM;
	second_param.case_num = input_data[0].size();
	second_param.preb_window_num = 0;

	second_param.vis_num = node_num[1];
	second_param.hid_num = node_num[2];
	
	second_param.epsilon_w = 0.01;
	second_param.epsilon_b = 0.01;
	second_param.epsilon_c = 0.01;
	second_param.epsilon_z = 0.01;
	second_param.epsilon_bp = 0.1;
	second_param.epsilon_A = 0.001;
	second_param.epsilon_B = 0.001;
	second_param.moment = 0.005;
	
	second_param.seed = 0;
	second_param.is_hidden_sparse = false;

	second_param.maxlearn_num = 10000;
	second_param.err_th = 0.001;
	
	second_param.log_num = 10;
	second_param.wlog_flag = true;

	cout << "Init" << endl;
	second_rbm->Initialize(second_param);

	cout << "Batch_num = " << BATCH_NUM << endl;
	cout << "Case_num = " << input_data[0].size() << endl;
	cout << "Window_num = " << WINDOW_NUM << endl;
	cout << "start" << endl;

	//for low layer
	first_rbm->InputData(input_data);
	first_rbm->CalcRBM();
	first_rbm->ConnectRBM(second_rbm);
	second_rbm->CalcRBM();

	cout << "Finish!" << endl;
	
	return 0;
}

bool InputDatafromFile(char* filename){
	int file_num = 0;
	ifstream inFile;
	char comma;
	char label[8192];
	int area = DATA_DIM;
	double tmp;
	vector<double> Vis;
	vector<vector<double>> all_data;

	cout << "Input from " << filename << endl;
	inFile.open(filename);
	if(!inFile.is_open()){
		cout << "File can't open" << endl;
		return false;
	}
	all_data.clear();

	while(!inFile.eof()){
		Vis.clear();
		for(int i=0;i<area;i++){
			inFile >> tmp;
			Vis.push_back(tmp);
			if(i!=(area-1)){
				inFile >> comma;
			}
		}
		all_data.push_back(Vis);
	}
	all_data.pop_back();
	
	inFile.close();

	input_data.clear();
	for(int t=0;t<BATCH_NUM;t++){
		input_data.push_back(all_data);
	}

	return true;
}

bool InputDatafromFile(string filename){
	int file_num = 0;
	ifstream inFile;
	char comma;
	char label[8192];
	int area = DATA_DIM;
	double tmp;
	vector<double> Vis;
	vector<vector<double>> all_data;

	cout << "Input from " << filename << endl;
	inFile.open(filename);
	if(!inFile.is_open()){
		cout << "File can't open" << endl;
		return false;
	}
	all_data.clear();

	while(!inFile.eof()){
		Vis.clear();
		for(int i=0;i<area;i++){
			inFile >> tmp;
			Vis.push_back(tmp);
			if(i!=(area-1)){
				inFile >> comma;
			}
		}
		all_data.push_back(Vis);
	}
	all_data.pop_back();
	
	inFile.close();

	input_data.clear();
	for(int t=0;t<BATCH_NUM;t++){
		input_data.push_back(all_data);
	}

	return true;
}

bool InputDatawithLabelfromFile(char* filename){
	int file_num = 0;
	ifstream inFile;
	char comma;
	char label[8192];
	int area = DATA_DIM+LABEL_NUM;
	double tmp;
	vector<double> Vis;
	vector<double> tmpl;
	vector<vector<double>> all_data;
	vector<vector<double>> l_tmp;

	cout << "Input from " << filename << endl;
	inFile.open(filename);
	if(!inFile.is_open()){
		cout << "File can't open" << endl;
		return false;
	}
	all_data.clear();

	while(!inFile.eof()){
		Vis.clear();
		for(int i=0;i<area;i++){
			inFile >> tmp;
			Vis.push_back(tmp);
			if(i!=(area-1)){
				inFile >> comma;
			}
		}
		all_data.push_back(Vis);
	}
	all_data.pop_back();
	
	inFile.close();

	l_tmp.clear();
	vector<double>::iterator it;
	for(int i=0;i<all_data.size();i++){
		tmpl.clear();
		for(int j=0;j<LABEL_NUM;j++){
			tmpl.push_back(all_data[i][0]);
			it = all_data[i].begin();
			all_data[i].erase(it);
		}
		l_tmp.push_back(tmpl);
	}

	input_data.clear();
	for(int t=0;t<BATCH_NUM;t++){
		input_data.push_back(all_data);
		label_data.push_back(l_tmp);
	}

	return true;
}