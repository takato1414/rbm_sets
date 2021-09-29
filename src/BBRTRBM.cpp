//----------------------------------------
// file name : BBRTRBM.cpp
// intended use : Bernoulli-Bernoulli Recurrent Temporal Restricted Boltzmann Machine
//
// author : Takato Horii
//
// last update : 2019/09/23
//----------------------------------------

#include "BBRTRBM.h"

//Constructor
BBRTRBM::BBRTRBM()
	:RecurrentTemporalRBMBase()
{
}

//Destructor
BBRTRBM::~BBRTRBM()
{
}

//RBM training
void BBRTRBM::CalcRBM()
{
	int batch_index = 0;
	int case_index = 0;

	err_all = 0.0;
	update_cnt = 0;

	//main loop
	while(update_cnt<p.maxlearn_num){
		
		if(update_cnt == 0){
			SaveWeight(update_cnt);
		}

		err_all = 0.0;
		for(batch_index=0;batch_index<p.batch_num;batch_index++){
			for(case_index=0;case_index<p.case_num;case_index++){
				err_all += Reconstruct(batch_index, case_index);
			}
			UpdateWeight(batch_index);
		}
		update_cnt+=1;

		if(update_cnt%p.log_num == 0 || update_cnt==1){
			cout << "epoch : " << update_cnt << " " << p.RBM_name <<" ERR = " << err_all << endl;
			SaveLog();
		}
		if(update_cnt%(p.log_num*10) == 0 || update_cnt==1){
			SaveWeight(update_cnt);
		}
		if(err_all < p.err_th){
			break;
		}
	}
	//output result
	for(batch_index=0;batch_index<p.batch_num;batch_index++){
		for(case_index=0;case_index<p.case_num;case_index++){
			Reconstruct(batch_index, case_index);
		}
	}
	SaveLog();
	SaveWeight();
}

//Data abstraction from trigger data (i.e., input)
void BBRTRBM::Abstract(vector<double> trigger_data, vector<double>& gen_data, vector<double>& gen_prob)
{
	VectorXd visible_data = Eigen::Map<Eigen::VectorXd>(&trigger_data[0], trigger_data.size());
	VectorXd hidden_prob;
	VectorXd hidden_data;

	VectorXd tmp_h = (-1.0)*(w.transpose() * visible_data) - b;

	hidden_prob = tmp_h.array().unaryExpr(CalcFP(sigmoid_a));
	hidden_data = hidden_prob.array().unaryExpr(CalcFP(act_a));

	gen_prob.resize(p.hid_num);
	Map<VectorXd>(&gen_prob[0], p.hid_num) = hidden_prob;
	
	gen_data.resize(p.hid_num);
	Map<VectorXd>(&gen_data[0], p.hid_num) = hidden_data;
}

//Recall visible data from trigger data (i.e., hidden data)
void BBRTRBM::Recall(vector<double> trigger_data, vector<double>& gen_data, vector<double>& gen_prob)
{
	VectorXd hidden_data = Eigen::Map<Eigen::VectorXd>(&trigger_data[0], trigger_data.size());
	VectorXd visible_prob;
	VectorXd visible_data;

	VectorXd tmp_v = (-1.0)*(w.transpose() * hidden_data) - c;

	visible_prob = tmp_v.array().unaryExpr(CalcFP(sigmoid_a));
	visible_data = visible_prob.array().unaryExpr(CalcFP(act_a));

	gen_prob.resize(p.vis_num);
	Map<VectorXd>(&gen_prob[0], p.vis_num) = visible_prob;
	
	gen_data.resize(p.vis_num);
	Map<VectorXd>(&gen_data[0], p.vis_num) = visible_data;
}

//Recall visible data
void BBRTRBM::Recall()
{
}

//Reconstruction from s single hidden data
double BBRTRBM::Reconstruct(int batch_index, int case_index)
{
	double err = 0.0;

	Act_v0[case_index] = inputdata[batch_index][case_index];
	bistar[case_index] = VectorXd::Zero(p.vis_num);
	bjstar[case_index] = VectorXd::Zero(p.hid_num);

	//Calculate contributions from directed autoregressive connections
	if(case_index>=p.preb_window_num){
		for(int t=1;t<=p.preb_window_num;t++){
			bjstar[case_index] = A[t] * P_h1[case_index-t];
		}
	}

	//Calculate contributions from directed hidden-to-visible connections
	if(case_index>=p.preb_window_num){
		for(int t=1;t<=p.preb_window_num;t++){
			bistar[case_index] = B[t] * P_h1[case_index-t];
		}
	}

	//First contrastive
	VectorXd tmp_h = (-1.0)*(w.transpose() * Act_v0[case_index]) - b - bjstar[case_index];
	P_h0[case_index] = tmp_h.array().unaryExpr(CalcFP(sigmoid_a));
	Act_h0[case_index] = P_h0[case_index].array().unaryExpr(CalcFP(act_a));
	
	//First reconstructive
	VectorXd tmp_v = (-1.0)*(w * Act_h0[case_index]) - c - bistar[case_index];
	P_v1[case_index] = tmp_v.array().unaryExpr(CalcFP(sigmoid_a));
	Act_v1[case_index] = P_v1[case_index].array().unaryExpr(CalcFP(act_a));

	//Second contrastive
	tmp_h = (-1.0)*(w.transpose() * Act_v1[case_index]) - b - bjstar[case_index];
	P_h1[case_index] = tmp_h.array().unaryExpr(CalcFP(sigmoid_a));
	Act_h1[case_index] = P_h1[case_index].array().unaryExpr(CalcFP(act_a));
	
	outputdata[batch_index][case_index] = Act_h0[case_index];
	outputprob[batch_index][case_index] = P_h0[case_index];

	//Calc Error
	err = (Act_v0[case_index] - Act_v1[case_index]).array().abs().array().sum();

	return err;
}

//Update network weight
void BBRTRBM::UpdateWeight(int batch_index)
{
	MatrixXd dw = MatrixXd::Zero(p.vis_num, p.hid_num);
	VectorXd db = VectorXd::Zero(p.hid_num);
	VectorXd dc = VectorXd::Zero(p.vis_num);
	VectorXd dz = VectorXd::Zero(p.vis_num);
	MatrixXd dA = MatrixXd::Zero(p.hid_num, p.hid_num);
	MatrixXd dB = MatrixXd::Zero(p.hid_num, p.vis_num);

	for(int c_index=0;c_index<p.case_num;c_index++){
		dw += p.epsilon_w/p.case_num*((Act_v0[c_index] * P_h0[c_index].transpose()) - (Act_v1[c_index] * P_h1[c_index].transpose()));
	}
	pre_w = w;
	w += (dw + p.moment*pre_dw);
	pre_dw = (dw + p.moment*pre_dw);

	for(int c_index=0;c_index<p.case_num;c_index++){
		db += p.epsilon_b/p.case_num*(P_h0[c_index] - P_h1[c_index]);
	}
	b += (db + p.moment*pre_db);
	pre_db = (db + p.moment*pre_db);

	for(int c_index=0;c_index<p.case_num;c_index++){
		dc += p.epsilon_c/p.case_num*(Act_v0[c_index] - P_v1[c_index]);
	}
	pre_c = c;
	c += (dc + p.moment*pre_dc);
	pre_dc = (dc + p.moment*pre_dc);
}