//----------------------------------------
// file name : RBMBase.cpp
// intended use : Base class of Restricted Boltzmann Machine
//
// creator : Takato Horii
//
// last update : 2019/09/23
//----------------------------------------

#include "RBMBase.h"

CRBMParam::CRBMParam()
	:batch_num(0)
	,case_num(0)
	,vis_num(0)
	,hid_num(0)
	,epsilon_w(0.01)
	,epsilon_b(0.01)
	,epsilon_c(0.01)
	,epsilon_bp(0.6)
	,moment(0.05)
	,err_th(0.01)
	,maxlearn_num(10000)
	,log_num(100)
	,wlog_flag(false)
	,RBM_name("unknown")
{
}

CRBMParam::~CRBMParam()
{
}

//Constructor
CRBMBase::CRBMBase()
	:epsilon_w(0.01)
	,epsilon_b(0.01)
	,epsilon_c(0.01)
	,epsilon_z(0.01)
	,epsilon_bp(0.6)
	,moment(0.05)
	,err_all(0.0)
	,err_th(0.01)
	,maxlearn_num(10000)
	,update_cnt(0)
	,log_num(100)
	,wlog_flag(false)
	,RBM_name("unknown")
{
}

//Destructor
CRBMBase::~CRBMBase()
{
}

//Init RBM
void CRBMBase::Initialize(string name, int b_num, int c_num, int v_num, int h_num, double ew, double eb, double ec, double ez, double ebp, double m, double e_th, int ml_num, int l_num, bool wlog)
{
	batch_num = b_num;
	case_num = c_num;

	vis_num = v_num;
	hid_num = h_num;

	epsilon_w = ew;
	epsilon_b = eb;
	epsilon_c = ec;
	epsilon_z = ez;
	epsilon_bp = ebp;
	moment = m;

	err_th = e_th;
	maxlearn_num = ml_num;

	log_num = l_num;
	wlog_flag = wlog;
	RBM_name = name;

	Init();
}

//Init RBM
void CRBMBase::Initialize(CRBMParam param)
{
	batch_num = param.batch_num;
	case_num = param.case_num;

	vis_num = param.vis_num;
	hid_num = param.hid_num;

	epsilon_w = param.epsilon_w;
	epsilon_b = param.epsilon_b;
	epsilon_c = param.epsilon_c;
	epsilon_z = param.epsilon_z;
	epsilon_bp = param.epsilon_bp;
	moment = param.moment;

	err_th = param.err_th;
	maxlearn_num = param.maxlearn_num;

	log_num = param.log_num;
	wlog_flag = param.wlog_flag;
	RBM_name = param.RBM_name;

	Init();
}

//Input data
void CRBMBase::InputData(vector<vector<vector<double>>> i_data)
{
	int batch_index = 0;
	int case_index = 0;

	for(batch_index=0;batch_index<batch_num;batch_index++){
		for(case_index=0;case_index<case_num;case_index++){
			for(int i=0;i<vis_num;i++){
				inputdata[batch_index][case_index][i] = i_data[batch_index][case_index][i];
				P_v0[batch_index][i] += i_data[batch_index][case_index][i]/case_num;
			}
		}
	}
}

//Output data
void CRBMBase::OutputData(vector<vector<vector<double>>>* o_data)
{
	int batch_index = 0;
	int case_index = 0;

	for(batch_index=0;batch_index<batch_num;batch_index++){
		vector<vector<double>> tmp2;
		for(case_index=0;case_index<case_num;case_index++){
			vector<double> tmp;
			for(int j=0;j<hid_num;j++){
				tmp.push_back(outputdata[batch_index][case_index][j]);
			}
			tmp2.push_back(tmp);
		}
		o_data->push_back(tmp2);
	}
}

//Training with BP
void CRBMBase::CalcRBM_BackProp()
{
	int batch_index = 0;
	int case_index = 0;

	err_all = 0.0;
	update_cnt = 0;

	//main loop
	while(update_cnt<maxlearn_num){

		update_cnt+=1;
		err_all = 0.0;
		for(batch_index=0;batch_index<batch_num;batch_index++){
			for(case_index=0;case_index<case_num;case_index++){
				Reconstruct(batch_index, case_index);
			}
			err_all += BackProp(batch_index);
		}

		if(update_cnt%log_num == 1){
			cout << "ERR_B = " << err_all << endl;
			SaveLog();
		}
		if(err_all < err_th){
			break;
		}
	}
	//output result	
	for(batch_index=0;batch_index<batch_num;batch_index++){
		for(case_index=0;case_index<case_num;case_index++){
			Reconstruct(batch_index, case_index);
		}
	}
	SaveLog();
	SaveWeight();
}

void CRBMBase::ConnectRBM(CRBMBase* next_rbm)
{
	int batch_index = 0;
	int case_index = 0;

	next_rbm->InputData(outputdata);
}

void CRBMBase::SetLabelData(vector<vector<vector<double>>> l_data)
{
	int batch_index = 0;
	int case_index = 0;

	for(batch_index=0;batch_index<batch_num;batch_index++){
		for(case_index=0;case_index<case_num;case_index++){
			for(int j=0;j<hid_num;j++){
				labeldata[batch_index][case_index][j] = l_data[batch_index][case_index][j];
				P_hhat[batch_index][j] += l_data[batch_index][case_index][j]/case_num;
			}
		}
	}
}

int CRBMBase::GetUpdateCnt()
{
	return update_cnt;
}

//Init function
void CRBMBase::Init()
{
	srand((unsigned int) time(0) );

	vector<double> tmp_v;
	for(int i=0;i<vis_num;i++){
		tmp_v.push_back(0.0);
	}
	vector<vector<double>> tmp2;
	for(int i=0;i<case_num;i++){
		tmp2.push_back(tmp_v);
	}
	inputdata.clear();
	for(int i=0;i<batch_num;i++){
		inputdata.push_back(tmp2);
	}

	vector<double> tmp_h;
	for(int i=0;i<hid_num;i++){
		tmp_h.push_back(0.0);
	}
	tmp2.clear();
	for(int i=0;i<case_num;i++){
		tmp2.push_back(tmp_h);
	}
	outputdata.clear();
	outputprob.clear();
	for(int i=0;i<batch_num;i++){
		outputdata.push_back(tmp2);
		outputprob.push_back(tmp2);
	}

	tmp2.clear();
	for(int i=0;i<case_num;i++){
		tmp2.push_back(tmp_h);
	}
	labeldata.clear();
	for(int i=0;i<batch_num;i++){
		labeldata.push_back(tmp2);
	}

	P_v0.clear();
	Act_v0.clear();
	for(int i=0;i<batch_num;i++){
		P_v0.push_back(tmp_v);
	}
	for(int i=0;i<case_num;i++){
		Act_v0.push_back(tmp_v);
	}

	P_v1.clear();
	Act_v1.clear();
	for(int i=0;i<case_num;i++){
		P_v1.push_back(tmp_v);
		Act_v1.push_back(tmp_v);
	}

	P_h0.clear();
	Act_h0.clear();
	for(int i=0;i<case_num;i++){
		P_h0.push_back(tmp_h);
		Act_h0.push_back(tmp_h);
	}

	P_h1.clear();
	Act_h1.clear();
	for(int i=0;i<case_num;i++){
		P_h1.push_back(tmp_h);
		Act_h1.push_back(tmp_h);
	}

	P_hhat.clear();
	for(int i=0;i<batch_num;i++){
		P_hhat.push_back(tmp_h);
	}

	P_re.clear();
	for(int i=0;i<vis_num;i++){
		P_re.push_back(0.0);
	}

	double alpha, beta;
	double w_rand;
	vector<double> w_tmp;
	vector<double> pre_dw_tmp;
	w.clear();
	pre_w.clear();
	pre_dw.clear();

	for(int i=0;i<vis_num;i++){
		w_tmp.clear();
		pre_dw_tmp.clear();
		for(int j=0;j<hid_num;j++){
			alpha = 1.0-((double)rand()/(RAND_MAX+1.0));
			beta = 1.0-((double)rand()/(RAND_MAX+1.0));
			w_rand = 0.01*sqrt(-2.0*log(alpha))*sin(2.0*M_PI*beta);
			w_tmp.push_back(w_rand);
			pre_dw_tmp.push_back(0.0);
		}
		w.push_back(w_tmp);
		pre_w.push_back(w_tmp);
		pre_dw.push_back(pre_dw_tmp);
	}

	b.clear();
	pre_db.clear();
	for(int i=0;i<hid_num;i++){
		b.push_back(0.0);
		pre_db.push_back(0.0);
	}

	c.clear();
	pre_dc.clear();
	for(int i=0;i<vis_num;i++){
		c.push_back(0.0);
		pre_c.push_back(0.0);
		pre_dc.push_back(0.0);
		in_var.push_back(0.0);
	}

	sig.clear();
	for(int i=0;i<vis_num;i++){
		sig.push_back(1.0);
	}

	z.clear();
	pre_dz.clear();
	for(int i=0;i<vis_num;i++){
		z.push_back(0.0);
		pre_dz.push_back(0.0);
	}
}

double CRBMBase::BackProp(int batch_index)
{
	int case_index = 0;
	double err = 0.0;

	double dw = 0.0;

	for(int i=0;i<vis_num;i++){
		for(int j=0;j<hid_num;j++){
			dw = 0.0;
			for(case_index=0;case_index<case_num;case_index++){
				dw += (-1.0)*epsilon_bp*Act_v0[case_index][i]*(P_h0[case_index][j] - labeldata[batch_index][case_index][j]);//case_num;
				err += fabs(P_h0[case_index][j] - labeldata[batch_index][case_index][j]);
			}
			w[i][j] += (dw + moment*pre_dw[i][j]);
			pre_dw[i][j] = dw;
		}
	}

	return err;
}

void CRBMBase::SaveWeight()
{
	weight_file.open(RBM_name+"_w.csv");

	for(int i=0;i<vis_num;i++){
		for(int j=0;j<hid_num;j++){
			weight_file << w[i][j];
			if(j==(hid_num-1)){
				weight_file << endl;
			}
			else{
				weight_file << ",";
			}
		}
	}

	for(int j=0;j<hid_num;j++){
		weight_file << b[j];
		if(j==(hid_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}

	for(int i=0;i<vis_num;i++){
		weight_file << c[i];
		if(i==(vis_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}

	for(int i=0;i<vis_num;i++){
		weight_file << z[i];
		if(i==(vis_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}
	weight_file.close();

	ofstream outfile;
	outfile.open(RBM_name+"_outdata.csv");
	for(int batch_index=0;batch_index<batch_num;batch_index++){
		for(int case_index=0;case_index<case_num;case_index++){
			for(int j=0;j<hid_num;j++){
				outfile << outputdata[batch_index][case_index][j];
				if(j==(hid_num-1)){
					outfile << endl;
				}
				else{
					outfile << ",";
				}
			}
		}
	}
	outfile.close();

	outfile.open(RBM_name+"_outprob.csv");
	for(int batch_index=0;batch_index<batch_num;batch_index++){
		for(int case_index=0;case_index<case_num;case_index++){
			for(int j=0;j<hid_num;j++){
				outfile << outputprob[batch_index][case_index][j];
				if(j==(hid_num-1)){
					outfile << endl;
				}
				else{
					outfile << ",";
				}
			}
		}
	}
	outfile.close();
}

void CRBMBase::SaveWeight(int num)
{

	stringstream ss;
	ss << num;
	weight_file.open(RBM_name+"_w"+ss.str()+".csv");

	for(int i=0;i<vis_num;i++){
		for(int j=0;j<hid_num;j++){
			weight_file << w[i][j];
			if(j==(hid_num-1)){
				weight_file << endl;
			}
			else{
				weight_file << ",";
			}
		}
	}

	for(int j=0;j<hid_num;j++){
		weight_file << b[j];
		if(j==(hid_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}

	for(int i=0;i<vis_num;i++){
		weight_file << c[i];
		if(i==(vis_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}
	weight_file.close();

	ofstream outfile;
	outfile.open(RBM_name+"_outdata"+ss.str()+".csv");
	for(int batch_index=0;batch_index<batch_num;batch_index++){
		for(int case_index=0;case_index<case_num;case_index++){
			for(int j=0;j<hid_num;j++){
				outfile << outputdata[batch_index][case_index][j];
				if(j==(hid_num-1)){
					outfile << endl;
				}
				else{
					outfile << ",";
				}
			}
		}
	}
	outfile.close();

	outfile.open(RBM_name+"_outprob"+ss.str()+".csv");
	for(int batch_index=0;batch_index<batch_num;batch_index++){
		for(int case_index=0;case_index<case_num;case_index++){
			for(int j=0;j<hid_num;j++){
				outfile << outputprob[batch_index][case_index][j];
				if(j==(hid_num-1)){
					outfile << endl;
				}
				else{
					outfile << ",";
				}
			}
		}
	}
	outfile.close();
}

void CRBMBase::SaveLog()
{
	if(wlog_flag){
		weight_file_all.open(RBM_name+"_wlog.csv",ios::app);
		for(int i=0;i<vis_num;i++){
			for(int j=0;j<hid_num;j++){
				weight_file_all << w[i][j] << ",";
			}
		}
		weight_file_all << endl;
		weight_file_all.close();

		b_file_all.open(RBM_name+"_blog.csv",ios::app);
		for(int i=0;i<hid_num;i++){
				b_file_all << b[i] << ",";
		}
		b_file_all << endl;
		b_file_all.close();

		c_file_all.open(RBM_name+"_clog.csv",ios::app);
		for(int i=0;i<vis_num;i++){
				c_file_all << c[i] << ",";
		}
		c_file_all << endl;
		c_file_all.close();

		z_file_all.open(RBM_name+"_zlog.csv",ios::app);
		for(int i=0;i<vis_num;i++){
				z_file_all << z[i] << ",";
		}
		z_file_all << endl;
		z_file_all.close();
	}

	err_file.open(RBM_name+"_err.csv",ios::app);
	err_file << err_all << endl;
	err_file.close();
}

void CRBMBase::LoadWeight(string filename)
{
	char comma;
	ifstream infile;
	infile.open(filename);
	if(!infile.is_open()){
		cout << "File can't load weight data" << endl;
		return;
	}

	for(int i=0;i<vis_num;i++){
		for(int j=0;j<hid_num;j++){
			infile >> w[i][j];
			if(j!=(hid_num-1)){
				infile >> comma;
			}
		}
	}

	for(int j=0;j<hid_num;j++){
		infile >> b[j];
		if(j!=(hid_num-1)){
			infile >> comma;
		}
	}

	for(int i=0;i<vis_num;i++){
		infile >> c[i];
		if(i!=(vis_num-1)){
			infile >> comma;
		}
	}

	infile.close();
}

double CRBMBase::Sigmoid(double x)
{
	return 1.0 / (1.0 + exp(x));
}

double CRBMBase::Gaussian(double x, double mu, double sigma_2)
{
	return 1.0/sqrt(2.0*M_PI*sigma_2) * exp(-(x-mu)*(x-mu)/(2.0*sigma_2));
}

double CRBMBase::Gauss_Rand(double mu, double sigma_2)
{
	double alpha = 1.0-((double)rand()/(RAND_MAX+1.0));
	double beta = 1.0-((double)rand()/(RAND_MAX+1.0));
	return sqrt(-2.0*log(alpha))*sin(2.0*M_PI*beta) * sqrt(sigma_2) + mu;
}
