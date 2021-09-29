//----------------------------------------
// file name : ConditionalRBMBase.cpp
// intended use : Base class of Conditional Restricted Boltzmann Machine
//
// creator : Takato Horii
//
// last update : 2019/09/23
//----------------------------------------

#include "ConditionalRBMBase.h"

ConditionalRBMParam::ConditionalRBMParam()
	:batch_num(0)
	,case_num(0)
	,preb_window_num(0)
	,vis_num(0)
	,hid_num(0)
	,epsilon_w(0.01)
	,epsilon_b(0.01)
	,epsilon_c(0.01)
	,epsilon_z(0.01)
	,epsilon_A(0.01)
	,epsilon_B(0.01)
	,epsilon_bp(0.6)
	,moment(0.05)
	,err_th(0.01)
	,maxlearn_num(10000)
	,log_num(100)
	,wlog_flag(false)
	,RBM_name("unknown")
	,seed((unsigned int)time(0))
	,is_hidden_sparse(false)
{
}

ConditionalRBMParam::~ConditionalRBMParam()
{
}

//Constructor
ConditionalRBMBase::ConditionalRBMBase()
{
}

//Destructor
ConditionalRBMBase::~ConditionalRBMBase()
{
}


//Init RBM
void ConditionalRBMBase::Initialize(ConditionalRBMParam param)
{
	p = param;

	Init();
}

//Set param for RBM
void ConditionalRBMBase::SetParameter(ConditionalRBMParam param)
{
	p = param;
}

//Input data
void ConditionalRBMBase::InputData(vector<vector<vector<double>>> i_data)
{
	int batch_index = 0;
	int case_index = 0;

	for(batch_index=0;batch_index<p.batch_num;batch_index++){
		for(case_index=0;case_index<p.case_num;case_index++){
			for(int i=0;i<p.vis_num;i++){
				inputdata[batch_index][case_index][i] = i_data[batch_index][case_index][i];
				P_v0[batch_index][i] += i_data[batch_index][case_index][i]/p.case_num;
			}
		}
	}
}

//Output data
void ConditionalRBMBase::OutputData(vector<vector<vector<double>>>* o_data)
{
	int batch_index = 0;
	int case_index = 0;

	for(batch_index=0;batch_index<p.batch_num;batch_index++){
		vector<vector<double>> tmp2;
		for(case_index=0;case_index<p.case_num;case_index++){
			vector<double> tmp;
			for(int j=0;j<p.hid_num;j++){
				tmp.push_back(outputdata[batch_index][case_index][j]);
			}
			tmp2.push_back(tmp);
		}
		o_data->push_back(tmp2);
	}
}

//Training with BP
void ConditionalRBMBase::CalcRBM_BackProp()
{
	int batch_index = 0;
	int case_index = 0;

	err_all = 0.0;
	update_cnt = 0;

	//main loop
	while(update_cnt<p.maxlearn_num){

		update_cnt+=1;
		err_all = 0.0;
		
		for(batch_index=0;batch_index<p.batch_num;batch_index++){
			for(case_index=0;case_index<p.case_num;case_index++){
				prebRBM->Reconstruct(batch_index, case_index);
			}
		}

		prebRBM->ConnectRBM(this);

		for(batch_index=0;batch_index<p.batch_num;batch_index++){
			for(case_index=0;case_index<p.case_num;case_index++){
				Reconstruct(batch_index, case_index);
			}
			err_all += BackProp(batch_index);
		}

		if(update_cnt%p.log_num == 1){
			cout << "epoch : " << update_cnt << " B_ERR = " << err_all << endl;
			prebRBM->SaveLog();
			SaveLog();
		}
		if(update_cnt%(p.log_num*10) == 1){
			prebRBM->SaveWeight(update_cnt);
			SaveWeight(update_cnt);
		}
		if(err_all < p.err_th){
			break;
		}
	}
	//output result
	for(batch_index=0;batch_index<p.batch_num;batch_index++){
		for(case_index=0;case_index<p.case_num;case_index++){
			prebRBM->Reconstruct(batch_index, case_index);
		}
	}

	prebRBM->ConnectRBM(this);
	for(batch_index=0;batch_index<p.batch_num;batch_index++){
		for(case_index=0;case_index<p.case_num;case_index++){
			Reconstruct(batch_index, case_index);
		}
	}
	prebRBM->SaveWeight();
	prebRBM->SaveLog();
	SaveLog();
	SaveWeight();
}

void ConditionalRBMBase::ConnectRBM(ConditionalRBMBase* next_rbm)
{
	int batch_index = 0;
	int case_index = 0;

	next_rbm->InputData(outputdata);

	next_rbm->prebRBM = this;
}

void ConditionalRBMBase::SetLabelData(vector<vector<vector<double>>> l_data)
{
	int batch_index = 0;
	int case_index = 0;

	for(batch_index=0;batch_index<p.batch_num;batch_index++){
		for(case_index=0;case_index<p.case_num;case_index++){
			for(int j=0;j<p.hid_num;j++){
				labeldata[batch_index][case_index][j] = l_data[batch_index][case_index][j];
				P_hhat[batch_index][j] += l_data[batch_index][case_index][j]/p.case_num;
			}
		}
	}
}

int ConditionalRBMBase::GetUpdateCnt()
{
	return update_cnt;
}

//Init function
void ConditionalRBMBase::Init()
{
	update_cnt = 0;
	err_all = 0.0;

	srand((unsigned int) p.seed );

	vector<double> tmp_v;
	tmp_v.clear();
	for(int i=0;i<p.vis_num;i++){
		tmp_v.push_back(0.0);
	}
	vector<vector<double>> tmp2;
	tmp2.clear();
	for(int i=0;i<p.case_num;i++){
		tmp2.push_back(tmp_v);
	}
	inputdata.clear();
	for(int i=0;i<p.batch_num;i++){
		inputdata.push_back(tmp2);
	}

	vector<double> tmp_h;
	tmp_h.clear();
	for(int i=0;i<p.hid_num;i++){
		tmp_h.push_back(0.0);
	}
	tmp2.clear();
	for(int i=0;i<p.case_num;i++){
		tmp2.push_back(tmp_h);
	}
	outputdata.clear();
	outputprob.clear();
	for(int i=0;i<p.batch_num;i++){
		outputdata.push_back(tmp2);
		outputprob.push_back(tmp2);
	}

	tmp2.clear();
	for(int i=0;i<p.case_num;i++){
		tmp2.push_back(tmp_h);
	}
	labeldata.clear();
	for(int i=0;i<p.batch_num;i++){
		labeldata.push_back(tmp2);
	}

	P_v0.clear();
	Act_v0.clear();
	for(int i=0;i<p.batch_num;i++){
		P_v0.push_back(tmp_v);
	}
	for(int i=0;i<p.case_num;i++){
		Act_v0.push_back(tmp_v);
	}

	P_v1.clear();
	Act_v1.clear();
	for(int i=0;i<p.case_num;i++){
		P_v1.push_back(tmp_v);
		Act_v1.push_back(tmp_v);
	}

	P_h0.clear();
	Act_h0.clear();
	for(int i=0;i<p.case_num;i++){
		P_h0.push_back(tmp_h);
		Act_h0.push_back(tmp_h);
	}

	P_h1.clear();
	Act_h1.clear();
	for(int i=0;i<p.case_num;i++){
		P_h1.push_back(tmp_h);
		Act_h1.push_back(tmp_h);
	}
	
	bistar.clear();
	bjstar.clear();
	for(int i=0;i<p.case_num;i++){
		bistar.push_back(tmp_v);
		bjstar.push_back(tmp_h);
	}

	P_hhat.clear();
	for(int i=0;i<p.batch_num;i++){
		P_hhat.push_back(tmp_h);
	}

	P_re.clear();
	for(int i=0;i<p.vis_num;i++){
		P_re.push_back(0.0);
	}
	
	double alpha, beta;
	double w_rand;
	vector<double> w_tmp;
	vector<double> pre_dw_tmp;
	w.clear();
	pre_w.clear();
	pre_dw.clear();

	for(int i=0;i<p.vis_num;i++){
		w_tmp.clear();
		pre_dw_tmp.clear();
		for(int j=0;j<p.hid_num;j++){
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

	A.clear();
	B.clear();
	pre_dA.clear();
	pre_dB.clear();
	vector<vector<double>> tmp_A;
	vector<vector<double>> tmp_B;
	vector<vector<double>> tmp_O;
	for(int t=0;t<=p.preb_window_num;t++){
		tmp_A.clear();
		tmp_O.clear();
		for(int i=0;i<p.vis_num;i++){
			w_tmp.clear();
			pre_dw_tmp.clear();
			for(int j=0;j<p.vis_num;j++){
				alpha = 1.0-((double)rand()/(RAND_MAX+1.0));
				beta = 1.0-((double)rand()/(RAND_MAX+1.0));
				w_rand = 0.01*sqrt(-2.0*log(alpha))*sin(2.0*M_PI*beta);
				if(p.preb_window_num==0){
					w_rand = 0;
				}
				w_tmp.push_back(w_rand);
				pre_dw_tmp.push_back(0.0);
			}
			tmp_A.push_back(w_tmp);
			tmp_O.push_back(pre_dw_tmp);
		}
		A.push_back(tmp_A);
		pre_dA.push_back(tmp_O);

		tmp_B.clear();
		tmp_O.clear();
		for(int i=0;i<p.vis_num;i++){
			w_tmp.clear();
			pre_dw_tmp.clear();
			for(int j=0;j<p.hid_num;j++){
				alpha = 1.0-((double)rand()/(RAND_MAX+1.0));
				beta = 1.0-((double)rand()/(RAND_MAX+1.0));
				w_rand = 0.01*sqrt(-2.0*log(alpha))*sin(2.0*M_PI*beta);
				if(p.preb_window_num==0){
					w_rand = 0;
				}
				w_tmp.push_back(w_rand);
				pre_dw_tmp.push_back(0.0);
			}
			tmp_B.push_back(w_tmp);
			tmp_O.push_back(pre_dw_tmp);
		}
		B.push_back(tmp_B);
		pre_dB.push_back(tmp_O);
	}

	b.clear();
	pre_db.clear();
	for(int i=0;i<p.hid_num;i++){
		b.push_back(0.0);
		pre_db.push_back(0.0);
	}

	c.clear();
	pre_dc.clear();
	for(int i=0;i<p.vis_num;i++){
		c.push_back(0.0);
		pre_c.push_back(0.0);
		pre_dc.push_back(0.0);
	}

	sig.clear();
	for(int i=0;i<p.vis_num;i++){
		sig.push_back(1.0);
	}

	z.clear();
	pre_dz.clear();
	for(int i=0;i<p.vis_num;i++){
		z.push_back(0.0);
		pre_dz.push_back(0.0);
	}
}

double ConditionalRBMBase::BackProp(int batch_index)
{
	int case_index = 0;
	double err = 0.0;
	double delta = 0.0;

	double dw = 0.0;

	for(int i=0;i<p.vis_num;i++){
		for(int j=0;j<p.hid_num;j++){
			dw = 0.0;
			for(case_index=0;case_index<p.case_num;case_index++){
				delta = (P_h0[case_index][j] - labeldata[batch_index][case_index][j]);
				dw += (-1.0)*p.epsilon_bp*delta*P_h0[case_index][j]*(1.0-P_h0[case_index][j])*Act_v0[case_index][i];//case_num;
				err += fabs(P_h0[case_index][j] - labeldata[batch_index][case_index][j]);
			}
			w[i][j] += (dw + p.moment*pre_dw[i][j]);
			pre_dw[i][j] = dw;
		}
	}

	ConditionalRBMParam pp = prebRBM->p;
	for(int i=0;i<pp.vis_num;i++){
		for(int j=0;j<pp.hid_num;j++){
			dw = 0.0;
			for(case_index=0;case_index<pp.case_num;case_index++){
				delta = 0.0;
				for(int k=0;k<p.hid_num;k++){
					delta += (P_h0[case_index][k] - labeldata[batch_index][case_index][k])*w[j][k];
				}
				dw += (-1.0)*pp.epsilon_bp*delta*prebRBM->P_h0[case_index][j]*(1.0-prebRBM->P_h0[case_index][j])*prebRBM->Act_v0[case_index][i];//case_num;
			}
			prebRBM->w[i][j] += (dw + pp.moment*prebRBM->pre_dw[i][j]);
			prebRBM->pre_dw[i][j] = dw;
		}
	}

	return err;
}

void ConditionalRBMBase::SaveWeight()
{
	weight_file.open(p.RBM_name+"_w.csv");

	for(int i=0;i<p.vis_num;i++){
		for(int j=0;j<p.hid_num;j++){
			weight_file << w[i][j];
			if(j==(p.hid_num-1)){
				weight_file << endl;
			}
			else{
				weight_file << ",";
			}
		}
	}

	for(int j=0;j<p.hid_num;j++){
		weight_file << b[j];
		if(j==(p.hid_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}

	for(int i=0;i<p.vis_num;i++){
		weight_file << c[i];
		if(i==(p.vis_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}

	for(int i=0;i<p.vis_num;i++){
		weight_file << z[i];
		if(i==(p.vis_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}

	for(int t=0;t<=p.preb_window_num;t++){
		for(int i=0;i<p.vis_num;i++){
			for(int j=0;j<p.vis_num;j++){
				weight_file << A[t][i][j];
				if(j==(p.vis_num-1)){
					weight_file << endl;
				}
				else{
					weight_file << ",";
				}
			}
		}
	}

	for(int t=0;t<=p.preb_window_num;t++){
		for(int i=0;i<p.vis_num;i++){
			for(int j=0;j<p.hid_num;j++){
				weight_file << B[t][i][j];
				if(j==(p.hid_num-1)){
					weight_file << endl;
				}
				else{
					weight_file << ",";
				}
			}
		}
	}
	weight_file.close();

	ofstream outfile;
	outfile.open(p.RBM_name+"_outdata.csv");
	for(int batch_index=0;batch_index<p.batch_num;batch_index++){
		for(int case_index=0;case_index<p.case_num;case_index++){
			for(int j=0;j<p.hid_num;j++){
				outfile << outputdata[batch_index][case_index][j];
				if(j==(p.hid_num-1)){
					outfile << endl;
				}
				else{
					outfile << ",";
				}
			}
		}
	}
	outfile.close();

	outfile.open(p.RBM_name+"_outprob.csv");
	for(int batch_index=0;batch_index<p.batch_num;batch_index++){
		for(int case_index=0;case_index<p.case_num;case_index++){
			for(int j=0;j<p.hid_num;j++){
				outfile << outputprob[batch_index][case_index][j];
				if(j==(p.hid_num-1)){
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

void ConditionalRBMBase::SaveWeight(int num)
{

	stringstream ss;
	ss << num;
	weight_file.open(p.RBM_name+"_w"+ss.str()+".csv");

	for(int i=0;i<p.vis_num;i++){
		for(int j=0;j<p.hid_num;j++){
			weight_file << w[i][j];
			if(j==(p.hid_num-1)){
				weight_file << endl;
			}
			else{
				weight_file << ",";
			}
		}
	}

	for(int j=0;j<p.hid_num;j++){
		weight_file << b[j];
		if(j==(p.hid_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}

	for(int i=0;i<p.vis_num;i++){
		weight_file << c[i];
		if(i==(p.vis_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}

	for(int i=0;i<p.vis_num;i++){
		weight_file << z[i];
		if(i==(p.vis_num-1)){
			weight_file << endl;
		}
		else{
			weight_file << ",";
		}
	}

	for(int t=0;t<=p.preb_window_num;t++){
		for(int i=0;i<p.vis_num;i++){
			for(int j=0;j<p.vis_num;j++){
				weight_file << A[t][i][j];
				if(j==(p.vis_num-1)){
					weight_file << endl;
				}
				else{
					weight_file << ",";
				}
			}
		}
	}

	for(int t=0;t<=p.preb_window_num;t++){
		for(int i=0;i<p.vis_num;i++){
			for(int j=0;j<p.hid_num;j++){
				weight_file << B[t][i][j];
				if(j==(p.hid_num-1)){
					weight_file << endl;
				}
				else{
					weight_file << ",";
				}
			}
		}
	}
	weight_file.close();

	ofstream outfile;
	outfile.open(p.RBM_name+"_outdata"+ss.str()+".csv");
	for(int batch_index=0;batch_index<p.batch_num;batch_index++){
		for(int case_index=0;case_index<p.case_num;case_index++){
			for(int j=0;j<p.hid_num;j++){
				outfile << outputdata[batch_index][case_index][j];
				if(j==(p.hid_num-1)){
					outfile << endl;
				}
				else{
					outfile << ",";
				}
			}
		}
	}
	outfile.close();

	outfile.open(p.RBM_name+"_outprob"+ss.str()+".csv");
	for(int batch_index=0;batch_index<p.batch_num;batch_index++){
		for(int case_index=0;case_index<p.case_num;case_index++){
			for(int j=0;j<p.hid_num;j++){
				outfile << outputprob[batch_index][case_index][j];
				if(j==(p.hid_num-1)){
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

void ConditionalRBMBase::SaveLog()
{
	if(p.wlog_flag){
		weight_file_all.open(p.RBM_name+"_wlog.csv",ios::app);
		for(int i=0;i<p.vis_num;i++){
			for(int j=0;j<p.hid_num;j++){
				weight_file_all << w[i][j] << ",";
			}
		}
		weight_file_all << endl;
		weight_file_all.close();

		b_file_all.open(p.RBM_name+"_blog.csv",ios::app);
		for(int i=0;i<p.hid_num;i++){
				b_file_all << b[i] << ",";
		}
		b_file_all << endl;
		b_file_all.close();

		c_file_all.open(p.RBM_name+"_clog.csv",ios::app);
		for(int i=0;i<p.vis_num;i++){
				c_file_all << c[i] << ",";
		}
		c_file_all << endl;
		c_file_all.close();

		z_file_all.open(p.RBM_name+"_zlog.csv",ios::app);
		for(int i=0;i<p.vis_num;i++){
				z_file_all << z[i] << ",";
		}
		z_file_all << endl;
		z_file_all.close();
	}

	err_file.open(p.RBM_name+"_err.csv",ios::app);
	err_file << err_all << endl;
	err_file.close();
}

void ConditionalRBMBase::LoadWeight(string filename)
{
	char comma;
	ifstream infile;
	infile.open(filename);
	if(!infile.is_open()){
		cout << "File can't load weight data" << endl;
		return;
	}

	for(int i=0;i<p.vis_num;i++){
		for(int j=0;j<p.hid_num;j++){
			infile >> w[i][j];
			if(j!=(p.hid_num-1)){
				infile >> comma;
			}
		}
	}

	for(int j=0;j<p.hid_num;j++){
		infile >> b[j];
		if(j!=(p.hid_num-1)){
			infile >> comma;
		}
	}

	for(int i=0;i<p.vis_num;i++){
		infile >> c[i];
		if(i!=(p.vis_num-1)){
			infile >> comma;
		}
	}

	for(int i=0;i<p.vis_num;i++){
		infile >> z[i];
		if(i!=(p.vis_num-1)){
			infile >> comma;
		}
	}

	for(int t=0;t<=p.preb_window_num;t++){
		for(int i=0;i<p.vis_num;i++){
			for(int j=0;j<p.vis_num;j++){
				infile >> A[t][i][j];
				if(j!=(p.vis_num-1)){
					infile >> comma;
				}
			}
		}
	}

	for(int t=0;t<=p.preb_window_num;t++){
		for(int i=0;i<p.vis_num;i++){
			for(int j=0;j<p.hid_num;j++){
				infile >> B[t][i][j];
				if(j!=(p.hid_num-1)){
					infile >> comma;
				}
			}
		}
	}

	infile.close();
}

double ConditionalRBMBase::Sigmoid(double x)
{
	return 1.0 / (1.0 + exp(x));
}

double ConditionalRBMBase::Gaussian(double x, double mu, double sigma_2)
{
	return 1.0/sqrt(2.0*M_PI*sigma_2) * exp(-(x-mu)*(x-mu)/(2.0*sigma_2));
}

double ConditionalRBMBase::Gauss_Rand(double mu, double sigma_2)
{
	double alpha = 1.0-((double)rand()/(RAND_MAX+1.0));
	double beta = 1.0-((double)rand()/(RAND_MAX+1.0));
	return sqrt(-2.0*log(alpha))*sin(2.0*M_PI*beta) * sqrt(sigma_2) + mu;
}
