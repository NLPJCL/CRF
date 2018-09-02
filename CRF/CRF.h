#pragma once
#include<iostream>
#include<map>
#include<vector>
#include<cmath>//��Ȼ������ָ����
#include <algorithm>  
#include"dataset.h"
#include "time.h"
#include<numeric>
using namespace std;
class CRF
{
public:
	CRF();
	~CRF();
public:
	void create_feature_space();
	void sgd_online_training();
	//�洢��
	void save_file(int i);
private:
	//�������ݼ���
	dataset train;
	dataset dev;
	map<string, int> model;//�����ռ䡣
	map<string, int> tag;//����
	map<int, double> g;
	vector<double> w;//һά������Ȩ�ء�
	vector<string> vector_tag;
	//vector<string> feature;//
						   //���������ռ䡣
	vector<string> create_feature(const sentence &sentence, int pos);
	//�����㷨
	void update_w();
	void updata_g(const sentence &sen);
	vector<string> max_score_sentence_tag(const sentence &sen);
	vector<int> get_id(const vector<string> &f);
	vector<double> count_score(const vector<string> &feature);

	//ǰ���㷨
	vector<vector<double>> forword(const sentence&);
	vector<vector<double>> backword(const sentence&);
	vector<vector<double>> head_prob;
	//���ۡ�
	double evaluate(dataset);
};
vector<double> logsumexp(vector<vector<double>> a);
vector<vector<double>> translation(vector<vector<double>> a);
