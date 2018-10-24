#pragma once
#include<iostream>
#include<map>
#include<vector>
#include<cmath>//��Ȼ������ָ����
#include <algorithm>  
#include<numeric>
#include"dataset.h"
#include "time.h"
#include"windows.h"
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
	void save_file();
private:
	//�������ݼ���
	dataset train;
	dataset dev;
	dataset test;
	map<string, int> model;//�����ռ䡣
	map<string, int> tag;//����
	map<int, double> g;
	vector<double> w;//һά������Ȩ�ء�
	vector<string> feature;
	vector<string> vector_tag;
	string create_one_feature(const sentence &sen, int pos);
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
vector<double> logsumexp(vector<vector<double>> &a);
vector<vector<double>> translation(vector<vector<double>> &a);
