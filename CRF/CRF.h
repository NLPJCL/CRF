#pragma once
#include<iostream>
#include<map>
#include<vector>
#include<cmath>//自然对数的指数。
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
	//存储。
	void save_file(int i);
private:
	//基础数据集。
	dataset train;
	dataset dev;
	dataset test;
	map<string, int> model;//特征空间。
	map<string, int> tag;//词性
	map<int, long double> g;
	vector<long double> w;//一维的特征权重。
	vector<string> vector_tag;
	//vector<string> feature;//
						   //创建特征空间。
	vector<string> create_feature(const sentence &sentence, int pos);
	//在线算法
	void update_w();
	void updata_g(const sentence &sen);
	vector<string> max_score_sentence_tag(const sentence &sen);
	vector<int> get_id(const vector<string> &f);
	vector<long double> count_score(const vector<string> &feature);

	//前后算法
	vector<vector<long double>> forword(const sentence&);
	vector<vector<long double>> backword(const sentence&);
	vector<vector<long double>> head_prob;
	//评价。
	double evaluate(dataset);
};
vector<long double> logsumexp(vector<vector<long double>> a);
vector<vector<long double>> translation(vector<vector<long double>> &a);
