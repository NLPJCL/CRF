#pragma once
#include<iostream>
#include<map>
#include<vector>
#include<cmath>//自然对数的指数。
#include <algorithm>  
#include<numeric>
#include"dataset.h"
#include "time.h"
#include"windows.h"
#include<unordered_map>
using namespace std;
class CRF
{
public:
	CRF(string train_, string dev_, string test_);
	~CRF();
public:
	void create_feature_space();
	void sgd_online_training(bool shuffle, int iterator, int exitor);
	//存储。
	void save_file();
private:
	//基础数据集。
	dataset train;
	dataset dev;
	dataset test;
	unordered_map<string, int> model;//特征空间。
	unordered_map<string, int> tag;//词性
	unordered_map<int, double> g;
	//map<int, double> g;
	//vector<double> g;
	vector<double> w;//一维的特征权重。
	vector<string> feature;
	vector<string> vector_tag;
	string create_one_feature(const sentence &sen, int pos);
	vector<string> create_feature(const sentence &sentence, int pos);
	//在线算法
	void update_w(double eta);
	void updata_g(const sentence &sen);
	vector<string> max_score_sentence_tag(const sentence &sen);
	vector<int> get_id(const vector<string> &f);
	vector<double> count_score(const vector<string> &feature);

	//前后算法
	vector<vector<double>> forword(const sentence&);
	vector<vector<double>> backword(const sentence&);
	vector<vector<double>> head_prob;
	//评价。
	double evaluate(dataset);
};
vector<double> logsumexp(const vector<vector<double>> &a);
vector<vector<double>> translation(vector<vector<double>> &a);
