#pragma once
#include<iostream>
#include<map>
#include<vector>
#include<cmath>//��Ȼ������ָ����
#include <algorithm>  
#include"dataset.h"
#include "time.h"
#include <Eigen/Dense>
#include<numeric>
using namespace Eigen;
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
	vector<string> feature;//
						   //���������ռ䡣
	vector<string> create_feature(const sentence &sentence, int pos);
	//�����㷨
	void update_w();
	void updata_g(const sentence &sen);
	void max_score_sentence_tag(const sentence &sen, vector<string> &path_max);
	vector<int> get_id(const vector<string> &f);
	int count_score(int offset, const vector<int> &fv);
	//ǰ���㷨
	vector<vector<double>> alpha, beta;
	vector<vector<double>> forword(const sentence&);
	vector<vector<double>> backword(const sentence&);

	//���ۡ�
	double evaluate(dataset);
};

