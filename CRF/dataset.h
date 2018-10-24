#pragma once
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include"sentence.h"
using namespace std;
class dataset
{
public:
	vector<sentence> sentences;
	string name;
	void read_data(const string &file_name);
	int sentence_count = 0, word_count = 0;
	dataset();
	~dataset();
};

