#include<iostream>
#include"dataset.h"
#include"CRF.h"
using namespace std;
int main()
{
	CRF b;
	b.create_feature_space();
	b.sgd_online_training();
	system("pause");
	return 0;
}