#include "CRF.h"
vector<double> logsumexp( vector<vector<double>> &a)
{
	vector<double> new_vector(a.size());
	double max, all;
	for (int i = 0; i <a.size(); i++)
	{
		auto w = max_element(begin(a[i]), end(a[i]));
		max = *w;
		all = 0.0;
		for (int j = 0; j < a[i].size(); j++)
		{
			a[i][j] = exp(a[i][j] - max);
		}
		all = log(accumulate(a[i].begin(), a[i].end(), 0.0));
		new_vector[i] = max + all;
	}
	return new_vector;
}
double logsumexp(const vector<double> &a)
{

		double new_double;
		auto w = max_element(begin(a), end(a));
		vector<double> new_vector(a.size());
		double max = *w,all = 0.0;
		vector<double> b(a.size());
		for (int j = 0; j < a.size(); j++)
		{
			b[j] = exp(a[j] - max);
		}
		all = log(accumulate(b.begin(), b.end(), 0.0));
		new_double = max + all;
		return new_double;
}
vector<vector<double>> translation(vector<vector<double>>& a)
{
	vector<vector<double>> b(a.size(), vector<double>(a.size(), 0));
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < a[i].size(); j++)
		{
			b[i][j] = a[j][i];
		}
	}
	return b;
}
CRF::CRF(string train_, string dev_, string test_)
{
	if (train_ != "")
	{
		train.read_data(train_);
	}
	if (dev_ != "")
	{
		dev.read_data(dev_);
	}
	if (test_ != "")
	{
		test.read_data(test_);
	}
}
CRF::~CRF()
{
}
vector<string> CRF::create_feature(const sentence &sentence, int pos)
{
	string word = sentence.word[pos];//当前词。
	string word_char_first = sentence.word_char[pos][0];//当前词的第一个元素。
	string word_char_last = sentence.word_char[pos][sentence.word_char[pos].size() - 1];//当前词的最后一个元素。
	string word_m1;
	string word_char_m1m1;
	string word_p1;
	string word_char_p1_first;
	int word_count = sentence.word.size();//当前句的总词数。
	if (pos == 0)
	{
		word_m1 = "$$";
		word_char_m1m1 = "$";
	}
	else
	{
		word_m1 = sentence.word[pos - 1];
		word_char_m1m1 = sentence.word_char[pos - 1][(sentence.word_char[pos - 1].size() - 1)];
	}
	if (pos == word_count - 1)
	{
		word_p1 = "##";
		word_char_p1_first = "#";
	}
	else
	{
		word_p1 = sentence.word[pos + 1];
		word_char_p1_first = sentence.word_char[pos + 1][0];
	}
	vector<string> f;
	f.reserve(50);
	f.emplace_back("02:*" + word);
	f.emplace_back("03:*" + word_m1);
	f.emplace_back("04:*" + word_p1);
	f.emplace_back("05:*" + word + "*" + word_char_m1m1);
	f.emplace_back("06:*" + word + "*" + word_char_p1_first);
	f.emplace_back("07:*" + word_char_first);
	f.emplace_back("08:*" + word_char_last);
	int pos_word_len = sentence.word_char[pos].size();
	for (int k = 1; k < pos_word_len - 1; k++)
	{
		string cik = sentence.word_char[pos][k];
		f.emplace_back("09:*" + cik);
		f.emplace_back("10:*" + word_char_first + "*" + cik);
		f.emplace_back("11:*" + word_char_last + "*" + cik);
		string cikp1 = sentence.word_char[pos][k + 1];
		if (cik == cikp1)
		{
			f.emplace_back("13:*" + cik + "*" + "consecutive");
		}
	}
	if (sentence.word_char[pos].size() > 1)
	{
		if (sentence.word_char[pos][0] == sentence.word_char[pos][1])
			f.emplace_back("13:*" + sentence.word_char[pos][0] + "*" + "consecutive");
	}
	if (pos_word_len == 1)
	{
		f.emplace_back("12:*" + word + "*" + word_char_m1m1 + "*" + word_char_p1_first);
	}
	for (int k = 0; k <pos_word_len; k++)
	{
		if (k >= 4)break;
		f.emplace_back("14:*" + accumulate(sentence.word_char[pos].begin(), sentence.word_char[pos].begin() + k + 1, string("")));
		f.emplace_back("15:*" + accumulate(sentence.word_char[pos].end() - (k + 1), sentence.word_char[pos].end(), string("")));
	}
	return f;
}
string CRF::create_one_feature(const sentence &sen,int pos)
{
	if (pos == 0)
	{
		return ("01:*&&");
	}
	else
	{
		return ("01:*" + sen.tag[pos-1]);
	}
}
void CRF::create_feature_space()
{
	int word_count = 0, tag_count = 0;
	vector<string> f;
	for (auto sen = train.sentences.begin(); sen != train.sentences.end(); sen++)
	{
		for (int i = 0; i < sen->word.size(); i++)
		{
			f = create_feature(*sen, i);
			f.emplace_back(create_one_feature(*sen, i));
			for (auto i = f.begin(); i != f.end(); i++)
			{
				if (model.find(*i) == model.end())
				{
					feature.emplace_back(*i);
					model.insert({ *i , word_count });
					word_count++;
				}
			}
			if (tag.find(sen->tag[i]) == tag.end())
			{
				tag.insert({ sen->tag[i], tag_count });
				vector_tag.emplace_back(sen->tag[i]);
				tag_count++;
			}
		}
	}
	w.reserve(tag.size()*model.size());
	for (int j = 0; j < tag.size()*model.size(); j++)
	{
		w.emplace_back(0.0);
	}
	vector<vector<double>> c(tag.size(), vector<double>(tag.size(), 0));
	head_prob = c;
	cout << "the total number of features is " << model.size() << endl;
	cout << "the total number of tags is " << tag.size() << endl;
}
vector<int> CRF::get_id(const vector<string> &f)
{
	vector<int> fv;
	for (auto q = f.begin(); q != f.end(); q++)
	{
		auto t = model.find(*q);
		if (t != model.end())
		{
			fv.emplace_back(t->second);
		}
	}
	return fv;
}
vector<double> CRF::count_score(const vector<string> &feature)
{
	int offset = 0;
	//vector<vector<double>> scores(tag.size(), vector<double>(feature.size(), 0));
	vector<double> score(tag.size());
	for (int j = 0; j < tag.size(); j++)
	{
		offset = j * model.size();
		for (int i = 0; i < feature.size(); i++)
		{
			if (model.find(feature[i]) != model.end())
			{
				//scores[j][i] = w[model[feature[i]] + offset];
				score[j] += w[model[feature[i]] + offset];
			}
		}
	}
	/*
	vector<double> score(tag.size());
	for (int j = 0; j<tag.size(); j++)
	{
		score[j] = accumulate(scores[j].begin(), scores[j].end(), 0);
	}
	*/
	return score;
}
vector<vector<double>> CRF::forword(const sentence &sen)
{
	vector<vector<double>> scores(sen.word.size(), vector<double>(tag.size(), 0));
	//第一个词
	vector<string> first_feature = create_feature(sen, 0);
	first_feature.emplace_back(create_one_feature(sen,0));
	scores[0]=count_score(first_feature);
	//其余的词。
	for (int z = 1; z < sen.word.size(); z++)
	{
		vector<string> curr_feature = create_feature(sen, z);
		vector<double> curr_score = count_score(curr_feature);
		vector<vector<double>> p(tag.size(), vector<double>(tag.size(), 0));
		for (int i = 0; i < tag.size(); i++)
		{
			for (int j = 0; j < tag.size(); j++)
			{
				p[i][j] = head_prob[i][j] + curr_score[i];
			}
		}
		for (int i = 0; i < tag.size(); i++)
		{
			for (int j = 0; j < tag.size(); j++)
			{
				p[j][i] += scores[z - 1][i];
			}
		}
		scores[z] = logsumexp(p);
	}
	return scores;
}
vector<vector<double>> CRF::backword(const sentence &sen)
{
	//方便计算。
	vector<vector<double>> scores(sen.tag.size(), vector<double>(tag.size(), 0));
	//其余的词。	
	vector<vector<double>> q = translation(head_prob);
	for (int z = sen.word.size()-2; z >=0; z--)
	{
		vector<string> curr_feature = create_feature(sen, z+1);
		vector<double> curr_score = count_score(curr_feature);
		vector<vector<double>> p(tag.size(), vector<double>(tag.size(), 0));
		for (int i = 0; i < tag.size(); i++)
		{
			for (int j = 0; j < tag.size(); j++)
			{
				p[j][i] = q[j][i] + curr_score[i]+scores[z+1][i];
			}
		}
		scores[z] = logsumexp(p);
	}
	return scores;
}
void CRF::update_w(double eta)
{
	for (auto z = g.begin(); z != g.end(); ++z)
	{
		w[z->first] += eta*z->second;
	}
	//cout << g.size();
	/*
	for (int i = 0; i < g.size(); i++)
	{
		w[i] += g[i];
	}
	*/
}
void CRF::updata_g(const sentence & sen)
{
	vector<vector<double>> alpha = forword(sen);
	vector<vector<double>> beta = backword(sen);
	//计算分母。
	double z = logsumexp(alpha[sen.word.size() - 1]);
//	double q = logsumexp(beta[0]);
//	cout << z << endl;
//	cout << q << endl;
	//对正确的序列进行处理，
	for (int i = 0; i < sen.word.size(); i++)
	{
		vector<string> first_feature = create_feature(sen, i);
		first_feature.emplace_back(create_one_feature(sen,i));
		vector<int> fv_id = get_id(first_feature);
		int offset = tag[sen.tag[i]] * model.size();
		for (int q = 0; q < fv_id.size(); q++)
		{
			g[fv_id[q] + offset]++;
		}
	}
	//对第一个词进行处理。
	vector<string> first_feature = create_feature(sen, 0);
	first_feature.emplace_back(create_one_feature(sen,0));
	vector<int> fv_id = get_id(first_feature);
	vector<double> first_score = count_score(first_feature);
	for (int i = 0; i < tag.size(); i++)
	{
		int offset = i * model.size();
		double p = exp(beta[0][i] +first_score[i]-z);
		for (int j = 0; j < fv_id.size(); j++)
		{
			g[offset+fv_id[j]] -= p;
		}
	}
	//对所有有可能的系列进行处理。
	for (int i = 1; i < sen.word.size(); i++)
	{
		vector<string> curr_feature = create_feature(sen, i);
		vector<double> curr_score = count_score(curr_feature);
		vector<int> fv_id = get_id(curr_feature);
		vector<vector<double>> p(tag.size(), vector<double>(tag.size(), 0));
		for (int q = 0; q < tag.size(); q++)
		{
			for (int k = 0; k < tag.size(); k++)
			{
				p[q][k] = exp(alpha[i - 1][k] + beta[i][q] + head_prob[q][k]+curr_score[q] - z);
				g[model["01:*" + vector_tag[k]] + q * model.size()] -= p[q][k];
				for (int w = 0; w < fv_id.size(); w++)
				{
					g[q*model.size() + fv_id[w]] -= p[q][k];
				}
			}
		}
	}
}
vector<string> CRF::max_score_sentence_tag(const sentence &sen)
{
	vector<vector<double>> score_prob(sen.tag.size(), vector<double>(tag.size(), 0));
	vector<vector<double>> score_path(sen.tag.size(), vector<double>(tag.size(), 0));
	for (int i = 0; i < tag.size(); i++)
	{
		score_path[0][i] = -1;
	}
	//第一个词。
	vector <string> feature = create_feature(sen, 0);
	feature.emplace_back("01:*&&");
	score_prob[0] = count_score(feature);
	//其余的词。
	for (int j = 1; j < sen.word.size(); j++)
	{

		vector<int> currect_prob(tag.size(), 0);
		vector <string> feature = create_feature(sen, j);
		vector<double> score = count_score(feature);
		for (int i = 0; i < tag.size(); i++)
		{
			for (int z = 0; z < tag.size(); z++)
			{
				currect_prob[z] = head_prob[i][z] + score_prob[j - 1][z];
			}
			auto w = max_element(currect_prob.begin(), currect_prob.end());
			score_prob[j][i] = *w + score[i];
			score_path[j][i] = distance(currect_prob.begin(), w);
		}
	}
	vector<string> path_max;
	vector<double> a = score_prob[sen.word.size() - 1];
	auto w = max_element(begin(a), end(a));
	int max_point = distance(begin(a), w);
	for (int j = sen.word.size() - 1; j > 0; j--)
	{
		path_max.emplace_back(vector_tag[max_point]);
		max_point = score_path[j][max_point];
	}
	path_max.emplace_back(vector_tag[max_point]);
	reverse(path_max.begin(), path_max.end());
	return path_max;
}
double CRF::evaluate(dataset data)
{
	int correct = 0, all = 0;
	for (auto sen = data.sentences.begin(); sen != data.sentences.end(); sen++)
	{
		vector<string> max_sentence_tag=max_score_sentence_tag(*sen);
		for (int i = 0; i < max_sentence_tag.size(); i++)
		{
			if (max_sentence_tag[i] == sen->tag[i])
			{
				correct++;
			}
			all++;
		}
	}
	return (correct / double(all));
}
void CRF::sgd_online_training(bool shuffle, int iterator, int exitor)
{
	string file_name;
	if (test.name.size() != 0)
	{
		file_name = "big_result_";
	}
	else
	{
		file_name = "small_result_";
	}
	if (shuffle)
	{
		file_name += "shuffle_";
	}
	file_name += ".txt";

	ofstream result(file_name);
	if (!result)
	{
		cout << "don't open  file" << endl;
	}
	result << train.name << "共" << train.sentence_count << "个句子，共" << train.word_count << "个词" << endl;
	result << dev.name << "共" << dev.sentence_count << "个句子，共" << dev.word_count << "个词" << endl;
	if (test.name.size() != 0)
	{
		result << test.name << "共" << test.sentence_count << "个句子，共" << test.word_count << "个词" << endl;
	}
	result << " the total number of features is " << model.size() << endl;
	int count = 0;
	int global_step = 30;
	int b = 0;
	double eta = 0.1;
	double max_train_precision = 0;
	int max_train_iterator = 0;

	double max_dev_precision = 0;
	int max_dev_iterator = 0;

	double max_test_precision = 0;
	int max_test_iterator = 0;
	cout << "using W to predict dev data..." << endl;
	result << "using w to predict dev data..." << endl;
	DWORD t1, t2, t3, t4;
	t1 = timeGetTime();

	for (int j = 0; j < 20; j++)
	{
		cout << "iterator" << j << endl;
		result << "iterator " << j << endl;
		t3 = timeGetTime();
		if (shuffle)
		{
			cout << "正在打乱数据" << endl;
			train.shuffle();
		}
		for (auto sen = train.sentences.begin(); sen != train.sentences.end(); ++sen)
		{
			b++;
			updata_g(*sen);
			if (b == global_step)
			{
				update_w(eta);
				b = 0;
				//fill(g.begin(), g.end(), 0.0);
				g.clear();
			}
			for (int i = 0; i < tag.size(); i++)
			{
				for (int j = 0; j < tag.size(); j++)
				{
					int max_tag_id = model["01:*" + vector_tag[j]];
					head_prob[i][j] = w[i*model.size() + max_tag_id];
				}
			}
		}
		double train_precision = evaluate(train);
		cout << train.name << "=" << train_precision << endl;
		result << train.name << "=" << train_precision << endl;

		double dev_precision = evaluate(dev);
		cout << dev.name << "=" << dev_precision << endl;
		result << dev.name << "=" << dev_precision << endl;

		if (train_precision > max_train_precision)
		{
			max_train_precision = train_precision;
			max_train_iterator = j;
		}
		if (dev_precision > max_dev_precision)
		{
			max_dev_precision = dev_precision;
			max_dev_iterator = j;
			count = 0;
		}
		else
		{
			count++;
		}
		if (test.name.size() != 0)
		{
			double test_precision = evaluate(test);
			cout << test.name << "=" << test_precision << endl;
			result << test.name << "=" << test_precision << endl;
			if (test_precision > max_test_precision)
			{
				max_test_precision = test_precision;
				max_test_iterator = j;
			}
		}

		t4 = timeGetTime();
		cout << "Use Time:" << (t4 - t3)*1.0 / 1000 << endl;
		result << "Use Time:" << (t4 - t3)*1.0 / 1000 << endl;
		if (count >= exitor)
		{
			break;
		}
	}
	cout << train.name + "最大值是：" << "=" << max_train_precision << "在" << max_train_iterator << "次" << endl;
	cout << dev.name + "最大值是：" << "=" << max_dev_precision << "在" << max_dev_iterator << "次" << endl;
	result << train.name + "最大值是：" << "=" << max_train_precision << "在" << max_train_iterator << "次" << endl;
	result << dev.name + "最大值是：" << "=" << max_dev_precision << "在" << max_dev_iterator << "次" << endl;
	if (test.name.size() != 0)
	{
		cout << test.name + "最大值是：" << "=" << max_test_precision << "在" << max_test_iterator << "次" << endl;
		result << test.name + "最大值是：" << "=" << max_test_precision << "在" << max_test_iterator << "次" << endl;
	}
	t2 = timeGetTime();
	cout << "Use Time:" << (t2 - t1)*1.0 / 1000 << endl;
	result << "Use Time:" << (t2 - t1)*1.0 / 1000 << endl;
}
void CRF::save_file()
{
	ofstream feature_file("feature.txt");
	feature_file << "词性的总数为： " << vector_tag.size() <<" "<<endl;
	feature_file << "词性分别为： ";
	for (int i = 0; i < vector_tag.size(); i++)
	{
		feature_file << vector_tag[i] << " ";
	}
	feature_file << endl;
	feature_file << "特征的大小为： " << model.size() <<" "<<endl;
	int model_size = model.size();
	for (int j=0; j<feature.size(); j++)
	{
		for (int i = 0; i < vector_tag.size(); i++)
		{
			int index= model_size * i+j;
			feature_file << feature[j] << " " << vector_tag[i]<<" "<< w[index] << endl;
		}
	}
}
