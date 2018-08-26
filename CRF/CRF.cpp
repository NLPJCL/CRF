#include "CRF.h"



CRF::CRF()
{
}


CRF::~CRF()
{
}
vector<string> CRF::create_feature(const sentence &sentence, int pos)
{
	string word = sentence.word[pos];//当前词。
	string former_tag;
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
	f.push_back("02:*" + word);
	f.push_back("03:*" + word_m1);
	f.push_back("04:*" + word_p1);
	f.push_back("05:*" + word + "*" + word_char_m1m1);
	f.push_back("06:*" + word + "*" + word_char_p1_first);
	f.push_back("07:*" + word_char_first);
	f.push_back("08:*" + word_char_last);
	int pos_word_len = sentence.word_char[pos].size();
	for (int k = 0; k < pos_word_len - 1; k++)
	{
		string cik = sentence.word_char[pos][k];
		f.push_back("09:*" + cik);
		f.push_back("10:*" + word_char_first + "*" + cik);
		f.push_back("11:*" + word_char_last + "*" + cik);
		string cikp1 = sentence.word_char[pos][k + 1];
		if (cik == cikp1)
		{
			f.push_back("13:*" + cik + "*" + "consecutive");
		}
	}
	if (pos_word_len == 1)
	{
		f.push_back("12:*" + word + "*" + word_char_m1m1 + "*" + word_char_p1_first);
	}
	for (int k = 0; k <pos_word_len; k++)
	{
		if (k >= 4)break;
		string prefix, suffix;
		//获取前缀
		for (int n = 0; n <= k; n++)
		{
			prefix = prefix + sentence.word_char[pos][n];
		}
		//获取后缀。
		for (int n = pos_word_len - k - 1; n <= pos_word_len - 1; n++)
		{
			suffix = suffix + sentence.word_char[pos][n];
		}
		f.push_back("14:*" + prefix);
		f.push_back("15:*" + suffix);
	}
	return f;
}
void CRF::create_feature_space()
{
	train.read_data("train");
	dev.read_data("dev");
	int word_count = 0, tag_count = 0;
	vector<string> f;
	for (auto sen = train.sentences.begin(); sen != train.sentences.end(); sen++)
	{
		for (int i = 0; i < sen->word.size(); i++)
		{
			if (i == 0)
			{
				f = create_feature(*sen, i);
				f.push_back("01:*&&");
			}
			else
			{
				f = create_feature(*sen, i);
				f.push_back("01:*" + sen->tag[i - 1]);
			}
			for (auto i = f.begin(); i != f.end(); i++)
			{
				if (model.find(*i) == model.end())
				{
					model[*i] = word_count;
					feature.push_back(*i);
					word_count++;
				}
			}
			f.clear();
			if (tag.find(sen->tag[i]) == tag.end())
			{
				tag[sen->tag[i]] = tag_count;
				vector_tag.push_back(sen->tag[i]);
				tag_count++;
			}
		}
	}
	w.reserve(tag.size()*model.size());
	for (int j = 0; j < tag.size()*model.size(); j++)
	{
		w.push_back(0);
	}
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
			fv.push_back(t->second);
		}
	}
	return fv;
}
int CRF::count_score(int offset, const vector<int> &fv)
{
	double score = 0;
	for (auto z0 = fv.begin(); z0 != fv.end(); z0++)
	{
		score = score + w[offset + *z0];
	}
	return score;
}
vector<vector<double>> CRF::forword(const sentence &sen)
{
	vector<vector<double>> scores(sen.word.size(), vector<double>(tag.size(), 0));
	//第一个词
	vector<string> first_feature = create_feature(sen, 0);
	first_feature.push_back("01:*&&");
	vector<int> fv_id = get_id(first_feature);
	for (int i = 0; i < vector_tag.size(); i++)
	{
		int offset = i*model.size();
		scores[0][i] = exp(count_score(offset, fv_id));
	}
	//其余的词。
	for (int z = 1; z < sen.word.size(); z++)
	{
		vector<string> curr_feature = create_feature(sen, z);
		curr_feature.push_back("01:*" + sen.tag[z - 1]);
		vector<int> fv_id = get_id(curr_feature);
		for (int i = 0; i < vector_tag.size(); i++)
		{
			int offset = i * model.size();
			double curr_score=exp(count_score(offset, fv_id));
			double all_score = 1;
			for (int j = 0; j < vector_tag.size(); j++)
			{
				all_score+= scores[z-1][j]*curr_score;
			}
			scores[z][i] = all_score;
		}
	}
	return scores;
}
vector<vector<double>> CRF::backword(const sentence &sen)
{
	//方便计算。
	vector<vector<double>> scores(sen.tag.size(), vector<double>(tag.size(), 0));
	//最后一个词
	vector<string> first_feature = create_feature(sen, sen.tag.size()-1);
	first_feature.push_back("01:*" + sen.tag[sen.tag.size() - 2]);
	vector<int> fv_id = get_id(first_feature);
	for (int i = 0; i < vector_tag.size(); i++)
	{
		int offset = i * model.size();
		scores[sen.tag.size()-1][i] = exp(count_score(offset, fv_id));
	}
	//其余的词。
	for (int z = sen.word.size()-2; z >=0; z--)
	{
		vector<string> curr_feature = create_feature(sen, z);
		if (z != 0)
		{
			curr_feature.push_back("01:*" + sen.tag[z - 1]);
		}
		else
		{
			curr_feature.push_back("01:*&&");

		}
		vector<int> fv_id = get_id(curr_feature);
		for (int i = 0; i < vector_tag.size(); i++)
		{
			int offset = i * model.size();
			double curr_score = exp(count_score(offset, fv_id));
			double all_score = 0;
			for (int j = 0; j < vector_tag.size(); j++)
			{
				all_score+= scores[z +1][i] * curr_score;
			}
			scores[z][i] = all_score;
		}
	}
	return scores;
}
void CRF::update_w()
{
	for (auto z = g.begin(); z != g.end(); z++)
	{
		w[z->first] += z->second;
	}
}
void CRF::updata_g(const sentence & sen)
{
	alpha = forword(sen);
	beta = backword(sen);
	//计算分母。
	double z = accumulate(alpha[sen.tag.size() - 1].begin(), alpha[sen.tag.size() - 1].end(), 0.0);
	//对第一个词进行处理。
	vector<string> first_feature = create_feature(sen, 0);
	first_feature.push_back("01:*&&");
	vector<int> fv_id = get_id(feature);
	int offset = tag[sen.tag[0]] * model.size();
	for (int q = 0; q < fv_id.size(); q++)
	{
		g[fv_id[q] + offset]++;
	}
	//减去第一个词错误的概率。
	double score = count_score(offset, fv_id);
	for (auto b = tag.begin(); b != tag.end(); b++)
	{
		int offset = tag[b->first] * model.size();
		double p = (beta[0][tag[b->first]] * exp(score)) / z;
		for (int q = 0; q < fv_id.size(); q++)
		{
			g[fv_id[q] + offset] -= p;
		}
	}
	//给其余的加权重。
	for (int i = 1; i < sen.word.size(); i++)
	{
		vector<string> curr_feature = create_feature(sen, i);
		curr_feature.push_back("01:*&&" + sen.tag[i - 1]);
		vector<int> fv_id = get_id(feature);
		int offset = tag[sen.tag[i]] * model.size();
		for (int j = 0; j < fv_id.size(); j++)
		{
			g[fv_id[j] + offset]++;
		}
		for (auto b = tag.begin(); b != tag.end(); b++)
		{
			int offset = b->second*model.size();
			double score = count_score(offset, fv_id);
			double p= (alpha[i - 1][tag[sen.tag[i - 1]]] * beta[i][tag[sen.tag[i]]] * exp(score) / z);
			for (int j = 0; j < fv_id.size(); j++)
			{
				g[fv_id[j] + offset] -= p;
			}
		}
	}
	int z = 0;
	int q = 0;
}
void CRF::max_score_sentence_tag(const sentence &sen, vector<string> &path_max)
{
	/*
	vector<vector<int>> score_prob (sen.tag.size(), vector<int>(tag.size(), 0));
	vector<vector<int>> score_path(sen.tag.size(), vector<int>(tag.size(), 0));
	for (int i = 0; i < tag.size(); i++)
	{
	score_path[0][i] = -1;
	}
	*/
	MatrixXi score_prob = MatrixXi::Zero(tag.size(), sen.tag.size());
	MatrixXi score_path = MatrixXi::Zero(tag.size(), sen.tag.size());
	//第一个词。
	vector <string> feature = create_feature(sen, 0);
	feature.push_back("01:*&&");
	vector<int> fv_id = get_id(feature);
	for (int i = 0; i <vector_tag.size(); i++)
	{
		int offset = i * model.size();
		int score = count_score(offset, fv_id);
		score_prob(i, 0) = score;
	}
	//其余的词。
	for (int j = 1; j < sen.word.size(); j++)
	{
		for (int curr_tag = 0; curr_tag <vector_tag.size(); curr_tag++)
		{
			//vector<int> currect_prob(tag.size(), 0);
			VectorXd currect_prob = RowVectorXd::Zero(tag.size());
			vector <string> feature = create_feature(sen, j);
			vector<int> fv_id = get_id(feature);
			int offset = curr_tag * model.size();
			int score = count_score(offset, fv_id);
			for (int pre_tag = 0; pre_tag <vector_tag.size(); pre_tag++)
			{
				int max_tag_id = model["01:*" + vector_tag[pre_tag]];
				score += w[max_tag_id + offset];
				currect_prob[pre_tag] = score + score_prob(pre_tag, j - 1);
			}
			//auto w = max_element(begin(currect_prob), end(currect_prob));
			int max_index;
			score_prob(curr_tag, j) = currect_prob.maxCoeff(&max_index);
			score_path(curr_tag, j) = max_index;
		}
	}
	/*
	//其余的词。
	for (int j = 1; j < sen.word.size(); j++)
	{
	for (int curr_tag = 0; curr_tag <vector_tag.size(); curr_tag++)
	{
	vector<int> currect_prob(tag.size(), 0);
	vector <string> feature = create_feature(sen, 0, vector_tag[pre_tag]);
	for (int pre_tag = 0; pre_tag <vector_tag.size(); pre_tag++)
	{
	vector <string> feature = create_feature(sen, 0, vector_tag[pre_tag]);
	vector<int> fv_id = get_id(feature);
	int offset = curr_tag*model.size();
	int score = count_score(offset, fv_id);
	currect_prob[curr_tag] = score + score_prob[j - 1][pre_tag];
	}
	auto w = max_element(begin(currect_prob), end(currect_prob));
	score_prob[j][curr_tag] = *w;
	score_path[j][curr_tag] = distance(begin(currect_prob), w);
	}
	}
	*/
	//vector<string> path_max;
	/*
	vector<int> a = score_prob[sen.word.size() - 1];
	auto w = max_element(begin(a), end(a));
	int max_point = distance(begin(a),w);
	*/
	VectorXi a = score_prob.col(score_prob.cols() - 1);
	int max_point;
	a.maxCoeff(&max_point);
	for (int j = sen.word.size() - 1; j > 0; j--)
	{
		path_max.push_back(vector_tag[max_point]);
		max_point = score_path(max_point, j);
	}
	path_max.push_back(vector_tag[max_point]);
	reverse(path_max.begin(), path_max.end());
	//	return path_max;
}
double CRF::evaluate(dataset data)
{
	int correct = 0, all = 0;
	for (auto sen = data.sentences.begin(); sen != data.sentences.end(); sen++)
	{
		vector<string> max_sentence_tag;
		max_score_sentence_tag(*sen, max_sentence_tag);
		for (int i = 0; i < max_sentence_tag.size(); i++)
		{
			if (max_sentence_tag[i] == sen->tag[i])
			{
				correct++;
			}
			all++;
		}
	}
	cout << data.name << ":" << correct << "/" << all << "=" << (correct / double(all)) << endl;
	return (correct / double(all));
}

void CRF::sgd_online_training()
{
	double max_train_precision = 0, max_dev_precision = 0;
	double start, stop, durationTime;
	int global_step = 1;
	int b = 0;
	for (int j = 0; j<20; j++)
	{
		start = clock();
		cout << "iterator" << j << endl;
		int i = 0;
		for (auto sen = train.sentences.begin(); sen != train.sentences.end(); sen++)
		{
			b++;
			i++;
			cout << i << endl;
			updata_g(*sen);
			if (b == global_step)
			{
				update_w();
				b = 0;
				g.clear();
			}

		}
		stop = clock();
		durationTime = ((double)(stop - start)) / CLK_TCK;
		cout << "第" << j << "次迭代总耗时：" << durationTime << endl << endl;
		//save_file(j);
		double train_precision = evaluate(train);
		double dev_precision = evaluate(dev);
		if (train_precision > max_train_precision)
		{
			max_train_precision = train_precision;
		}
		if (dev_precision > max_dev_precision)
		{
			max_dev_precision = dev_precision;
		}

	}
	cout << train.name << "=" << max_train_precision << endl;
	cout << dev.name << "=" << max_dev_precision << endl;
}