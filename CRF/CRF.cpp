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
vector<double> CRF::count_score(const vector<string> &feature)
{
	int offset = 0;
	vector<vector<double>> scores(tag.size(), vector<double>(feature.size(), 0));
	for (int j = 0; j < tag.size(); j++)
	{
		offset = j * model.size();
		for (int i = 0; i < feature.size(); i++)
		{
			if (model.find(feature[i]) != model.end())
			{
				scores[j][i] += w[model[feature[i]] + offset];
			}
		}
	}
	vector<double> score(tag.size());
	for (int j = 0; j<tag.size(); j++)
	{
		score[j] = accumulate(scores[j].begin(), scores[j].end(), 0);
	}
	return score;
}
vector<vector<double>> CRF::forword(const sentence &sen)
{
	vector<vector<double>> scores(sen.word.size(), vector<double>(tag.size(), 0));
	//第一个词
	vector<string> first_feature = create_feature(sen, 0);
	first_feature.push_back("01:*&&");
	scores[0]=count_score(first_feature);
	for (int i = 0; i < tag.size(); i++)
	{
		scores[0][i] = exp(scores[0][i]);
	}
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
				p[i][j] = exp(head_prob[i][j] + curr_score[i]);
			}
		}
		vector<double> all_score(tag.size());
		for (int i = 0; i < tag.size(); i++)
		{
			int all = 0;
			for (int j = 0; j < vector_tag.size(); j++)
			{
				all =all+scores[z - 1][j] * p[i][j];
			}
			all_score[i] = all;
		}
		scores[z] = all_score;
	}

	//测试
	for (int i = 0; i < sen.tag.size(); i++)
	{
		for (int j = 0; j < tag.size(); j++)
		{
			cout << scores[i][j]<<"\t";
		}
		cout << endl;
	}
	return scores;
}
vector<vector<double>> CRF::backword(const sentence &sen)
{
	//方便计算。
	vector<vector<double>> scores(sen.tag.size(), vector<double>(tag.size(), 0));
	for (int i = 0; i < tag.size(); i++)
	{
		scores[sen.word.size()-1][i] = 1;//存个疑惑。
	}
	//其余的词。
	for (int z = sen.word.size()-2; z >=0; z--)
	{
		vector<string> curr_feature = create_feature(sen, z+1);
		vector<double> curr_score = count_score(curr_feature);
		vector<vector<double>> p(tag.size(), vector<double>(tag.size(), 0));
		for (int i = 0; i < tag.size(); i++)
		{

			for (int j = 0; j < tag.size(); j++)
			{
				p[i][j] = exp(head_prob[i][j] + curr_score[i]);
			}
		}
		vector<double> all_score(tag.size());
		for (int i = 0; i < tag.size(); i++)
		{
			int all = 0;
			for (int j = 0; j < vector_tag.size(); j++)
			{
				all = all + scores[z +1][j] * p[i][j];
			}
			all_score[i] = all;
		}
		scores[z] = all_score;
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
	vector<vector<double>> c(tag.size(), vector<double>(tag.size(), 0));
	for (int i = 0; i < tag.size(); i++)
	{

		for (int j = 0; j < tag.size(); j++)
		{
			int max_tag_id = model["01:*" + vector_tag[j]];
			c[i][j] = w[i*model.size() + max_tag_id];
		}
	}
	head_prob = c;
	vector<vector<double>> alpha = forword(sen);
	vector<vector<double>> beta = backword(sen);
	//计算分母。
	double z = accumulate(alpha[sen.tag.size() - 1].begin(), alpha[sen.tag.size() - 1].end(), 0.0);
	//对正确的序列进行处理，
	for (int i = 0; i < sen.word.size(); i++)
	{
		vector<string> first_feature = create_feature(sen, i);
		if (i == 0)
		{
			first_feature.push_back("01:*&&");
		}
		else
		{
			first_feature.push_back("01:*&&" + sen.tag[i - 1]);
		}
		vector<int> fv_id = get_id(first_feature);
		int offset = tag[sen.tag[i]] * model.size();
		for (int q = 0; q < fv_id.size(); ++q)
		{
			g[fv_id[q] + offset]++;
		}
	}
	//对第一个词进行处理。
	vector<string> first_feature = create_feature(sen, 0);
	first_feature.push_back("01:*&&");
	vector<int> fv_id = get_id(first_feature);
	vector<double> first_score = count_score(first_feature);
	for (int i = 0; i < tag.size(); i++)
	{
		int offset = i * model.size();
		double p = (beta[0][i] * exp(first_score[i])) / z;
		for (int j = 0; j < fv_id.size(); j++)
		{
			g[i + fv_id[j]] = g[i + fv_id[j]] - p;
		}
	}
	//对所有有可能的系列进行处理。
	for (int i = 1; i < sen.word.size(); ++i)
	{
		vector<string> curr_feature = create_feature(sen, i);
		vector<double> curr_score = count_score(curr_feature);//计算没有01项的分数总和。
		vector<vector<double>> head(tag.size(), vector<double>(tag.size(), 0));
		for (int i = 0; i < tag.size(); i++)
		{

			for (int j = 0; j < tag.size(); j++)
			{
				head[i][j] = exp(head_prob[i][j] + curr_score[i]);
			}
		}	
		vector<vector<double>> p(tag.size(), vector<double>(tag.size(), 0));
		for (int q = 0; q < tag.size(); q++)
		{
			for (int k = 0; k < tag.size(); k++)
			{
				p[q][k] = alpha[i - 1][k] * beta[i][q] * head[q][k] / z;
			}
		}
		for (int w = 0; w < tag.size(); w++)
		{
			int offset = w * model.size();
			for (int j = 0; j > tag.size(); j++)
			{
				for (int k = 0; k < fv_id.size(); k++)
				{
					g[fv_id[k] + offset] -= p[w][j];
				}
			}
		}
	}
	head_prob.clear();
}
vector<string> CRF::max_score_sentence_tag(const sentence &sen)
{
	vector<vector<double>> score_prob(sen.tag.size(), vector<double>(tag.size(), 0));
	vector<vector<int>> score_path(sen.tag.size(), vector<int>(tag.size(), 0));
	vector<vector<double>> head_prob(tag.size(), vector<double>(tag.size(), 0));
	for (int i = 0; i < tag.size(); i++)
	{

		for (int j = 0; j < tag.size(); j++)
		{
			int max_tag_id = model["01:*" + vector_tag[j]];
			head_prob[i][j] = w[i*model.size() + max_tag_id];
		}
	}
	for (int i = 0; i < tag.size(); i++)
	{
		score_path[0][i] = -1;
	}
	//第一个词。
	vector <string> feature = create_feature(sen, 0);
	feature.push_back("01:*&&");
	score_prob[0] = count_score(feature);
	//其余的词。
	for (int j = 1; j < sen.word.size(); j++)
	{

		vector<double> currect_prob(tag.size(), 0);
		vector <string> feature = create_feature(sen, j);
		vector<double> score = count_score(feature);
		for (int i = 0; i < tag.size(); i++)
		{
			for (int z = 0; z < tag.size(); z++)
			{
				currect_prob[z] = head_prob[i][z] + score_prob[j - 1][z];
			}
			auto w = max_element(begin(currect_prob), end(currect_prob));
			score_prob[j][i] = *w + score[i];
			score_path[j][i] = distance(begin(currect_prob), w);
		}
	}
	vector<string> path_max;
	vector<double> a = score_prob[sen.word.size() - 1];
	auto w = max_element(begin(a), end(a));
	int max_point = distance(begin(a), w);
	for (int j = sen.word.size() - 1; j > 0; j--)
	{
		path_max.push_back(vector_tag[max_point]);
		max_point = score_path[j][max_point];
	}
	path_max.push_back(vector_tag[max_point]);
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
		for (auto sen = train.sentences.begin(); sen != train.sentences.end(); sen++)
		{
			b++;
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
		//save_file(j);
		double train_precision = evaluate(train);
		double dev_precision = evaluate(dev);
		cout << "第" << j << "次迭代总耗时：" << durationTime << endl << endl;
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