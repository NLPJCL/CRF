#include "CRF.h"
vector<long double> logsumexp( vector<vector<long double>> a)
{
	vector<long double> new_vector(a.size());
	for (int i = 0; i <a.size(); i++)
	{
		auto w = max_element(begin(a[i]), end(a[i]));
		long double max = *w;
		long double all = 0.0;
		for (int j = 0; j < a[i].size(); j++)
		{
			a[i][j] = exp(a[i][j] - max);
		}
		all = log(accumulate(a[i].begin(), a[i].end(), 0.0));
		new_vector[i] = max + all;
	}
	return new_vector;
}
long double logsumexp(vector<long double> a)
{

		long double new_double;
		auto w = max_element(begin(a), end(a));
		long double max = *w;
		long double all = 0.0;
		for (int j = 0; j < a.size(); j++)
		{
			a[j] = exp(a[j] - max);
		}
		all = log(accumulate(a.begin(), a.end(), 0.0));
		new_double = max + all;
		return new_double;
}
vector<vector<long double>> translation(vector<vector<long double>>& a)
{
	vector<vector<long double>> b(a.size(), vector<long double>(a.size(), 0));
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < a[i].size(); j++)
		{
			b[i][j] = a[j][i];
		}
	}
	return b;
}
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
//	test.read_data("test");
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
				f.push_back("01:*"+ sen->tag[i - 1]);
			}
			for (auto i = f.begin(); i != f.end(); i++)
			{
				if (model.find(*i) == model.end())
				{
					model[*i] = word_count;
					//feature.push_back(*i);
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
	vector<vector<long double>> c(tag.size(), vector<long double>(tag.size(), 0));
	for (int i = 0; i < tag.size(); i++)
	{
		for (int j = 0; j < tag.size(); j++)
		{
			if (model.find("01:*" + vector_tag[j]) != model.end())
			{
				int max_tag_id = model["01:*" + vector_tag[j]];
				c[i][j] = w[i*model.size() + max_tag_id];
			}
			else
			{
				cout <<"出错，不是所有词性都在model里面。扩充了model，model乱了。"<< endl;
			}
		}
	}
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
			fv.push_back(t->second);
		}
	}
	return fv;
}
vector<long double> CRF::count_score(const vector<string> &feature)
{
	int offset = 0;
	vector<vector<long double>> scores(tag.size(), vector<long double>(feature.size(), 0));
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
	vector<long double> score(tag.size());
	for (int j = 0; j<tag.size(); j++)
	{
		score[j] = accumulate(scores[j].begin(), scores[j].end(), 0);
	}
	return score;
}
vector<vector<long double>> CRF::forword(const sentence &sen)
{
	vector<vector<long double>> scores(sen.word.size(), vector<long double>(tag.size(), 0));
	//第一个词
	vector<string> first_feature = create_feature(sen, 0);
	first_feature.push_back("01:*&&");
	scores[0]=count_score(first_feature);
	//其余的词。
	for (int z = 1; z < sen.word.size(); z++)
	{
		vector<string> curr_feature = create_feature(sen, z);
		vector<long double> curr_score = count_score(curr_feature);
		vector<vector<long double>> p(tag.size(), vector<long double>(tag.size(), 0));
		for (int i = 0; i < tag.size(); i++)
		{
			for (int j = 0; j < tag.size(); j++)
			{
				p[i][j] = head_prob[i][j] + curr_score[j]+scores[z-1][j];//新修改了i到j。
			}
		}
		scores[z] = logsumexp(p);
	}
	return scores;
}
vector<vector<long double>> CRF::backword(const sentence &sen)
{
	//方便计算。
	vector<vector<long double>> scores(sen.tag.size(), vector<long double>(tag.size(), 0));
	//其余的词。	
	vector<vector<long double>> q = translation(this->head_prob);
	for (int z = sen.word.size()-2; z >=0; z--)
	{
		vector<string> curr_feature = create_feature(sen, z+1);
		vector<long double> curr_score = count_score(curr_feature);
		vector<vector<long double>> p(tag.size(), vector<long double>(tag.size(), 0));
		for (int i = 0; i < tag.size(); i++)
		{
			for (int j = 0; j < tag.size(); j++)
			{
				p[i][j] = q[i][j] + curr_score[i]+scores[z+1][j];
			}
		}
		scores[z] = logsumexp(p);
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
	vector<vector<long double>> alpha = forword(sen);
	vector<vector<long double>> beta = backword(sen);
	//计算分母。
	long double z = logsumexp(alpha[sen.word.size() - 1]);
//	double q = logsumexp(beta[0]);
//	cout << z << endl;
//	cout << q << endl;
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
			first_feature.push_back("01:*" + sen.tag[i - 1]);
		}
		vector<int> fv_id = get_id(first_feature);
		int offset = tag[sen.tag[i]] * model.size();
		for (int q = 0; q < fv_id.size(); q++)
		{
			g[fv_id[q] + offset]++;
		}
	}
	//对第一个词进行处理。
	vector<string> first_feature = create_feature(sen, 0);
	first_feature.push_back("01:*&&");
	vector<int> fv_id = get_id(first_feature);
	vector<long double> first_score = count_score(first_feature);
	//double all = 0.0;
	for (int i = 0; i < tag.size(); i++)
	{
		int offset = i * model.size();
		long double p = exp(beta[0][i] +first_score[i]-z);
	//	all += p;
		for (int j = 0; j < fv_id.size(); j++)
		{
			g[offset+fv_id[j]] -= p;
		}
	}
	//cout << all << endl;
	//对所有有可能的系列进行处理。
	for (int i = 1; i < sen.word.size(); i++)
	{
		vector<string> curr_feature = create_feature(sen, i);
		vector<long double> curr_score = count_score(curr_feature);
		vector<int> fv_id = get_id(curr_feature);
		vector<vector<long double>> head(tag.size(), vector<long double>(tag.size(), 0));
		for (int r = 0; r < tag.size(); r++)
		{
			for (int j = 0; j < tag.size(); j++)
			{
				head[r][j] = head_prob[r][j] + curr_score[r];
			}
		}	
	//	double all_prob = 0.0;
		vector<vector<long double>> p(tag.size(), vector<long double>(tag.size(), 0));
		for (int q = 0; q < tag.size(); q++)
		{
			for (int k = 0; k < tag.size(); k++)
			{
				p[q][k] = exp(alpha[i - 1][k] + beta[i][q] +head[q][k] -z);
		//		all_prob += p[q][k];
			}
		}	
	//	cout << all_prob << endl;
		for (int z = 0; z < tag.size(); z++)
		{
			for (int t= 0; t < tag.size(); t++)
			{
				
				for (int w = 0; w < fv_id.size(); w++)
				{
					g[z*model.size() + fv_id[w]] -= p[z][t];
				}
				g[model["01:*" + vector_tag[t]] + z * model.size()] -= p[z][t];
			}
		}
	}
}
vector<string> CRF::max_score_sentence_tag(const sentence &sen)
{
	vector<vector<long double>> score_prob(sen.tag.size(), vector<long double>(tag.size(), 0));
	vector<vector<int>> score_path(sen.tag.size(), vector<int>(tag.size(), 0));
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

		vector<long double> currect_prob(tag.size(), 0);
		vector <string> feature = create_feature(sen, j);
		vector<long double> score = count_score(feature);
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
	vector<long double> a = score_prob[sen.word.size() - 1];
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
	for (int j = 0; j<20; j++)
	{
		start = clock();
		cout << "iterator" << j << endl;
		int b = 0;
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
			for (int i = 0; i < tag.size(); i++)
			{
				for (int j = 0; j < tag.size(); j++)
				{
						int max_tag_id = model["01:*" + vector_tag[j]];
						head_prob[i][j] = w[i*model.size() + max_tag_id];
				}
			}
		}
		stop = clock();
		durationTime = ((double)(stop - start)) / CLK_TCK;
		//save_file(j);
		double train_precision = evaluate(train);
		double dev_precision = evaluate(dev);
	//	double test_precision = evaluate(test);
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