#include "dataset.h"
void dataset::read_data(const string & file_name)
{
	name = file_name;
	ifstream file(file_name + ".conll.txt");
	if (!file)
	{
		cout << "don't open " + file_name + "conll.txt" << endl;
	}
	string line;
	sentence sen;
	int sentence_count = 0, word_count = 0;
	while (getline(file, line))
	{
		if (line.size() == 0)
		{
			sentences.push_back(sen);
			sen.~sentence();
			sentence_count++;
			continue;
		}
		word_count++;
		int t0 = line.find("\t") + 1;
		int t1 = line.find("\t", t0);
		string word = line.substr(t0, t1 - t0);
		int t2 = line.find("\t", t1 + 1) + 1;
		int t3 = line.find("\t", t2);
		string tag = line.substr(t2, t3 - t2);
		sen.word.push_back(word);
		sen.tag.push_back(tag);
		//构建词的单个元素。
		vector<string> word_char;
		for (unsigned t4 = 0; t4 < word.size();)
		{
			if ((unsigned char)word[t4] > 129 && (unsigned char)word[t4 + 1] > 64)
			{
				word_char.push_back(word.substr(t4, 2));
				t4 = t4 + 2;
			}
			else
			{
				word_char.push_back(word.substr(t4, 1));
				t4++;
			}
		}
		sen.word_char.push_back(word_char);
	}
	cout << name << ".conll contains sentence count " << sentence_count << endl;
	cout << name << ".conll contains word count " << word_count << endl;
}

dataset::dataset()
{
}


dataset::~dataset()
{
}
