# -*- coding: utf-8 -*-
# python list[],tuple(),dict{},set={}无序不重复
import jieba
import pandas as pd
from zhon.hanzi import punctuation
from string import punctuation as spun
# gensim.summarization import bm25
import bm25new
#得到干扰符号集合
stop_symbol = punctuation+spun
stop_symbol = list(set(list(stop_symbol)))

# 停用词导入
stop_words = []
path = '../data/stop_words.txt'
with open(path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop_words.append(line[:-1])  # line[:-1]其实就是去除了这行文本的最后一个字符（换行符）后剩下的部分。
    stop_words = list(set(stop_words))

#  主函数：训练和测试模型
def train_text():
    datas = pd.read_csv("../data/train_data.csv")
    titles = datas['title']
    ids = datas['id']
    # 字典, key为去重句子，value为索引
    deltitle_id = {}
    # 去重句子
    all_doc = []
    # 去重句子id
    all_ids = []
    # 重复句索引
    remainID_2_delID = {} #key：test中id；value：train中对应重复句id
    # 完整title
    id_2_title = {}

    #开始筛选训练集之前，先记录下test，防止删除test中相同句子
    test_data = pd.read_csv('../data/test_data.csv', encoding='gbk')
    test_titles = test_data['title']
    for i in range(len(test_titles)):
        test_title = fun_stop(test_titles[i]) # 对test中每个句子，提前用停用词处理，得到处理后句子记为test_title
        id = test_data['id'][i]
        deltitle_id[test_title] = id # 然后字典deltitle用来记录对应句子和id,目的是为了训练集中发现test中句子不去删除

    #开始删除相同的训练样本，提高训练精度
    for i in range(len(titles)):
        title = titles[i]
        id = ids[i]
        id_2_title[id] = title # 将训练集中id和title构建字典索引
        del_title = fun_stop(title) # 将训练集中句子处理为去掉停用词后句子
        # 判断title重复，删除；并确保test句子没删
        if del_title in deltitle_id.keys() and deltitle_id[del_title] != id:
		# 如果处理后训练集中存在和test中重复句子，并且不是同一个句子，那么
            remain_id = deltitle_id[del_title] # 记录test中id
            if remain_id in remainID_2_delID.keys():
                remainID_2_delID[remain_id].append(id)# 如果已有，就在对应要删除重复的字典中再添加id
            else:
                remainID_2_delID[remain_id] = [id]#如果没有，当前id就是对应要删除的id
            continue
        else:
            deltitle_id[del_title] = id # 训练集中没重复的也加入deltitle_id字典中
            all_doc.append(title)
            all_ids.append(id)
			
    print('训练数据处理...............')
    all_doc_list = []
    for doc in all_doc:
        # 结巴分词
        words = jieba.cut(doc)
        # 删除停用词
        doc_list = [word for word in words if word not in stop_words]
        all_doc_list.append(doc_list)
		
    print('测试数据处理...............')
    test_datas = pd.read_csv("../data/test_data.csv", encoding="gbk")
	# 测试集数据很少，所以没有额外处理，只是分词和停用词
    test_titles = test_datas["title"]
    test_doc = [title for title in test_titles]
    test_doc_list = []
    for doc in test_doc:
        # 结巴分词
        words = jieba.cut(doc)
        # 删除停用词
        doc_list = [word for word in words if word not in stop_words]
        test_doc_list.append(doc_list)
		
    print('相似度计算筛选开始..............')
    corpus = all_doc_list
    # 使用bm25模型计算相似度和排序筛选
	# 使用bm25模型内使用模板：可以计算scores
    bm25Model = bm25new.BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    results = []
    for doc in test_doc_list:
        scores = bm25Model.get_scores(doc, average_idf) # 得分是每条新闻和训练集中逐条新闻相似度
		# 后面是按自己需求对得到的score排序
        similiar_sorted = sorted(enumerate(scores), key=lambda item: -item[1])[:21]
        indexs = [str(all_ids[item[0]]) for item in similiar_sorted]
        results.append(" ".join(indexs))
    # 存储数据到results文件中
    with open("../data/results.txt", "w") as f:
        f.write("source_id"+"\t"+"target_id"+'\t' + "source_title" + '\t' + "target_title" + '\n')
        for item in results:
            item = item.strip().split()
            print(item)
            for i in range(1, 21):
                try:
                    f.write(item[0] + "\t" + item[i] + "\t" + datas["title"][int(item[0])-1] + "\t" + datas["title"][int(item[i])-1] + "\n")
                except:
                    f.write(item[0] + "\t" + item[i] + "\n")

#删除相同的训练样本前，先把样本中的干扰词删掉
def fun_stop(title):
    a = ""
    for i in range(len(title)):
        if title[i] not in stop_symbol:
            a = a + title[i]
    return a

if __name__ == "__main__":
	# 训练和得到结果
    train_text()
	# 写入最后txt文本
    with open("../data/results.txt", "r") as f, open("bm25new_param.txt", "w") as wf:
        datas = f.readlines()
        for data in datas:
            data = data.strip().split("\t")
            try:
                wf.write(data[0] + "\t" + data[1] + "\t" + data[2] + '\t' + data[3] + '\n')
            except:
                wf.write(data[0] + "\t" + data[1] + '\n')