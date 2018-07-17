# -fintech-
需求分析：
财经新闻作为重要却海量的投资数据，无时无刻不在影响着投资者们的投资决策，为了更好地提示客户当下新闻事件对应的投资机会和投资风险，本课以研发“历史事件连连看”为目的，旨在根据当前新闻内容从历史事件中搜索出相似新闻报道，后期可以结合事件与行情，辅助客户采取相应投资策略。
 该赛题是让参赛者为每一条测试集数据寻找其最相似的TOP 20条新闻（不包含测试新闻本身），我们会根据参赛者提交的结果和实际的数据进行对比，采用mAP值作为评价指标。 
* bm25new_param.py  
使用bm25算法进行文本相似度分析,为了提高相似度计算精确度，除了对训练测试集数据进行分词和停用词操作以外，还对训练集数据额外进行了重复语句的去重操作，结果证明这样的一列数据处理过程，提升了bm25算法的理论上限；
* bm25new.py  
针对bm25算法的实现代码，使用了IDF原理判断一个词与一个文档的相关性的权重，
![](https://raw.githubusercontent.com/jaygle17/-finthech-/master/bm25.png)
