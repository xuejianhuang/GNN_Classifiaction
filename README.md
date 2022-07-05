# 基于图神经网络的学术文献自动分类
   随着自然语言处理技术的发展，学者们提出了各种基于文本分类的学术论文分类方法。在早期研究中，主要基于特征工程，从论文内容、题目、摘要、关键字等提取论文的表示特征，然后基于朴素贝叶斯（NB）、最近邻算法（KNN）、支持向量机（SVM）等传统机器学习算法构建分类模型。近年来，随着深度学习技术的日益成熟，基于CNN、LSTM、Transformer等模型的文本分类成为研究热点。虽然，基于文本分类的学术论文分类研究取得了不少进展，但该分类方法仅考虑了论文标题、摘要、关键字等自身特征，没有融合参考文献和引证文献的特征，导致分类准确率不佳。所以，本实验通过引用关系构建论文关系图数据，提出基于图神经网络（GNN）的学术论文自动分类模型，提高学术论文自动分类的准确率。
  * 开发语言：python、pytorch、sklearn
# 数据集说明
  本文实验使用的数据是基于微软学术文献生成的论文关系图，其中的节点是论文，边是论文间的引用关系。原始数据集包含3063061篇论文，29168650条边，共23个类别。
  数据集下载地址链接：https://pan.baidu.com/s/1G7CtmsZ-thsMW2zjVhH2IQ  提取码：isc3 

# 实验模型
### 图神经网络模型
   * GCN
   * GraphSAGE
   * GAT 
### 传统文本分类模型
   * 朴素贝叶斯
   * 最近邻算法
   * 支持向量机
   * LSTM
   * Transformer
# 文件目录说明
  * baseline:传统文本分类模型目录
      - results：baseline模型实验结果目录
      - baseline_config.py：baseline配置文件
      - DataSampler.py：数据采样类
      - DNN.py：朴素贝叶斯模型
      - KNN.py:最近邻模型
      - SVM.py：支持向量机模型
      - LSTM.py：LSTM模型
      - Transformer.py:Transformer模型
  * checkpoint:检查点保存目录，为了实现训练过程中突然中断的恢复
  * pre_data:数据预处理和统计分析代码
  * results:GNN模型实验结果保存目录
  * confusin.py:可视化混合矩阵
  * GAT.py:GAT及其改进模型实现类，通过参数可以配置多头注意力个数和残差结构
  * GCN.py:GCN模型实现类
  * main.py:项目主文件
  * SAGE.py：GraphSAGE模型实现类
  * utils.py:工具包文件
  * visualize.py:visdom动态结果可视化类
 # 主要第三方包
   * numpy=1.21.4
   * pandas=1.2.0
   * matplotlib=3.3.2
   * sklearn=0.24.1
   * dgl=0.7.2
   * pytorch=1.9.1
   * visdom=0.1.8.9
# 实验结果
  基于多头注意力机制和残差连接改进的GAT模型的准确率最高，达到了61%。
