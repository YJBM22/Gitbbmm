# 邮件分类系统

## 核心功能
- 支持 ​**高频词特征** 与 ​**TF-IDF 加权特征** 双模式切换
- 集成 SMOTE 过采样算法解决样本不平衡问题
- 提供精确率(Precision)/召回率(Recall)/F1值 多维度评估报告

## 算法基础
https://latex.codecogs.com/svg.latex?P(C%7CX)%20%3D%20%5Cfrac%7BP(X%7CC)P(C)%7D%7BP(X)%7D
### 多项式朴素贝叶斯
基于贝叶斯定理与特征条件独立性假设：
$$ P(C|X) = \frac{P(X|C)P(C)}{P(X)} $$
- ​**先验概率** $P(C)$：通过训练数据统计类别分布
- ​**条件概率** $P(X|C)$：假设特征间相互独立，计算各特征在类别的出现频率
- ​**预测规则**：取后验概率最大的类别

## 数据处理流程
1. ​**文本清洗**：使用正则表达式过滤标点、数字等无效字符
2. ​**中文分词**：调用 Jieba 分词工具进行分词
3. ​**停用词过滤**：移除长度 ≤1 的词语
4. ​**特征构建**：
   - 高频词模式：统计前 100 个高频词的出现次数
   - TF-IDF 模式：计算词频-逆文档频率权重

## 特征工程对比
| 特征类型   | 数学表达                          | 优势                        | 劣势                  |
|------------|----------------------------------|----------------------------|-----------------------|
| 高频词     | $ \text{count}(w_i) $           | 计算高效，解释性强          | 忽略词语区分度        |
| TF-IDF     | $ \text{tf}(w_i) \times \text{idf}(w_i) $ | 强调区分性强的特征词        | 计算复杂度较高        |

## 使用方法
### 1. 特征模式切换
```python
# 初始化时指定特征方法
classifier = EmailClassifier(feature_method='tfidf')  # 或 'freq'
2. 样本平衡控制
python
# 启用 SMOTE 过采样
classifier = EmailClassifier(balance=True)
3. 执行预测
python
classifier.predict('邮件_files/151.txt')
模型评估
训练完成后自动输出分类评估报告：

              precision    recall  f1-score
普通邮件        0.8958      1.0000    0.9451
垃圾邮件        1.0000      0.8898    0.9417
项目结构
邮件分类系统/
├── 邮件_files/       # 邮件数据集
│   ├── 0.txt
│   └── ...
├── classify.py      # 主程序
└── README.md        # 项目文档
依赖库
bash
pip install jieba scikit-learn imbalanced-learn
