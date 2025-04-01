# 邮件分类系统

##  核心功能
基于朴素贝叶斯的二分类系统，支持以下特性：
- 双特征模式：高频词统计 / TF-IDF加权
- 样本平衡处理：SMOTE过采样技术
- 多维度评估：精度、召回率、F1值报告

##  算法原理
### 多项式朴素贝叶斯
**数学表达**：
```
P(y|x₁,...,xₙ) ∝ P(y)∏ᵢP(xᵢ|y)
```
**邮件分类应用**：
1. 计算先验概率：`P(垃圾邮件)` 和 `P(普通邮件)`
2. 计算条件概率：`P(特征词|邮件类型)`
3. 根据贝叶斯定理预测后验概率

##  数据处理流程
```python
流程图
文本输入 → 正则清洗 → Jieba分词 → 停用词过滤 → 特征构建
```

**关键处理**：
- 正则清洗：`re.sub(r'[.【】0-9、——。，！~\*]', '', text)`
- 分词处理：`jieba.cut(line)`
- 停用词过滤：`filter(lambda w: len(w) > 1, words)`

##  特征工程对比
| 特征类型   | 数学表达                  | 实现差异                 |
|----------|-------------------------|-----------------------|
| 高频词特征 | `count(w_i)`           | 基于词频统计Top100高频词     |
| TF-IDF   | `tf(w_i)*log(N/(1+df))` | 使用TfidfVectorizer计算 |

**配置参数**：
```python
TfidfVectorizer(
    tokenizer=lambda x: x.split(),  # 自定义分词器
    max_features=100                # 与高频词特征维度一致
)
```

##  使用指南
### 初始化配置
```python
# 创建分类器（可配置参数）
classifier = EmailClassifier(
    feature_method='tfidf',  # 'freq'或'tfidf'
    balance=True              # 启用SMOTE平衡
)
```

### 训练与预测
```python
# 训练模型
train_files = [f'{i}.txt' for i in range(151)]
labels = [1]*127 + [0]*24  # 初始数据分布
classifier.train(train_files, labels)

# 预测新邮件
classifier.predict('test_mail.txt')
```

## 📈 扩展功能
### 样本平衡效果
| 处理阶段   | 垃圾邮件 | 普通邮件 |
|----------|--------|--------|
| 原始数据   | 127    | 24     |
| SMOTE后  | 127    | 127    |

### 评估指标示例
```
              precision    recall  f1-score   support
    普通邮件     0.9730    0.9600    0.9665       127
    垃圾邮件     0.9608    0.9730    0.9669       127
```

##  项目部署
1. 克隆仓库
```bash
git clone https://github.com/yourname/email-classifier.git
```
2. 安装依赖
```bash
pip install -r requirements.txt  # 包含jieba, sklearn, imblearn等
```
3. 数据准备
```
项目结构：
/homework3/
├── 邮件_files/      # 存放所有txt文件
├── email_classifier.py
└── README.md
