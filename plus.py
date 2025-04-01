import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '邮件_files')


class EmailClassifier:
    def __init__(self, feature_method='freq', balance=False):
        """
        初始化邮件分类器
        :param feature_method: 'freq' 高频词 | 'tfidf' TF-IDF
        :param balance: 是否启用SMOTE样本平衡
        """
        self.feature_method = feature_method
        self.balance = balance
        self.top_words = None
        self.tfidf_vectorizer = None
        self.model = MultinomialNB()

    def _validate_path(self, filename):
        """增强路径验证"""
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"无效文件路径: {os.path.abspath(filename)}")
        return filename

    def get_words(self, filename):
        """文本预处理流程"""
        self._validate_path(filename)
        words = []
        with open(filename, 'r', encoding='utf-8', errors='ignore') as fr:
            for line in fr:
                # 增强文本清洗
                line = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', line.strip())
                words.extend(filter(lambda w: len(w) > 1, cut(line)))
        return ' '.join(words)

    def _build_features(self, filenames):
        """特征工程核心方法"""
        if self.feature_method == 'freq':
            all_words = [self.get_words(f).split() for f in filenames]
            word_counter = Counter(chain(*all_words))
            self.top_words = [w for w, _ in word_counter.most_common(100)]
            return np.array([[words.count(w) for w in self.top_words] for words in all_words])
        else:
            texts = [self.get_words(f) for f in filenames]
            self.tfidf_vectorizer = TfidfVectorizer(
                tokenizer=lambda x: x.split(),
                token_pattern=None,
                max_features=100,
                sublinear_tf=True,  # 亚线性缩放
                norm='l2'  # L2归一化
            )
            return self.tfidf_vectorizer.fit_transform(texts).toarray()

    def train(self, train_files, labels):
        """模型训练流程"""
        X = self._build_features(train_files)
        y = np.array(labels)

        # 样本平衡处理
        if self.balance:
            X, y = SMOTE(random_state=42).fit_resample(X, y)

        self.model.fit(X, y)

        # 训练集评估
        print("\n[模型训练] 训练集分类报告:")
        print(classification_report(y, self.model.predict(X),
                                    target_names=['普通邮件', '垃圾邮件'],
                                    digits=4))

    def predict(self, filename, verbose=True):
        """执行预测"""
        filename = self._validate_path(filename)

        if self.feature_method == 'freq':
            words = self.get_words(filename).split()
            vec = np.array([words.count(w) for w in self.top_words])
        else:
            vec = self.tfidf_vectorizer.transform([self.get_words(filename)]).toarray()[0]

        pred = self.model.predict(vec.reshape(1, -1))[0]
        result = '垃圾邮件' if pred == 1 else '普通邮件'

        if verbose:
            print(f"文件 {os.path.basename(filename)} 分类结果: {result}")
        return pred  # 返回数值型结果便于评估

    def evaluate(self, test_files, true_labels):
        """新增测试集评估方法"""
        preds = [self.predict(f, verbose=False) for f in test_files]
        print("\n[模型评估] 测试集分类报告:")
        print(classification_report(true_labels, preds,
                                    target_names=['普通邮件', '垃圾邮件'],
                                    digits=4))


if __name__ == "__main__":
    # 初始化分类器
    classifier = EmailClassifier(
        feature_method='tfidf',  # 可选 'freq' 或 'tfidf'
        balance=True
    )

    # 训练数据（示例数据需要实际存在）
    train_files = [os.path.join(DATA_DIR, f'{i}.txt') for i in range(151)]
    train_labels = [1] * 127 + [0] * 24
    classifier.train(train_files, train_labels)

    # 测试数据（假设152-156为测试文件，需要真实标签）
    test_files = [os.path.join(DATA_DIR, f'{i}.txt') for i in range(151, 156)]
    test_labels = [1, 0, 1, 1, 0]  # 示例标签

    # 执行测试评估
    classifier.evaluate(test_files, test_labels)

    # 单文件预测示例
    sample_file = os.path.join(DATA_DIR, '152.txt')
    classifier.predict(sample_file)