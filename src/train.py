import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

os.makedirs('models', exist_ok=True)

# 加载数据
df = pd.read_csv('D:/Python learn/credit-card-fraud-detection/data/creditcard.csv')

# 定义特征列
v_features = [f'V{i}' for i in range(1,29)]
original_features = ['Amount', 'Time']
all_features = v_features + original_features
X = df[all_features]
y = df['Class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 定义预处理步骤：对Amount和Time做标准化
preprocessor = ColumnTransformer(
    transformers=[
        ('amount_time', StandardScaler(), ['Amount', 'Time']),
        ('v_features', 'passthrough', v_features)
    ])

# 构建完整pipeline（包含SMOTE和分类器）
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# 训练
print("训练模型...")
pipeline.fit(X_train, y_train)

# 保存整个pipeline
joblib.dump(pipeline, 'models/full_pipeline.pkl')
print("模型已保存到 models/full_pipeline.pkl")