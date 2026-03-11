# 信用卡欺诈检测系统

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)

## 项目简介
本项目使用机器学习方法检测信用卡欺诈交易。针对高度不平衡的数据，采用SMOTE过采样和逻辑回归模型，最终AUC达到0.98，召回率0.92（阈值0.3）。同时使用Streamlit构建了交互式Web应用，方便实时预测。

## 数据来源
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) 数据集，包含284,807笔交易，其中欺诈交易492笔（0.172%）。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
