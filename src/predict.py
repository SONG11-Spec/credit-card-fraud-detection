import joblib
import pandas as pd

# 加载pipeline
pipeline = joblib.load('models/full_pipeline.pkl')


def predict(input_dict):
    """
    输入字典，包含所有原始特征：V1-V28, Amount, Time
    返回预测结果
    """
    df = pd.DataFrame([input_dict])
    # 确保所有特征存在（如果缺失，补0）
    expected_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]  # 按正确顺序排列

    prob = pipeline.predict_proba(df)[0, 1]
    pred = 1 if prob >= 0.3 else 0
    return {
        'prediction': int(pred),
        'probability': float(prob),
        'risk_level': '高风险' if pred == 1 else '低风险'
    }