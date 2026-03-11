import streamlit as st
from src.predict import predict

st.set_page_config(page_title="信用卡欺诈检测", layout="wide")
st.title("💳 信用卡欺诈检测系统")
st.markdown("输入交易特征，模型将预测是否为欺诈交易。")

st.sidebar.header("输入交易特征")

# 创建V1-V28输入框（默认值0）
with st.sidebar.expander("V1-V28 (PCA特征)", expanded=False):
    v_inputs = {}
    for i in range(1, 29):
        v_inputs[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.2f", key=f'v{i}')

# 原始特征输入
amount = st.sidebar.number_input('交易金额 (Amount)', value=100.0)
time = st.sidebar.number_input('交易时间 (秒)', value=50000)

# 组合所有特征
features = {**v_inputs, 'Amount': amount, 'Time': time}

if st.sidebar.button("开始预测"):
    with st.spinner("模型推理中..."):
        result = predict(features)

    col1, col2, col3 = st.columns(3)
    col1.metric("预测结果", result['risk_level'])
    col2.metric("欺诈概率", f"{result['probability']:.2%}")
    col3.metric("阈值", "0.3")

    # 进度条
    st.progress(result['probability'])

    if result['probability'] < 0.3:
        st.success("✅ 低风险交易")
    elif result['probability'] < 0.7:
        st.warning("⚠️ 中等风险，建议复核")
    else:
        st.error("🚨 高风险欺诈交易")