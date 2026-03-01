#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析试验开始年份分布，按2021年分割数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_time_distribution():
    """分析试验开始年份分布"""
    
    # 读取数据
    data = pd.read_csv('processed_data/glp1_18clinical_features_with_labels_correct.csv')
    
    print('=== 试验开始年份分布分析 ===')
    print(f'数据总数: {len(data)}')
    print(f'年份范围: {data["start_year"].min():.1f} - {data["start_year"].max():.1f}')
    print(f'年份中位数: {data["start_year"].median():.1f}')
    print(f'年份均值: {data["start_year"].mean():.1f}')
    
    # 按2021年分割
    before_2021 = data[data['start_year'] < 2021]
    after_2021 = data[data['start_year'] >= 2021]
    
    print(f'\n=== 按2021年分割结果 ===')
    print(f'2021年之前样本数: {len(before_2021)} ({len(before_2021)/len(data)*100:.1f}%)')
    print(f'2021年及之后样本数: {len(after_2021)} ({len(after_2021)/len(data)*100:.1f}%)')
    
    # 分析标签分布
    print(f'\n=== 标签分布分析 ===')
    print('总体标签分布:')
    print(data['label'].value_counts())
    print(f'高风险比例: {data["label"].mean()*100:.2f}%')
    
    print(f'\n2021年之前标签分布:')
    print(before_2021['label'].value_counts())
    print(f'高风险比例: {before_2021["label"].mean()*100:.2f}%')
    
    print(f'\n2021年及之后标签分布:')
    print(after_2021['label'].value_counts())
    print(f'高风险比例: {after_2021["label"].mean()*100:.2f}%')
    
    # 年份分布直方图
    print(f'\n=== 年份分布详情 ===')
    year_counts = data['start_year'].value_counts().sort_index()
    print('年份分布:')
    for year, count in year_counts.items():
        print(f'{year:.0f}: {count}个试验')
    
    # 可视化年份分布
    plt.figure(figsize=(12, 6))
    
    # 年份分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(data['start_year'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(2021, color='red', linestyle='--', linewidth=2, label='2021年分割线')
    plt.xlabel('试验开始年份')
    plt.ylabel('试验数量')
    plt.title('试验开始年份分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 高风险试验年份分布
    plt.subplot(1, 2, 2)
    high_risk_data = data[data['label'] == 1]
    if len(high_risk_data) > 0:
        plt.hist(high_risk_data['start_year'], bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(2021, color='red', linestyle='--', linewidth=2, label='2021年分割线')
        plt.xlabel('试验开始年份')
        plt.ylabel('高风险试验数量')
        plt.title('高风险试验年份分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, '无高风险试验数据', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('results/time_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\n✅ 时间分布分析图已保存至 results/time_distribution_analysis.png')
    
    return data, before_2021, after_2021

def create_time_split_model():
    """创建基于时间分割的模型"""
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    
    # 读取数据
    data = pd.read_csv('processed_data/glp1_18clinical_features_with_labels_correct.csv')
    
    # 特征列和标签列
    feature_cols = [col for col in data.columns if col not in ['nct_id', 'label']]
    X = data[feature_cols]
    y = data['label']
    
    # 方案1: 传统随机分割（作为基准）
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 方案2: 按2021年时间分割
    # 训练集: 2021年之前的数据
    # 测试集: 2021年及之后的数据
    train_mask = data['start_year'] < 2021
    test_mask = data['start_year'] >= 2021
    
    X_train_time = X[train_mask]
    X_test_time = X[test_mask]
    y_train_time = y[train_mask]
    y_test_time = y[test_mask]
    
    print(f'\n=== 时间分割方案详情 ===')
    print(f'训练集大小: {len(X_train_time)} ({len(X_train_time)/len(data)*100:.1f}%)')
    print(f'测试集大小: {len(X_test_time)} ({len(X_test_time)/len(data)*100:.1f}%)')
    print(f'训练集高风险比例: {y_train_time.mean()*100:.2f}%')
    print(f'测试集高风险比例: {y_test_time.mean()*100:.2f}%')
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_rand_scaled = scaler.fit_transform(X_train_rand)
    X_test_rand_scaled = scaler.transform(X_test_rand)
    
    X_train_time_scaled = scaler.fit_transform(X_train_time)
    X_test_time_scaled = scaler.transform(X_test_time)
    
    # 训练逻辑回归模型
    lr_rand = LogisticRegression(class_weight='balanced', random_state=42)
    lr_time = LogisticRegression(class_weight='balanced', random_state=42)
    
    lr_rand.fit(X_train_rand_scaled, y_train_rand)
    lr_time.fit(X_train_time_scaled, y_train_time)
    
    # 预测
    y_pred_rand = lr_rand.predict(X_test_rand_scaled)
    y_pred_time = lr_time.predict(X_test_time_scaled)
    
    y_pred_proba_rand = lr_rand.predict_proba(X_test_rand_scaled)[:, 1]
    y_pred_proba_time = lr_time.predict_proba(X_test_time_scaled)[:, 1]
    
    # 评估指标
    def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        return {
            '模型': model_name,
            '准确率': accuracy,
            '精确率': precision,
            '召回率': recall,
            'F1分数': f1,
            'AUC': auc
        }
    
    results = []
    results.append(evaluate_model(y_test_rand, y_pred_rand, y_pred_proba_rand, '随机分割'))
    results.append(evaluate_model(y_test_time, y_pred_time, y_pred_proba_time, '时间分割(2021)'))
    
    # 输出结果
    results_df = pd.DataFrame(results)
    print(f'\n=== 模型性能比较 ===')
    print(results_df.round(4))
    
    # 保存结果
    results_df.to_csv('results/time_split_comparison.csv', index=False, encoding='utf-8-sig')
    print(f'✅ 时间分割比较结果已保存至 results/time_split_comparison.csv')
    
    return results_df

if __name__ == "__main__":
    print("🔍 开始分析时间分割效果...")
    
    # 分析时间分布
    data, before_2021, after_2021 = analyze_time_distribution()
    
    # 创建时间分割模型
    results = create_time_split_model()
    
    print("\n🎉 时间分割分析完成！")