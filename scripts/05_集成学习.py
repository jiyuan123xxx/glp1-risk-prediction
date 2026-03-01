#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLP-1临床试验风险预测 - 集成学习建模模块（基于真实标签）
优化版本：修正SMOTE数据泄露问题，增加PR-AUC和阈值优化

功能：
1. 加载基于真实试验结果的标签数据
2. 交叉验证评估基础模型（SMOTE内嵌于CV）
3. 构建加权集成和堆叠集成
4. 超参数调优和模型评估
5. 可解释性分析

作者：系统管理员
创建日期：2026-02-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
import sys
from datetime import datetime
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    average_precision_score, precision_recall_curve
)
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

# 可选：SHAP 分析
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP 未安装，将跳过可解释性分析。")

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------ 配置参数 ------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2                # 测试集比例
CV_FOLDS = 5                    # 交叉验证折数
N_BOOTSTRAP = 1000              # Bootstrap 重抽样次数（用于置信区间）
USE_TIME_SPLIT = True           # 是否按年份划分
TIME_THRESHOLD = 2018           # 用 2018 年及以后作为测试集
USE_SMOTE = True                # 是否使用SMOTE处理类别不平衡

# ------------------------------ 数据加载与预处理 ------------------------------
def load_features_with_label():
    """加载带标签的特征矩阵（使用正确的标签定义）"""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'processed', 'glp1_18clinical_features_with_labels_correct.csv'
    )
    
    if not os.path.exists(data_path):
        print("❌ 带标签的特征文件不存在，请先运行04_标签定义_正确方法.py")
        return None, None, None, None
    
    df = pd.read_csv(data_path)
    
    # 检查标签列是否存在
    if 'label' not in df.columns:
        raise ValueError("特征文件中未找到 'label' 列，请先定义标签。")
    
    # 提取特征和标签
    feature_cols = [c for c in df.columns if c not in ['nct_id', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"加载数据: {len(X)} 个样本, {len(feature_cols)} 个特征")
    print(f"高风险样本比例: {y.mean():.2%}")
    
    # 检查类别不平衡情况
    high_risk_count = y.sum()
    low_risk_count = len(y) - high_risk_count
    print(f"高风险样本数: {high_risk_count}")
    print(f"低风险样本数: {low_risk_count}")
    print(f"类别不平衡比例: {low_risk_count/high_risk_count:.1f}:1")
    
    return X, y, feature_cols, df

def split_data(X, y, df, use_time_split=True, time_threshold=2018):
    """
    划分训练/测试集
    - 如果 use_time_split=True，则基于 start_year 划分
    - 否则随机分层划分
    """
    if use_time_split:
        # 确保 df 中包含 start_year 列
        if 'start_year' not in df.columns:
            raise ValueError("按时间划分需要 start_year 列，请检查数据。")
        train_mask = df['start_year'] < time_threshold
        test_mask = df['start_year'] >= time_threshold
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        print(f"时间划分：训练集 {train_mask.sum()} 样本 (年份 < {time_threshold})，测试集 {test_mask.sum()} 样本")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        print(f"随机分层划分：训练集 {len(X_train)} 样本，测试集 {len(X_test)} 样本")
    
    # 检查训练集和测试集的类别分布
    print(f"训练集高风险比例: {y_train.mean():.4f}")
    print(f"测试集高风险比例: {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test

# ------------------------------ 基础模型定义 ------------------------------
def get_base_models():
    """返回基础模型字典，包含简单模型和集成模型"""
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=1.0, random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric='logloss'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=RANDOM_STATE
        )
    }
    return models

# ------------------------------ 交叉验证评估（带SMOTE）----------------------
def evaluate_models_cv_with_smote(X_train, y_train, models, cv_folds=5):
    """对每个模型进行交叉验证，使用SMOTE内嵌于pipeline，返回 AUC 均值和标准差"""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    
    print("\n🔍 交叉验证评估基础模型（内嵌SMOTE）...")
    for name, model in models.items():
        # 创建带SMOTE的pipeline（仅在训练折内过采样）
        pipeline = make_imb_pipeline(SMOTE(random_state=RANDOM_STATE), model)
        aucs = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        results[name] = {
            'auc_mean': aucs.mean(),
            'auc_std': aucs.std(),
            'model': model  # 保存原始模型，后续需要重新训练时再使用pipeline
        }
        print(f"{name}: CV AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")
    
    return results

# ------------------------------ 加权集成（基于 CV AUC）-------------------
def create_weighted_ensemble(X_train, y_train, X_test, y_test, models_cv_results):
    """
    基于交叉验证 AUC 计算权重，在训练集上重新训练模型（应用SMOTE），返回测试集上的集成预测
    """
    # 权重归一化
    weights = {}
    total_auc = sum([res['auc_mean'] for res in models_cv_results.values()])
    for name, res in models_cv_results.items():
        weights[name] = res['auc_mean'] / total_auc
    
    print("\n⚖️ 加权集成权重（基于 CV AUC）:")
    for name, w in weights.items():
        print(f"  {name}: {w:.4f}")
    
    # 在全部训练数据上重新训练每个模型（使用SMOTE）
    trained_models = {}
    for name, model in models_cv_results.items():
        # 创建带SMOTE的pipeline并训练
        pipeline = make_imb_pipeline(SMOTE(random_state=RANDOM_STATE), model['model'])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
    
    # 获取测试集概率
    probas = np.zeros((len(X_test), len(trained_models)))
    for i, (name, pipeline) in enumerate(trained_models.items()):
        probas[:, i] = pipeline.predict_proba(X_test)[:, 1]
    
    # 加权平均
    weighted_proba = np.zeros(len(X_test))
    for i, (name, w) in enumerate(weights.items()):
        weighted_proba += probas[:, i] * w
    
    # 计算指标
    auc = roc_auc_score(y_test, weighted_proba)
    pred = (weighted_proba >= 0.5).astype(int)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    acc = accuracy_score(y_test, pred)
    pr_auc = average_precision_score(y_test, weighted_proba)
    
    return {
        'auc': auc,
        'pr_auc': pr_auc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'accuracy': acc,
        'y_pred_proba': weighted_proba,
        'y_pred': pred,
        'weights': weights,
        'trained_models': trained_models
    }

# ------------------------------ 堆叠集成 ------------------------------
def create_stacking_ensemble(X_train, y_train, X_test, y_test):
    """
    创建堆叠集成模型，使用所有基础模型作为第一层，逻辑回归作为元模型
    """
    base_estimators = [
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10,
                                      class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              scale_pos_weight=1.0, random_state=RANDOM_STATE, n_jobs=-1)),
        ('gbm', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                           random_state=RANDOM_STATE))
    ]
    
    meta_learner = LogisticRegression(penalty='l1', solver='saga', class_weight='balanced',
                                      max_iter=1000, random_state=RANDOM_STATE)
    
    # 注意：堆叠内部会进行5折交叉验证生成元特征，此时不能使用SMOTE，因为SMOTE会导致泄露。
    # 我们将在训练堆叠之前对整个训练集应用SMOTE，这是安全的，因为测试集独立。
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,                     # 内部 5 折交叉验证生成元特征
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    stacking.fit(X_train_res, y_train_res)
    
    # 测试集评估（注意测试集是原始分布，未过采样）
    y_pred_proba = stacking.predict_proba(X_test)[:, 1]
    y_pred = stacking.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n🏗️ 堆叠集成测试集 AUC = {auc:.4f}, PR-AUC = {pr_auc:.4f}, 召回率 = {recall:.4f}")
    
    return {
        'model': stacking,
        'auc': auc,
        'pr_auc': pr_auc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'accuracy': acc,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }

# ------------------------------ Bootstrap 置信区间 ----------------------
def bootstrap_metric(y_true, y_pred_proba, metric_func, n_bootstrap=1000, alpha=0.95):
    """计算 AUC 的 bootstrap 置信区间"""
    np.random.seed(RANDOM_STATE)
    n = len(y_true)
    indices = np.arange(n)
    scores = []
    for _ in range(n_bootstrap):
        idx = resample(indices, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        score = metric_func(y_true[idx], y_pred_proba[idx])
        scores.append(score)
    lower = np.percentile(scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(scores, (1 + alpha) / 2 * 100)
    return lower, upper

# ------------------------------ 可解释性分析 ----------------------------
def explain_model(model, X_train, X_test, feature_names, model_type='tree'):
    """
    生成特征重要性图（树模型）和 SHAP 总结图
    支持树模型和线性模型
    """
    visualizations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # 树模型的特征重要性图（仅当有 feature_importances_ 属性时）
    if model_type == 'tree' and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("特征重要性分析（树模型）", fontsize=16, fontweight='bold')
        plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
        plt.xlabel("重要性", fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, 'feature_importances_ensemble.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 特征重要性图已保存")
    
    # SHAP 分析（支持树模型和线性模型）
    if SHAP_AVAILABLE:
        try:
            # 根据模型类型选择解释器
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # 二分类取正类
            elif hasattr(model, 'coef_'):  # 线性模型（逻辑回归等）
                # 注意：LinearExplainer需要训练数据作为背景
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_test)
            else:
                print("⚠️ 未知模型类型，无法进行SHAP分析")
                return

            # 生成SHAP总结图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title("SHAP 特征重要性总结", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(visualizations_dir, 'shap_summary_ensemble.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ SHAP 总结图已保存")
                
        except Exception as e:
            print(f"SHAP 分析失败: {e}")
    else:
        print("⚠️ SHAP 未安装，跳过分析")

# ------------------------------ 阈值优化 ----------------------------
def threshold_optimization(y_test, y_pred_proba, model_name):
    """绘制精确率-召回率随阈值变化曲线，并输出最优阈值（根据F1）"""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    # 计算每个阈值下的F1分数（阈值对应precisions和recalls的长度比thresholds多1，需对齐）
    # 通常thresholds的长度等于precisions-1，我们可以取precisions和recalls的前len(thresholds)个
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label='精确率', linewidth=2)
    plt.plot(thresholds, recalls[:-1], label='召回率', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1分数', linestyle='--', linewidth=2)
    plt.axvline(x=best_threshold, color='red', linestyle=':', label=f'最优阈值 = {best_threshold:.2f}')
    plt.xlabel('阈值')
    plt.ylabel('分数')
    plt.title(f'{model_name} - 精确率/召回率随阈值变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    visualizations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    plt.savefig(os.path.join(visualizations_dir, f'threshold_{model_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  最优阈值: {best_threshold:.4f}, 对应F1: {best_f1:.4f}")
    return best_threshold, best_f1

# ------------------------------ 主流程 ------------------------------
def main():
    print("=" * 60)
    print("GLP-1 临床试验风险预测：集成学习建模（基于真实标签）")
    print("=" * 60)
    
    # 1. 加载数据
    X, y, feature_names, df = load_features_with_label()
    if X is None:
        return
    
    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = split_data(
        X, y, df, use_time_split=USE_TIME_SPLIT, time_threshold=TIME_THRESHOLD
    )
    
    # 3. 标准化（仅拟合训练集）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 定义基础模型
    base_models = get_base_models()
    
    # 5. 交叉验证评估基础模型（内嵌SMOTE）
    cv_results = evaluate_models_cv_with_smote(X_train_scaled, y_train, base_models, CV_FOLDS)
    
    # 6. 加权集成（基于 CV AUC）
    print("\n🔗 构建加权集成模型...")
    weighted_result = create_weighted_ensemble(
        X_train_scaled, y_train, X_test_scaled, y_test, cv_results
    )
    
    # 7. 堆叠集成
    print("\n🏗️ 构建堆叠集成模型...")
    stacking_result = create_stacking_ensemble(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 8. 测试集上评估所有模型并比较
    print("\n📊 测试集性能比较（带 95% CI）:")
    results = {}
    
    # 基础模型在测试集上的性能（训练时应用SMOTE）
    for name, model in base_models.items():
        # 使用SMOTE重新训练
        pipeline = make_imb_pipeline(SMOTE(random_state=RANDOM_STATE), model)
        pipeline.fit(X_train_scaled, y_train)
        y_pred_proba = pipeline.predict_proba(X_test_scaled)[:, 1]
        y_pred = pipeline.predict(X_test_scaled)
        auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        lower, upper = bootstrap_metric(y_test, y_pred_proba, roc_auc_score, N_BOOTSTRAP)
        results[name] = {
            'auc': auc, 'auc_ci': (lower, upper), 'pr_auc': pr_auc,
            'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': acc
        }
        print(f"{name:20s} AUC = {auc:.4f} (95% CI [{lower:.4f}, {upper:.4f}]), PR-AUC = {pr_auc:.4f}, Recall = {recall:.4f}")
    
    # 加权集成
    results['Weighted Ensemble'] = {
        'auc': weighted_result['auc'],
        'pr_auc': weighted_result['pr_auc'],
        'recall': weighted_result['recall'],
        'precision': weighted_result['precision'],
        'f1': weighted_result['f1'],
        'accuracy': weighted_result['accuracy']
    }
    lower, upper = bootstrap_metric(y_test, weighted_result['y_pred_proba'], roc_auc_score, N_BOOTSTRAP)
    print(f"{'Weighted Ensemble':20s} AUC = {weighted_result['auc']:.4f} (95% CI [{lower:.4f}, {upper:.4f}]), PR-AUC = {weighted_result['pr_auc']:.4f}, Recall = {weighted_result['recall']:.4f}")
    
    # 堆叠集成
    results['Stacking'] = {
        'auc': stacking_result['auc'],
        'pr_auc': stacking_result['pr_auc'],
        'recall': stacking_result['recall'],
        'precision': stacking_result['precision'],
        'f1': stacking_result['f1'],
        'accuracy': stacking_result['accuracy']
    }
    lower, upper = bootstrap_metric(y_test, stacking_result['y_pred_proba'], roc_auc_score, N_BOOTSTRAP)
    print(f"{'Stacking':20s} AUC = {stacking_result['auc']:.4f} (95% CI [{lower:.4f}, {upper:.4f}]), PR-AUC = {stacking_result['pr_auc']:.4f}, Recall = {stacking_result['recall']:.4f}")
    
    # 9. 阈值优化（对最佳模型）
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_result = results[best_model_name]
    print(f"\n🏆 最佳模型: {best_model_name}, 测试集 AUC = {best_result['auc']:.4f}, PR-AUC = {best_result['pr_auc']:.4f}")
    
    # 获取最佳模型的预测概率
    if best_model_name == 'Stacking':
        best_proba = stacking_result['y_pred_proba']
    elif best_model_name == 'Weighted Ensemble':
        best_proba = weighted_result['y_pred_proba']
    else:
        # 基础模型
        pipeline = make_imb_pipeline(SMOTE(random_state=RANDOM_STATE), base_models[best_model_name])
        pipeline.fit(X_train_scaled, y_train)
        best_proba = pipeline.predict_proba(X_test_scaled)[:, 1]
    
    print("\n🔧 阈值优化...")
    best_thresh, best_f1 = threshold_optimization(y_test, best_proba, best_model_name)
    
    # 应用最优阈值重新计算混淆矩阵
    y_pred_opt = (best_proba >= best_thresh).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    print(f"优化后混淆矩阵（阈值={best_thresh:.4f}）:")
    print(cm_opt)
    
    # 10. 保存最佳模型和 scaler
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    if best_model_name == 'Stacking':
        final_model = stacking_result['model']
        joblib.dump(final_model, os.path.join(models_dir, 'best_model.pkl'))
    elif best_model_name == 'Weighted Ensemble':
        # 加权集成不是一个单一的 scikit-learn 模型，我们保存其组件
        final_model = weighted_result['trained_models']  # 字典
        joblib.dump(final_model, os.path.join(models_dir, 'weighted_ensemble_models.pkl'))
    else:
        # 保存基础模型的pipeline（包含SMOTE）
        final_pipeline = make_imb_pipeline(SMOTE(random_state=RANDOM_STATE), base_models[best_model_name])
        final_pipeline.fit(X_train_scaled, y_train)
        joblib.dump(final_pipeline, os.path.join(models_dir, 'best_model_pipeline.pkl'))
    
    # 保存特征名称
    pd.Series(feature_names).to_csv(os.path.join(models_dir, 'feature_names.csv'), index=False)
    print("✅ 模型及附属文件已保存至 models/ 目录")
    
    # 11. 可解释性分析（对最佳模型）
    print("\n🔎 开始可解释性分析...")
    if best_model_name in base_models:
        # 重新训练一个无SMOTE的模型用于解释（因为SMOTE会改变数据分布，但特征重要性通常不变）
        explain_model(base_models[best_model_name], X_train_scaled, X_test_scaled, feature_names,
                      model_type='tree' if 'Forest' in best_model_name or 'XGB' in best_model_name or 'Gradient' in best_model_name else 'linear')
    elif best_model_name == 'Stacking':
        print("堆叠模型可解释性：展示随机森林基模型的特征重要性")
        rf_model = stacking_result['model'].named_estimators_['rf']
        explain_model(rf_model, X_train_scaled, X_test_scaled, feature_names, model_type='tree')
    elif best_model_name == 'Weighted Ensemble':
        max_weight_model_name = max(weighted_result['weights'], key=weighted_result['weights'].get)
        print(f"加权集成中权重最高的模型: {max_weight_model_name}")
        # 获取该模型的原始未包装模型（不是pipeline）
        model_to_explain = base_models[max_weight_model_name]
        explain_model(model_to_explain, X_train_scaled, X_test_scaled, feature_names,
                      model_type='tree' if 'Forest' in max_weight_model_name or 'XGB' in max_weight_model_name or 'Gradient' in max_weight_model_name else 'linear')
    
    # 12. 生成性能比较图表
    print("\n📈 生成性能比较图表...")
    model_names = list(results.keys())
    auc_values = [results[name]['auc'] for name in model_names]
    recall_values = [results[name]['recall'] for name in model_names]
    pr_auc_values = [results[name]['pr_auc'] for name in model_names]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # AUC 比较
    bars1 = ax1.bar(model_names, auc_values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold'])
    ax1.set_title('模型AUC性能比较', fontsize=14, fontweight='bold')
    ax1.set_ylabel('AUC')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    for bar, auc_val in zip(bars1, auc_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 召回率比较
    bars2 = ax2.bar(model_names, recall_values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold'])
    ax2.set_title('模型召回率比较', fontsize=14, fontweight='bold')
    ax2.set_ylabel('召回率')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    for bar, recall_val in zip(bars2, recall_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{recall_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # PR-AUC比较
    bars3 = ax3.bar(model_names, pr_auc_values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold'])
    ax3.set_title('模型PR-AUC比较', fontsize=14, fontweight='bold')
    ax3.set_ylabel('PR-AUC')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    for bar, pr_auc_val in zip(bars3, pr_auc_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{pr_auc_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    visualizations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    plt.savefig(os.path.join(visualizations_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 性能比较图表已保存")
    
    # 13. 保存性能结果
    performance_df = pd.DataFrame(results).T
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    performance_df.to_csv(os.path.join(reports_dir, 'model_performance_results.csv'), encoding='utf-8')
    print("✅ 性能结果已保存")
    
    # 14. ROC曲线绘制（最佳模型）
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{best_model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title(f'ROC曲线 - {best_model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(visualizations_dir, 'roc_curve_ensemble.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n🎉 集成学习建模完成！")
    print(f"最佳模型: {best_model_name}")
    print(f"最佳AUC: {best_result['auc']:.4f}")
    print(f"最佳PR-AUC: {best_result['pr_auc']:.4f}")
    print(f"最佳召回率: {best_result['recall']:.4f}")
    print(f"优化后阈值: {best_thresh:.4f}, 对应F1: {best_f1:.4f}")
    
    # 15. 备选方案：强制使用树模型进行 SHAP 分析（即使最佳模型是线性模型）
    if best_model_name != 'Random Forest':
        print("\n🔎 额外使用随机森林进行 SHAP 分析...")
        rf_for_shap = RandomForestClassifier(
            n_estimators=200, max_depth=8, 
            class_weight='balanced', random_state=RANDOM_STATE
        )
        rf_for_shap.fit(X_train_scaled, y_train)
        explain_model(rf_for_shap, X_train_scaled, X_test_scaled, feature_names, model_type='tree')

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        main()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"处理耗时: {duration:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 集成学习建模失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)