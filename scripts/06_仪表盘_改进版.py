#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLP-1 模型性能监控仪表板（完整版：SHAP + LIME + PDP）
功能：
1. 加载最佳模型（逻辑回归）及测试数据
2. 绘制 ROC、PR、混淆矩阵、概率分布、阈值优化曲线
3. 输出特征重要性（系数）图
4. 进行 SHAP 全局解释，生成总结图
5. 进行 LIME 局部解释，生成一个高风险样本的 HTML 解释
6. 进行 PDP 分析，展示关键特征与预测概率的边际关系
7. 生成 Markdown 性能报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from datetime import datetime
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay

# 可选导入 SHAP 和 LIME，若未安装则跳过
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP 未安装，跳过 SHAP 分析。")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME 未安装，跳过 LIME 分析。")

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 特征名称中文映射
FEATURE_NAMES_CN = {
    'enrollment_log': '注册人数对数变换（试验规模与统计效力特征）',
    'start_year': '试验开始年份（药物研发时代与里程碑特征）',
    'pre_semaglutide_era': '司美格鲁肽上市前时代（药物研发时代与里程碑特征）',
    'post_semaglutide_era': '司美格鲁肽上市后时代（药物研发时代与里程碑特征）',
    'phase_Unknown': '试验阶段未知（试验阶段与监管风险特征）',
    'phase_PHASE4': 'IV期上市后研究（试验阶段与监管风险特征）',
    'is_obesity': '肥胖相关试验（适应症与目标人群风险特征）',
    'is_t2d': '2型糖尿病试验（适应症与目标人群风险特征）',
    'is_weight_loss': '减重为主要终点的试验（适应症与目标人群风险特征）',
    'exc_count': '排除标准数量（入排标准与患者选择特征）',
    'criteria_total_len': '入排标准总字符数（入排标准与患者选择特征）',
    'mentions_bmi': '提及BMI（安全性文本信号强度特征）',
    'mentions_contraindication': '提及禁忌症（安全性文本信号强度特征）',
    'mentions_renal_cutoff': '提及肾功能阈值（安全性文本信号强度特征）',
    'high_risk_term_count': '高风险术语计数（安全性文本信号强度特征）',
    'risk_ratio': '风险比率（安全性文本信号强度特征）',
    'year_x_enrollment': '年份 × 注册人数对数（交互特征）',
    'enrollment_log_x_phase_Unknown': '注册人数对数 × 阶段未知（交互特征）'
}

# ==================== 配置参数（与 05_集成学习.py 保持一致） ====================
RANDOM_STATE = 42
TEST_SIZE = 0.2
USE_TIME_SPLIT = True           # 是否按年份划分
TIME_THRESHOLD = 2018            # 用 2018 年及以后作为测试集

# 路径设置
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, 'visualizations')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_full_data():
    """加载完整的带标签特征数据"""
    data_path = os.path.join(PROCESSED_DIR, 'glp1_18clinical_features_with_labels_correct.csv')
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ['nct_id', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    return X, y, feature_cols, df


def split_data(X, y, df, use_time_split, time_threshold):
    """与 05_集成学习.py 相同的划分逻辑"""
    if use_time_split:
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
    return X_train, X_test, y_train, y_test


def load_model_and_scaler():
    """加载标准化器和最佳模型 pipeline"""
    scaler = joblib.load(os.path.join(MODELS_DIR, 'standard_scaler.pkl'))
    model_pipeline = joblib.load(os.path.join(MODELS_DIR, 'best_model_pipeline.pkl'))
    feature_names = pd.read_csv(os.path.join(MODELS_DIR, 'feature_names.csv'), header=None).squeeze().tolist()
    return scaler, model_pipeline, feature_names


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """计算常用指标"""
    return {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def plot_roc_curve(y_true, y_pred_proba, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', label='随机分类器')
    ax.set_xlabel('假正率 (FPR)', fontsize=12)
    ax.set_ylabel('真正率 (TPR)', fontsize=12)
    ax.set_title('ROC曲线 - 模型区分能力评估', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    return roc_auc


def plot_pr_curve(y_true, y_pred_proba, ax=None):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, lw=2, label=f'PR AUC = {pr_auc:.4f}')
    ax.set_xlabel('召回率 (Recall)', fontsize=12)
    ax.set_ylabel('精确率 (Precision)', fontsize=12)
    ax.set_title('精确率-召回率曲线 - 不平衡数据性能评估', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    return pr_auc


def plot_confusion_matrix(y_true, y_pred, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测低风险', '预测高风险'],
                yticklabels=['实际低风险', '实际高风险'],
                ax=ax, cbar=False, annot_kws={"size": 12})
    ax.set_title('混淆矩阵 - 分类结果可视化', fontsize=14, fontweight='bold')
    ax.set_xlabel('预测类别', fontsize=12)
    ax.set_ylabel('实际类别', fontsize=12)


def plot_probability_distribution(y_true, y_pred_proba, ax=None):
    ax.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.5, label='低风险', color='blue', density=True)
    ax.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.5, label='高风险', color='red', density=True)
    ax.set_xlabel('预测概率', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('预测概率分布 - 高风险与低风险样本区分度', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_threshold_optimization(y_true, y_pred_proba, ax=None):
    thresholds = np.linspace(0.1, 0.9, 17)
    accs, f1s = [], []
    for thresh in thresholds:
        y_pred_t = (y_pred_proba >= thresh).astype(int)
        accs.append(accuracy_score(y_true, y_pred_t))
        f1s.append(f1_score(y_true, y_pred_t, zero_division=0))
    ax.plot(thresholds, accs, 'o-', label='准确率', linewidth=2)
    ax.plot(thresholds, f1s, 's-', label='F1分数', linewidth=2)
    ax.set_xlabel('分类阈值', fontsize=12)
    ax.set_ylabel('性能分数', fontsize=12)
    ax.set_title('阈值优化曲线 - 平衡准确率与F1分数', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    ax.axvline(best_thresh, color='red', linestyle='--', linewidth=2, label=f'最佳F1阈值={best_thresh:.2f}')
    ax.legend(fontsize=10)
    return best_thresh, best_f1


def plot_feature_importance_logistic(model, feature_names, ax=None):
    """逻辑回归系数图（绝对值排序，标注正负）"""
    clf = model.steps[-1][1]  # pipeline最后一步
    if not hasattr(clf, 'coef_'):
        ax.text(0.5, 0.5, '模型无系数信息', ha='center', va='center')
        return
    coef = clf.coef_[0]
    indices = np.argsort(np.abs(coef))[::-1]
    top_n = min(18, len(feature_names))
    top_indices = indices[:top_n]
    top_coef = coef[top_indices]
    top_names = [FEATURE_NAMES_CN.get(feature_names[i], feature_names[i]) for i in top_indices]
    colors = ['red' if c > 0 else 'green' for c in top_coef]
    ax.barh(range(top_n), np.abs(top_coef), color=colors, alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel('系数绝对值', fontsize=12)
    ax.set_title('特征重要性分析（逻辑回归系数）\n红色：正相关（增加风险），绿色：负相关（降低风险）', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)


def shap_analysis(model, X_train, X_test, feature_names, save_dir):
    """SHAP 全局解释：生成 summary plot"""
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP 未安装，跳过 SHAP 分析。")
        return None
    try:
        # 从 pipeline 中提取分类器（最后一步）
        clf = model.steps[-1][1]
        # 创建 LinearExplainer（需要训练数据作为背景）
        explainer = shap.LinearExplainer(clf, X_train)
        shap_values = explainer.shap_values(X_test)

        # 使用中文特征名称
        feature_names_cn = [FEATURE_NAMES_CN.get(name, name) for name in feature_names]

        # 绘制 summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names_cn, show=False)
        plt.title('SHAP 特征重要性总结 - 全局可解释性分析（基于测试集）', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'shap_summary_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP 总结图已保存至 {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ SHAP 分析失败: {e}")
        return None


def lime_analysis(model, X_train, X_test, y_test, feature_names, save_dir):
    """LIME 局部解释：选择一个高风险测试样本生成解释"""
    if not LIME_AVAILABLE:
        print("⚠️ LIME 未安装，跳过 LIME 分析。")
        return None
    try:
        # 找出高风险样本（实际标签为1）且模型预测正确的样本（可选）
        high_risk_idx = np.where(y_test == 1)[0]
        if len(high_risk_idx) == 0:
            print("⚠️ 测试集中无高风险样本，跳过 LIME 分析。")
            return None
        # 选择第一个高风险样本
        idx = high_risk_idx[0]
        instance = X_test[idx].reshape(1, -1)

        # 获取预测函数（pipeline 的 predict_proba）
        predict_fn = model.predict_proba

        # 使用中文特征名称
        feature_names_cn = [FEATURE_NAMES_CN.get(name, name) for name in feature_names]

        # 创建 LIME 解释器（使用训练数据拟合分布）
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names_cn,
            class_names=['低风险', '高风险'],
            mode='classification',
            discretize_continuous=True
        )

        # 解释单个样本
        exp = explainer.explain_instance(
            data_row=instance.flatten(),
            predict_fn=predict_fn,
            num_features=10,
            top_labels=1
        )

        # 保存为 HTML 文件
        save_path = os.path.join(save_dir, 'lime_explanation.html')
        exp.save_to_file(save_path)
        print(f"✅ LIME 解释已保存至 {save_path}")

        # 同时生成一个简单的文本描述
        print("\n📋 LIME 解释摘要（高风险样本局部解释）：")
        for feat, weight in exp.as_list(label=1):
            print(f"  {feat}: {weight:.4f}")
        return save_path
    except Exception as e:
        print(f"❌ LIME 分析失败: {e}")
        return None


def plot_pdp_analysis(model, X_train, feature_names, save_dir):
    """PDP 分析：展示关键特征与预测概率的边际关系"""
    # 选择6个关键特征（可根据 SHAP 重要性或临床意义调整）
    pdp_features = [
        ('exc_count', '排除标准数量（入排标准与患者选择特征）'),
        ('enrollment_log', '注册人数对数变换（试验规模与统计效力特征）'),
        ('risk_ratio', '风险比率（安全性文本信号强度特征）'),
        ('phase_Unknown', '试验阶段未知（试验阶段与监管风险特征）'),
        ('mentions_contraindication', '提及禁忌症（安全性文本信号强度特征）'),
        ('mentions_renal_cutoff', '提及肾功能阈值（安全性文本信号强度特征）')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('部分依赖图 (PDP) - 关键特征与高风险概率的边际关系分析', fontsize=18, fontweight='bold')

    for idx, (feat_name, feat_label) in enumerate(pdp_features):
        ax = axes[idx // 3, idx % 3]
        try:
            feat_idx = feature_names.index(feat_name)
            # 使用中文特征名称
            feature_names_cn = [FEATURE_NAMES_CN.get(name, name) for name in feature_names]
            
            # 绘制 PDP
            display = PartialDependenceDisplay.from_estimator(
                model, X_train, [feat_idx],
                feature_names=feature_names_cn,
                ax=ax, grid_resolution=50,
                kind='average'
            )
            ax.set_xlabel(feat_label, fontsize=12)
            ax.set_ylabel('预测高风险概率', fontsize=12)
            ax.set_title(f'{feat_label}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=10)
        except ValueError:
            ax.text(0.5, 0.5, f'特征 {feat_name} 不存在', ha='center', va='center', fontsize=12)
            ax.set_title(feat_label, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, 'pdp_analysis_dashboard.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ PDP 分析图已保存至 {save_path}")
    return save_path


def generate_report(metrics, best_thresh, best_f1, y_true, y_pred,
                    shap_path=None, lime_path=None, pdp_path=None):
    cm = confusion_matrix(y_true, y_pred)
    report = f"""# GLP-1 临床试验风险预测模型性能报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据集概览
- 测试样本总数: {len(y_true)}
- 高风险样本数: {np.sum(y_true)} ({np.mean(y_true)*100:.2f}%)
- 低风险样本数: {len(y_true)-np.sum(y_true)} ({(1-np.mean(y_true))*100:.2f}%)

## 模型性能指标
| 指标 | 值 |
|------|-----|
| AUC | {metrics['auc']:.4f} |
| PR-AUC | {metrics['pr_auc']:.4f} |
| 准确率 | {metrics['accuracy']:.4f} |
| 精确率 | {metrics['precision']:.4f} |
| 召回率 | {metrics['recall']:.4f} |
| F1分数 | {metrics['f1']:.4f} |

## 阈值优化
- 最优阈值（基于F1）: {best_thresh:.4f}
- 对应F1分数: {best_f1:.4f}

## 混淆矩阵（默认阈值0.5）
```
         预测低风险  预测高风险
实际低风险   {cm[0,0]:6d}      {cm[0,1]:6d}
实际高风险   {cm[1,0]:6d}      {cm[1,1]:6d}
```

## 可解释性分析
"""
    if shap_path:
        report += f"- SHAP 全局解释图：`{shap_path}`\n"
    else:
        report += "- SHAP 分析未执行或失败\n"
    if lime_path:
        report += f"- LIME 局部解释（高风险样本）：`{lime_path}`\n"
    else:
        report += "- LIME 分析未执行或失败\n"
    if pdp_path:
        report += f"- PDP 边际效应分析图：`{pdp_path}`\n"
    else:
        report += "- PDP 分析未执行或失败\n"

    report += f"""
## 解释与建议
- AUC = {metrics['auc']:.3f} 表明模型具有一定的区分能力。
- 召回率 = {metrics['recall']:.3f} 表示模型能识别 {metrics['recall']*100:.1f}% 的实际高风险试验。
- 精确率偏低，提示假阳性较多，可通过提高阈值牺牲部分召回率换取精确率。
- 类别极度不平衡（高风险仅 {np.mean(y_true)*100:.2f}%），建议持续关注召回率而非准确率。
"""
    return report


def main():
    print("="*60)
    print("GLP-1 模型性能监控仪表板（完整版：SHAP + LIME + PDP）")
    print("="*60)

    # 1. 加载完整数据并重新划分训练/测试集（用于 SHAP 背景和 LIME 训练数据）
    print("\n📦 加载完整数据并划分训练/测试集...")
    X, y, feature_names, df = load_full_data()
    X_train, X_test, y_train, y_test = split_data(X, y, df, USE_TIME_SPLIT, TIME_THRESHOLD)

    # 2. 标准化（重新拟合训练集）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. 加载已保存的模型
    print("\n🤖 加载已训练的模型...")
    scaler_saved, model_pipeline, feature_names_saved = load_model_and_scaler()
    # 验证特征名是否一致（检查集合是否相同，忽略顺序）
    if set(feature_names) != set(feature_names_saved):
        print("⚠️ 特征名称不一致，但继续执行...")
        print(f"  数据特征: {sorted(feature_names)}")
        print(f"  模型特征: {sorted(feature_names_saved)}")

    # 4. 对测试集进行预测
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = model_pipeline.predict(X_test)

    # 5. 计算指标
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    # 6. 绘制仪表板
    print("\n📊 绘制性能仪表板...")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('GLP-1 临床试验风险预测模型性能仪表板 - 综合性能评估与可解释性分析', fontsize=20, fontweight='bold')

    ax1 = plt.subplot(3, 3, 1)
    plot_roc_curve(y_test, y_pred_proba, ax1)

    ax2 = plt.subplot(3, 3, 2)
    plot_pr_curve(y_test, y_pred_proba, ax2)

    ax3 = plt.subplot(3, 3, 3)
    plot_confusion_matrix(y_test, y_pred, ax3)

    ax4 = plt.subplot(3, 3, 4)
    plot_probability_distribution(y_test, y_pred_proba, ax4)

    ax5 = plt.subplot(3, 3, 5)
    best_thresh, best_f1 = plot_threshold_optimization(y_test, y_pred_proba, ax5)

    ax6 = plt.subplot(3, 3, 6)
    plot_feature_importance_logistic(model_pipeline, feature_names, ax6)

    ax7 = plt.subplot(3, 3, (7, 9))
    ax7.axis('off')
    summary = f"""
    📊 模型性能总结（测试集）
    ========================
    
    🎯 主要指标
    AUC: {metrics['auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}
    准确率: {metrics['accuracy']:.4f} | 召回率: {metrics['recall']:.4f}
    精确率: {metrics['precision']:.4f} | F1: {metrics['f1']:.4f}
    
    ⚙️ 阈值优化
    最优阈值: {best_thresh:.4f} (F1={best_f1:.4f})
    
    📈 数据分布
    高风险样本数: {np.sum(y_test)} / {len(y_test)} ({np.mean(y_test)*100:.2f}%)
    低风险样本数: {len(y_test)-np.sum(y_test)} / {len(y_test)} ({(1-np.mean(y_test))*100:.2f}%)
    
    💡 模型特点
    • 基于18个临床驱动的风险特征
    • 使用逻辑回归算法（线性模型）
    • 支持SHAP、LIME、PDP可解释性分析
    """
    ax7.text(0.05, 0.95, summary, transform=ax7.transAxes, fontsize=12,
             verticalalignment='top', family='SimHei', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    dashboard_path = os.path.join(VISUALIZATIONS_DIR, 'model_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    print(f"✅ 仪表板已保存至 {dashboard_path}")

    # 7. SHAP 分析
    print("\n🔍 执行 SHAP 分析...")
    shap_path = shap_analysis(model_pipeline, X_train_scaled, X_test_scaled, feature_names, VISUALIZATIONS_DIR)

    # 8. LIME 分析
    print("\n🔎 执行 LIME 分析...")
    lime_path = lime_analysis(model_pipeline, X_train_scaled, X_test_scaled, y_test, feature_names, VISUALIZATIONS_DIR)

    # 9. PDP 分析
    print("\n📈 执行 PDP 分析...")
    pdp_path = plot_pdp_analysis(model_pipeline, X_train_scaled, feature_names, VISUALIZATIONS_DIR)

    # 10. 生成报告
    report = generate_report(metrics, best_thresh, best_f1, y_test, y_pred,
                             shap_path, lime_path, pdp_path)
    report_path = os.path.join(REPORTS_DIR, 'model_performance_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 性能报告已保存至 {report_path}")

    print("\n🎉 所有分析完成！")


if __name__ == "__main__":
    main()