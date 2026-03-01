#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLP-1临床试验风险预测特征构建 - 特征分析与可视化模块

功能：
1. 加载特征矩阵
2. 特征统计分析
3. 特征相关性分析
4. 特征分布可视化
5. 特征质量评估

作者：系统管理员
创建日期：2026-02-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
from datetime import datetime
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GLPTrialFeatureAnalyzer:
    """GLP-1临床试验特征分析器"""
    
    def __init__(self, features_file):
        """初始化分析器"""
        self.features_file = features_file
        self.features_df = None
        self.numeric_features = None
        self.categorical_features = None
        
        # 特征中文解释映射
        self.feature_names_cn = {
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
        
    def load_features(self):
        """加载特征数据"""
        print("1. 加载特征数据...")
        
        # 加载特征矩阵
        self.features_df = pd.read_csv(self.features_file)
        
        print(f"   - 特征矩阵维度: {self.features_df.shape}")
        print(f"   - 特征数量: {len(self.features_df.columns) - 1}")
        
        # 分离特征列
        feature_columns = [col for col in self.features_df.columns if col != 'nct_id']
        
        # 分类数值型和类别型特征
        self.numeric_features = []
        self.categorical_features = []
        
        for col in feature_columns:
            if self.features_df[col].dtype in ['int64', 'float64']:
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        print(f"   - 数值型特征: {len(self.numeric_features)} 个")
        print(f"   - 类别型特征: {len(self.categorical_features)} 个")
        
    def basic_statistics(self):
        """基本统计分析"""
        print("2. 基本统计分析...")
        
        if self.numeric_features:
            print("\n数值型特征统计:")
            stats_df = self.features_df[self.numeric_features].describe()
            print(stats_df.round(3))
            
        if self.categorical_features:
            print("\n类别型特征统计:")
            for col in self.categorical_features:
                value_counts = self.features_df[col].value_counts()
                print(f"\n{col}:")
                print(value_counts.head(10))  # 只显示前10个
                if len(value_counts) > 10:
                    print(f"... 还有 {len(value_counts) - 10} 个类别")
    
    def correlation_analysis(self):
        """特征相关性分析"""
        print("3. 特征相关性分析...")
        
        if len(self.numeric_features) < 2:
            print("   - 数值型特征不足，跳过相关性分析")
            return
        
        # 计算相关系数矩阵
        corr_matrix = self.features_df[self.numeric_features].corr()
        
        print("\n特征相关性矩阵 (前10个特征):")
        top_features = self.numeric_features[:10]
        print(corr_matrix.loc[top_features, top_features].round(3))
        
        # 可视化相关性热图
        self.plot_correlation_heatmap(corr_matrix)
        
        # 识别高度相关的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                          corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print("\n高度相关的特征对 (|r| > 0.8):")
            for pair in high_corr_pairs:
                print(f"   - {pair[0]} & {pair[1]}: r = {pair[2]:.3f}")
        else:
            print("\n未发现高度相关的特征对")
    
    def plot_correlation_heatmap(self, corr_matrix):
        """绘制相关性热图（使用中文特征名称）"""
        plt.figure(figsize=(16, 14))
        
        # 使用mask只显示下三角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # 创建中文特征名称列表
        cn_feature_names = []
        for feature in corr_matrix.columns:
            if feature in self.feature_names_cn:
                cn_feature_names.append(self.feature_names_cn[feature])
            else:
                cn_feature_names.append(feature)
        
        # 创建新的相关系数矩阵（使用中文名称）
        corr_matrix_cn = corr_matrix.copy()
        corr_matrix_cn.columns = cn_feature_names
        corr_matrix_cn.index = cn_feature_names
        
        sns.heatmap(corr_matrix_cn, mask=mask, annot=True, fmt=".2f", 
                   cmap='coolwarm', center=0, square=True,
                   cbar_kws={"shrink": .8}, annot_kws={"size": 8})
        plt.title('GLP-1临床试验特征相关性热图', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   - 相关性热图已保存（使用中文特征名称）")
    
    def feature_distribution_analysis(self):
        """特征分布分析（使用中文特征名称）"""
        print("4. 特征分布分析...")
        
        if not self.numeric_features:
            print("   - 无数值型特征，跳过分布分析")
            return
        
        # 创建分布图
        n_features = len(self.numeric_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, feature in enumerate(self.numeric_features):
            if i < len(axes):
                # 获取中文特征名称
                feature_name_cn = self.feature_names_cn.get(feature, feature)
                
                # 直方图
                axes[i].hist(self.features_df[feature].dropna(), bins=30, alpha=0.7, color='skyblue')
                axes[i].set_title(f'{feature_name_cn}分布', fontsize=10)
                axes[i].set_xlabel(feature_name_cn, fontsize=9)
                axes[i].set_ylabel('频数', fontsize=9)
                
                # 添加统计信息
                mean_val = self.features_df[feature].mean()
                std_val = self.features_df[feature].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.2f}')
                axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
                axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
                axes[i].legend(fontsize=8)
                
                # 设置刻度字体大小
                axes[i].tick_params(axis='both', which='major', labelsize=8)
        
        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('GLP-1临床试验特征分布分析', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图像
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'visualizations')
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   - 特征分布图已保存（使用中文特征名称）")
    
    def feature_quality_assessment(self):
        """特征质量评估"""
        print("5. 特征质量评估...")
        
        quality_report = []
        
        for feature in self.numeric_features + self.categorical_features:
            col_data = self.features_df[feature]
            
            # 缺失值检查
            missing_count = col_data.isnull().sum()
            missing_percent = (missing_count / len(col_data)) * 100
            
            # 唯一值检查
            unique_count = col_data.nunique()
            
            # 零值检查（数值型特征）
            zero_count = 0
            if feature in self.numeric_features:
                zero_count = (col_data == 0).sum()
            
            quality_report.append({
                'feature': feature,
                'type': '数值型' if feature in self.numeric_features else '类别型',
                'missing_count': missing_count,
                'missing_percent': missing_percent,
                'unique_count': unique_count,
                'zero_count': zero_count if feature in self.numeric_features else 'N/A'
            })
        
        # 创建质量报告数据框
        quality_df = pd.DataFrame(quality_report)
        
        print("\n特征质量报告:")
        print(quality_df.to_string(index=False))
        
        # 识别有问题的特征
        problematic_features = quality_df[quality_df['missing_percent'] > 50]
        if not problematic_features.empty:
            print("\n⚠️ 警告: 以下特征缺失值超过50%:")
            for _, row in problematic_features.iterrows():
                print(f"   - {row['feature']}: {row['missing_percent']:.1f}% 缺失")
        
        # 保存质量报告
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'reports')
        os.makedirs(output_dir, exist_ok=True)
        quality_df.to_csv(os.path.join(output_dir, 'feature_quality_report.csv'), index=False)
        
        print("   - 特征质量报告已保存")
    
    def save_analysis_results(self):
        """保存分析结果"""
        print("6. 保存分析结果...")
        
        # 保存特征统计摘要到reports目录
        if self.numeric_features:
            stats_summary = self.features_df[self.numeric_features].describe().T
            stats_summary['variance'] = self.features_df[self.numeric_features].var()
            stats_summary['skewness'] = self.features_df[self.numeric_features].apply(stats.skew)
            stats_summary['kurtosis'] = self.features_df[self.numeric_features].apply(stats.kurtosis)
            
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            stats_summary.to_csv(os.path.join(reports_dir, 'feature_statistics.csv'))
            print("   - 特征统计摘要已保存")
        
        # 保存特征数据到results根目录
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        self.features_df.to_csv(os.path.join(results_dir, 'analysis_features.csv'), index=False)
        print("   - 分析用特征数据已保存")

def main():
    """主函数"""
    try:
        start_time = datetime.now()
        print(f"=== GLP-1临床试验特征分析与可视化 ===")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查特征文件是否存在
        features_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'data', 'processed', 'glp1_18clinical_features.csv'
        )
        
        if not os.path.exists(features_file):
            print("❌ 特征文件不存在，请先运行02_特征工程.py")
            sys.exit(1)
        
        # 创建分析器实例
        analyzer = GLPTrialFeatureAnalyzer(features_file)
        
        # 执行分析流程
        analyzer.load_features()
        analyzer.basic_statistics()
        analyzer.correlation_analysis()
        analyzer.feature_distribution_analysis()
        analyzer.feature_quality_assessment()
        analyzer.save_analysis_results()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ 特征分析完成!")
        print(f"完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"处理耗时: {duration:.2f} 秒")
        
        # 显示生成的文件
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            if files:
                print("\n生成的分析文件:")
                for file in files:
                    print(f"   - {file}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 特征分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()