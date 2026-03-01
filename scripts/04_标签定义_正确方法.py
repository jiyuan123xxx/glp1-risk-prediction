#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLP-1临床试验风险预测 - 正确的标签定义方法（基于真实试验结果）

根据文档要求，基于真实试验结果定义风险标签：
- 高风险：试验因安全性/伦理原因提前终止
- 低风险：试验顺利完成或因非安全性原因终止

作者：系统管理员
创建日期：2026-02-21
"""

import pandas as pd
import numpy as np
import re
import os
import sys
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def is_high_risk(row):
    """
    判断试验是否为高风险
    
    高风险条件：
    1. 试验状态为 TERMINATED、WITHDRAWN 或 SUSPENDED
    2. why_stopped 字段包含安全性关键词
    
    参数:
        row: DataFrame 行
    
    返回:
        int: 1=高风险, 0=低风险
    """
    # 安全性关键词列表（不区分大小写）
    safety_keywords = [
        'safety', 'adverse', 'toxicity', 'side effect', 'unsafe',
        'death', 'fatal', 'severe', 'serious', 'unethical', 'harm',
        'patient died', 'mortality', 'elevated liver enzymes',
        'pancreatitis', 'hypoglycemia', 'cardiovascular', 'stroke',
        'hospitalization', 'withdrawn due to ae'  # AE = adverse event
    ]
    
    # 只处理终止状态
    if row['overall_status'] not in ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
        return 0  # 低风险
    
    # 检查 why_stopped 字段
    why = str(row['why_stopped']).lower() if pd.notna(row['why_stopped']) else ''
    
    # 检查是否包含任何安全性关键词
    for kw in safety_keywords:
        if re.search(kw, why):
            return 1
    
    return 0  # 低风险

def create_correct_labels():
    """创建基于真实试验结果的正确标签"""
    
    # 加载原始试验数据
    studies_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'raw', 'studies.txt'
    )
    
    if not os.path.exists(studies_file):
        print("❌ 原始试验数据文件不存在")
        return None
    
    # 加载特征数据
    features_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'processed', 'glp1_18clinical_features.csv'
    )
    
    if not os.path.exists(features_file):
        print("❌ 特征文件不存在，请先运行02_特征工程.py")
        return None
    
    # 读取数据
    print("加载原始试验数据...")
    studies = pd.read_csv(studies_file, sep='|', dtype=str, low_memory=False)
    
    print("加载特征数据...")
    features = pd.read_csv(features_file)
    
    print(f"原始试验数据: {studies.shape[0]} 行, {studies.shape[1]} 列")
    print(f"特征数据: {features.shape[0]} 行, {features.shape[1]} 列")
    
    # 应用高风险判断函数
    print("应用高风险判断逻辑...")
    studies['label'] = studies.apply(is_high_risk, axis=1)
    
    # 统计标签分布
    label_counts = studies['label'].value_counts()
    high_risk_count = label_counts.get(1, 0)
    low_risk_count = label_counts.get(0, 0)
    total_count = len(studies)
    
    print(f"\n标签分布统计:")
    print(f"高风险试验 (label=1): {high_risk_count} ({high_risk_count/total_count*100:.2f}%)")
    print(f"低风险试验 (label=0): {low_risk_count} ({low_risk_count/total_count*100:.2f}%)")
    
    # 检查 why_stopped 字段的填充情况
    why_stopped_fill_rate = studies['why_stopped'].notna().sum() / len(studies) * 100
    print(f"why_stopped 字段填充率: {why_stopped_fill_rate:.2f}%")
    
    # 检查终止状态分布
    terminated_studies = studies[studies['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])]
    print(f"终止状态试验数量: {len(terminated_studies)}")
    
    # 合并到特征矩阵
    print("\n合并特征数据和标签...")
    merged = features.merge(studies[['nct_id', 'label']], on='nct_id', how='inner')
    
    print(f"合并后数据: {merged.shape[0]} 行, {merged.shape[1]} 列")
    
    # 检查合并后的标签分布
    merged_label_counts = merged['label'].value_counts()
    merged_high_risk = merged_label_counts.get(1, 0)
    merged_low_risk = merged_label_counts.get(0, 0)
    
    print(f"\n合并后标签分布:")
    print(f"高风险试验: {merged_high_risk} ({merged_high_risk/len(merged)*100:.2f}%)")
    print(f"低风险试验: {merged_low_risk} ({merged_low_risk/len(merged)*100:.2f}%)")
    
    # 保存最终数据
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'glp1_18clinical_features_with_labels_correct.csv')
    merged.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ 带标签的特征数据已保存: {output_file}")
    
    # 保存标签定义说明
    label_description = {
        'label_definition_method': '基于真实试验结果',
        'high_risk_criteria': [
            '试验状态为 TERMINATED、WITHDRAWN 或 SUSPENDED',
            'why_stopped 字段包含安全性关键词'
        ],
        'safety_keywords': [
            'safety', 'adverse', 'toxicity', 'side effect', 'unsafe',
            'death', 'fatal', 'severe', 'serious', 'unethical', 'harm',
            'patient died', 'mortality', 'elevated liver enzymes',
            'pancreatitis', 'hypoglycemia', 'cardiovascular', 'stroke',
            'hospitalization', 'withdrawn due to ae'
        ],
        'data_statistics': {
            'total_studies': int(total_count),
            'high_risk_studies': int(high_risk_count),
            'low_risk_studies': int(low_risk_count),
            'high_risk_percentage': float(high_risk_count/total_count*100),
            'why_stopped_fill_rate': float(why_stopped_fill_rate),
            'terminated_studies_count': int(len(terminated_studies)),
            'merged_total': int(len(merged)),
            'merged_high_risk': int(merged_high_risk),
            'merged_low_risk': int(merged_low_risk)
        }
    }
    
    label_info_file = os.path.join(output_dir, 'label_definition_correct.json')
    with open(label_info_file, 'w', encoding='utf-8') as f:
        json.dump(label_description, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 标签定义说明已保存: {label_info_file}")
    
    # 显示高风险试验的示例
    high_risk_examples = studies[studies['label'] == 1].head(5)
    if not high_risk_examples.empty:
        print("\n高风险试验示例:")
        for idx, row in high_risk_examples.iterrows():
            print(f"  - {row['nct_id']}: {row['overall_status']} - {row.get('why_stopped', 'N/A')}")
    
    return merged

def main():
    """主函数"""
    try:
        start_time = datetime.now()
        print(f"=== GLP-1临床试验风险标签定义（正确方法） ===")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 创建正确的标签
        df_with_labels = create_correct_labels()
        
        if df_with_labels is not None:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n✅ 正确的标签定义完成!")
            print(f"完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"处理耗时: {duration:.2f} 秒")
            
            # 显示数据概览
            print(f"\n数据概览:")
            print(f"总试验数: {len(df_with_labels)}")
            print(f"特征数: {len(df_with_labels.columns) - 2}")  # 减去nct_id和label
            print(f"高风险试验: {df_with_labels['label'].sum()}")
            print(f"低风险试验: {len(df_with_labels) - df_with_labels['label'].sum()}")
            
            # 显示前5个高风险试验
            high_risk_trials = df_with_labels[df_with_labels['label'] == 1].head()
            if not high_risk_trials.empty:
                print(f"\n前5个高风险试验的nct_id:")
                print(high_risk_trials['nct_id'].tolist())
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 标签定义失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 