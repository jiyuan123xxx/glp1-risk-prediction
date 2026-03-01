#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析studies.txt文件中的overall_status和why_stopped字段
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import sys

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_studies_data():
    """加载studies.txt数据"""
    studies_file = os.path.join('raw_data', 'studies.txt')
    
    if not os.path.exists(studies_file):
        print("❌ studies.txt文件不存在")
        return None
    
    print("正在加载studies.txt数据...")
    df = pd.read_csv(studies_file, sep='|', low_memory=False)
    print(f"加载成功: {len(df)} 行, {len(df.columns)} 列")
    
    return df

def analyze_overall_status(df):
    """分析overall_status字段"""
    print("\n" + "="*60)
    print("1. overall_status字段分析")
    print("="*60)
    
    # 检查字段是否存在
    if 'overall_status' not in df.columns:
        print("❌ overall_status字段不存在")
        return None
    
    # 统计状态分布
    status_counts = df['overall_status'].value_counts()
    status_percent = df['overall_status'].value_counts(normalize=True) * 100
    
    print(f"总试验数: {len(df):,}")
    print(f"overall_status字段填充率: {df['overall_status'].notna().mean():.2%}")
    print("\n状态分布统计:")
    
    for status, count in status_counts.items():
        percent = status_percent[status]
        print(f"  {status:20s}: {count:8,} ({percent:5.2f}%)")
    
    # 高风险状态分类
    high_risk_statuses = ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']
    low_risk_statuses = ['COMPLETED', 'ACTIVE_NOT_RECRUITING', 'RECRUITING', 'NOT_YET_RECRUITING']
    
    high_risk_count = df[df['overall_status'].isin(high_risk_statuses)].shape[0]
    low_risk_count = df[df['overall_status'].isin(low_risk_statuses)].shape[0]
    other_count = len(df) - high_risk_count - low_risk_count
    
    print(f"\n风险状态分类:")
    print(f"  高风险状态 ({', '.join(high_risk_statuses)}): {high_risk_count:,} ({high_risk_count/len(df)*100:.2f}%)")
    print(f"  低风险状态 ({', '.join(low_risk_statuses)}): {low_risk_count:,} ({low_risk_count/len(df)*100:.2f}%)")
    print(f"  其他状态: {other_count:,} ({other_count/len(df)*100:.2f}%)")
    
    return status_counts

def analyze_why_stopped(df):
    """分析why_stopped字段"""
    print("\n" + "="*60)
    print("2. why_stopped字段分析")
    print("="*60)
    
    # 检查字段是否存在
    if 'why_stopped' not in df.columns:
        print("❌ why_stopped字段不存在")
        return None
    
    # 统计填充情况
    why_stopped_filled = df['why_stopped'].notna().sum()
    why_stopped_fill_rate = df['why_stopped'].notna().mean()
    
    print(f"why_stopped字段填充率: {why_stopped_fill_rate:.2%}")
    print(f"有why_stopped信息的试验数: {why_stopped_filled:,}")
    
    # 分析终止状态试验的why_stopped
    terminated_trials = df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])]
    terminated_with_reason = terminated_trials[terminated_trials['why_stopped'].notna()]
    
    print(f"\n终止状态试验总数: {len(terminated_trials):,}")
    print(f"有终止原因的试验数: {len(terminated_with_reason):,}")
    print(f"终止状态试验的why_stopped填充率: {len(terminated_with_reason)/len(terminated_trials):.2%}")
    
    # 分析why_stopped内容模式
    if len(terminated_with_reason) > 0:
        why_stopped_texts = terminated_with_reason['why_stopped'].str.lower().dropna()
        
        # 关键词分析
        safety_keywords = ['safety', 'adverse', 'toxicity', 'side effect', 'unsafe', 'death', 'fatal', 'severe', 'serious', 'unethical', 'harm']
        enrollment_keywords = ['enrollment', 'recruitment', 'patient', 'subject']
        funding_keywords = ['funding', 'sponsor', 'financial', 'budget']
        efficacy_keywords = ['efficacy', 'effectiveness', 'benefit', 'futility']
        
        keyword_counts = {}
        for keyword_list, category in [(safety_keywords, '安全性'), 
                                      (enrollment_keywords, '入组问题'),
                                      (funding_keywords, '资金问题'),
                                      (efficacy_keywords, '疗效问题')]:
            count = why_stopped_texts.str.contains('|'.join(keyword_list), na=False).sum()
            keyword_counts[category] = count
        
        print(f"\n终止原因关键词分析:")
        for category, count in keyword_counts.items():
            percent = count / len(terminated_with_reason) * 100
            print(f"  {category}: {count} ({percent:.1f}%)")
    
    return why_stopped_fill_rate

def analyze_high_risk_trials(df):
    """分析高风险试验的终止原因"""
    print("\n" + "="*60)
    print("3. 高风险试验详细分析")
    print("="*60)
    
    # 高风险状态定义
    high_risk_statuses = ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']
    high_risk_trials = df[df['overall_status'].isin(high_risk_statuses)]
    
    print(f"高风险试验总数: {len(high_risk_trials):,}")
    
    # 按状态分类
    status_counts = high_risk_trials['overall_status'].value_counts()
    print("\n高风险试验状态分布:")
    for status, count in status_counts.items():
        percent = count / len(high_risk_trials) * 100
        print(f"  {status}: {count:,} ({percent:.1f}%)")
    
    # 分析有终止原因的高风险试验
    high_risk_with_reason = high_risk_trials[high_risk_trials['why_stopped'].notna()]
    print(f"\n有终止原因的高风险试验: {len(high_risk_with_reason):,}")
    
    if len(high_risk_with_reason) > 0:
        # 显示前10个高风险试验的终止原因
        print("\n高风险试验终止原因示例:")
        for i, (idx, row) in enumerate(high_risk_with_reason.head(10).iterrows()):
            print(f"  {i+1}. {row['nct_id']} - {row['overall_status']}: {row['why_stopped'][:100]}...")
        
        # 安全性相关的高风险试验
        safety_keywords = ['safety', 'adverse', 'toxicity', 'side effect', 'unsafe', 'death', 'fatal', 'severe', 'serious', 'unethical', 'harm']
        safety_related = high_risk_with_reason[high_risk_with_reason['why_stopped'].str.lower().str.contains('|'.join(safety_keywords), na=False)]
        
        print(f"\n安全性相关的高风险试验: {len(safety_related):,}")
        
        # 显示安全性相关试验示例
        if len(safety_related) > 0:
            print("安全性相关试验示例:")
            for i, (idx, row) in enumerate(safety_related.head(5).iterrows()):
                print(f"  {i+1}. {row['nct_id']}: {row['why_stopped']}")

def generate_visualizations(df, status_counts):
    """生成可视化图表"""
    print("\n" + "="*60)
    print("4. 生成可视化图表")
    print("="*60)
    
    # 创建结果目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. overall_status分布图
    plt.figure(figsize=(12, 8))
    status_counts.plot(kind='bar', color='skyblue')
    plt.title('临床试验状态分布 (overall_status)', fontsize=16, fontweight='bold')
    plt.xlabel('试验状态', fontsize=12)
    plt.ylabel('试验数量', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'overall_status_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ overall_status分布图已保存")
    
    # 2. 高风险状态分析
    high_risk_statuses = ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']
    low_risk_statuses = ['COMPLETED', 'ACTIVE_NOT_RECRUITING', 'RECRUITING', 'NOT_YET_RECRUITING']
    
    high_risk_count = df[df['overall_status'].isin(high_risk_statuses)].shape[0]
    low_risk_count = df[df['overall_status'].isin(low_risk_statuses)].shape[0]
    other_count = len(df) - high_risk_count - low_risk_count
    
    labels = ['高风险状态', '低风险状态', '其他状态']
    sizes = [high_risk_count, low_risk_count, other_count]
    colors = ['lightcoral', 'lightgreen', 'lightgray']
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('临床试验风险状态分布', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(results_dir, 'risk_status_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 风险状态分布图已保存")

def generate_report(df):
    """生成分析报告"""
    print("\n" + "="*60)
    print("5. 生成分析报告")
    print("="*60)
    
    # 创建结果目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成报告内容
    report_content = """# studies.txt文件分析报告

## 数据概览
- 总试验数: {total_trials:,}
- overall_status字段填充率: {status_fill_rate:.2%}
- why_stopped字段填充率: {why_stopped_fill_rate:.2%}

## overall_status分析

### 状态分布
{status_table}

### 风险分类
- 高风险状态 (TERMINATED, WITHDRAWN, SUSPENDED): {high_risk_count:,} ({high_risk_percent:.2f}%)
- 低风险状态 (COMPLETED, ACTIVE_NOT_RECRUITING, RECRUITING, NOT_YET_RECRUITING): {low_risk_count:,} ({low_risk_percent:.2f}%)
- 其他状态: {other_count:,} ({other_percent:.2f}%)

## why_stopped分析

### 填充情况
- 有why_stopped信息的试验数: {why_stopped_filled:,}
- 终止状态试验总数: {terminated_total:,}
- 有终止原因的试验数: {terminated_with_reason:,}
- 终止状态试验的why_stopped填充率: {terminated_fill_rate:.2%}

## 结论

1. **数据质量**: overall_status字段填充良好，why_stopped字段填充率较低
2. **风险分布**: 高风险试验占比较低，符合临床试验安全性监管的实际情况
3. **终止原因**: 安全性相关的终止原因需要重点关注，这是风险预测的重要信号

""".format(
        total_trials=len(df),
        status_fill_rate=df['overall_status'].notna().mean(),
        why_stopped_fill_rate=df['why_stopped'].notna().mean(),
        status_table='\n'.join([f"- {status}: {count:,} ({count/len(df)*100:.2f}%)" 
                              for status, count in df['overall_status'].value_counts().items()]),
        high_risk_count=df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])].shape[0],
        high_risk_percent=df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])].shape[0]/len(df)*100,
        low_risk_count=df[df['overall_status'].isin(['COMPLETED', 'ACTIVE_NOT_RECRUITING', 'RECRUITING', 'NOT_YET_RECRUITING'])].shape[0],
        low_risk_percent=df[df['overall_status'].isin(['COMPLETED', 'ACTIVE_NOT_RECRUITING', 'RECRUITING', 'NOT_YET_RECRUITING'])].shape[0]/len(df)*100,
        other_count=len(df) - df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])].shape[0] - df[df['overall_status'].isin(['COMPLETED', 'ACTIVE_NOT_RECRUITING', 'RECRUITING', 'NOT_YET_RECRUITING'])].shape[0],
        other_percent=(len(df) - df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])].shape[0] - df[df['overall_status'].isin(['COMPLETED', 'ACTIVE_NOT_RECRUITING', 'RECRUITING', 'NOT_YET_RECRUITING'])].shape[0])/len(df)*100,
        why_stopped_filled=df['why_stopped'].notna().sum(),
        terminated_total=df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])].shape[0],
        terminated_with_reason=df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED']) & df['why_stopped'].notna()].shape[0],
        terminated_fill_rate=df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED']) & df['why_stopped'].notna()].shape[0] / df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])].shape[0] if df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])].shape[0] > 0 else 0
    )
    
    # 保存报告
    report_file = os.path.join(results_dir, 'studies_status_analysis_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 分析报告已保存: {report_file}")

def main():
    print("="*80)
    print("studies.txt文件 - overall_status和why_stopped字段分析")
    print("="*80)
    
    # 加载数据
    df = load_studies_data()
    if df is None:
        return
    
    # 分析overall_status
    status_counts = analyze_overall_status(df)
    
    # 分析why_stopped
    why_stopped_fill_rate = analyze_why_stopped(df)
    
    # 分析高风险试验
    analyze_high_risk_trials(df)
    
    # 生成可视化图表
    if status_counts is not None:
        generate_visualizations(df, status_counts)
    
    # 生成分析报告
    generate_report(df)
    
    print("\n" + "="*80)
    print("✅ 分析完成!")
    print("="*80)

if __name__ == "__main__":
    main()