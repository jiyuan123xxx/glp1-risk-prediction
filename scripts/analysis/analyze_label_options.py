#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析标签定义方案的可行性
分析方案A和方案B的实际效果
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import os

def load_studies_data():
    """加载studies.txt数据"""
    studies_file = os.path.join('raw_data', 'studies.txt')
    
    if not os.path.exists(studies_file):
        print("❌ studies.txt文件不存在")
        return None
    
    print("正在加载studies.txt数据...")
    df = pd.read_csv(studies_file, sep='|', low_memory=False)
    print(f"加载成功: {len(df)} 行")
    
    return df

def extract_safety_keywords(df):
    """提取why_stopped中的高频安全性关键词"""
    print("\n" + "="*60)
    print("方案A分析：扩展安全性关键词列表")
    print("="*60)
    
    # 获取终止状态试验的why_stopped文本
    terminated_trials = df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])]
    why_texts = terminated_trials['why_stopped'].dropna().astype(str).str.lower()
    
    print(f"终止状态试验总数: {len(terminated_trials):,}")
    print(f"有why_stopped信息的试验数: {len(why_texts):,}")
    
    # 提取所有单词
    all_words = []
    for text in why_texts:
        words = re.findall(r'\b[a-z]+\b', text)
        all_words.extend(words)
    
    # 统计词频
    word_counter = Counter(all_words)
    
    # 预定义的安全性相关关键词种子
    safety_seed_words = [
        'safety', 'adverse', 'toxicity', 'side', 'effect', 'unsafe',
        'death', 'fatal', 'severe', 'serious', 'unethical', 'harm',
        'died', 'mortality', 'elevated', 'liver', 'enzymes',
        'pancreatitis', 'hypoglycemia', 'cardiovascular', 'stroke',
        'hospitalization', 'withdrawn', 'due', 'ae', 'event',
        'failure', 'injury', 'cardiac', 'arrest', 'heart', 'attack',
        'myocardial', 'infarction', 'suicide', 'suicidal', 'sepsis',
        'infection', 'gastrointestinal', 'bleeding', 'hemorrhage',
        'allergic', 'reaction', 'anaphylaxis', 'drug', 'interaction',
        'overdose', 'toxic', 'risk', 'danger', 'unsafe', 'harmful'
    ]
    
    # 找出与安全性相关的常见词
    safety_words = []
    for word, count in word_counter.most_common(200):  # 查看前200个高频词
        if word in safety_seed_words:
            safety_words.append((word, count))
    
    print(f"\n发现的安全性相关关键词 (前50个):")
    for i, (word, count) in enumerate(safety_words[:50]):
        percent = count / len(why_texts) * 100
        print(f"  {i+1:2d}. {word:20s}: {count:5d} ({percent:.1f}%)")
    
    # 分析当前关键词列表的效果
    current_keywords = [
        'safety', 'adverse', 'toxicity', 'side effect', 'unsafe',
        'death', 'fatal', 'severe', 'serious', 'unethical', 'harm',
        'patient died', 'mortality', 'elevated liver enzymes',
        'pancreatitis', 'hypoglycemia', 'cardiovascular', 'stroke',
        'hospitalization', 'withdrawn due to ae'
    ]
    
    # 测试当前关键词的召回率
    current_matches = 0
    for text in why_texts:
        for kw in current_keywords:
            if kw in text:
                current_matches += 1
                break
    
    current_recall = current_matches / len(why_texts) * 100
    print(f"\n当前关键词列表召回率: {current_recall:.1f}% ({current_matches}/{len(why_texts)})")
    
    # 建议扩展的关键词
    suggested_extensions = [
        'hepatic failure', 'renal failure', 'liver injury', 'kidney injury',
        'cardiac arrest', 'heart attack', 'myocardial infarction',
        'suicide', 'suicidal', 'sepsis', 'infection',
        'gastrointestinal bleeding', 'hemorrhage',
        'allergic reaction', 'anaphylaxis', 'drug interaction', 'overdose'
    ]
    
    # 测试扩展后的效果
    extended_keywords = current_keywords + suggested_extensions
    extended_matches = 0
    for text in why_texts:
        for kw in extended_keywords:
            if kw in text:
                extended_matches += 1
                break
    
    extended_recall = extended_matches / len(why_texts) * 100
    improvement = extended_recall - current_recall
    
    print(f"扩展关键词列表召回率: {extended_recall:.1f}% ({extended_matches}/{len(why_texts)})")
    print(f"召回率提升: +{improvement:.1f}%")
    
    return current_recall, extended_recall, safety_words

def analyze_missing_why_stopped(df):
    """分析缺失why_stopped的终止试验"""
    print("\n" + "="*60)
    print("方案B分析：处理缺失why_stopped的终止试验")
    print("="*60)
    
    # 获取终止状态试验
    terminated_trials = df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED'])]
    
    # 分析why_stopped缺失情况
    missing_why_stopped = terminated_trials[terminated_trials['why_stopped'].isna()]
    filled_why_stopped = terminated_trials[terminated_trials['why_stopped'].notna()]
    
    print(f"终止状态试验总数: {len(terminated_trials):,}")
    print(f"有why_stopped信息的试验: {len(filled_why_stopped):,} ({len(filled_why_stopped)/len(terminated_trials)*100:.1f}%)")
    print(f"缺失why_stopped信息的试验: {len(missing_why_stopped):,} ({len(missing_why_stopped)/len(terminated_trials)*100:.1f}%)")
    
    # 分析缺失why_stopped试验的特征
    print(f"\n缺失why_stopped试验的状态分布:")
    status_dist = missing_why_stopped['overall_status'].value_counts()
    for status, count in status_dist.items():
        percent = count / len(missing_why_stopped) * 100
        print(f"  {status}: {count:,} ({percent:.1f}%)")
    
    # 分析这些试验的其他特征（如果有的话）
    # 这里可以添加对其他字段的分析，如年份分布、试验阶段等
    
    # 方案B的影响分析
    print(f"\n方案B影响分析:")
    print(f"  如果剔除缺失why_stopped的试验，将损失: {len(missing_why_stopped):,} 个样本")
    print(f"  剩余样本数: {len(filled_why_stopped):,}")
    print(f"  样本损失比例: {len(missing_why_stopped)/len(terminated_trials)*100:.1f}%")
    
    return len(missing_why_stopped), len(filled_why_stopped)

def compare_schemes(current_recall, extended_recall, missing_count, filled_count):
    """比较两种方案的优缺点"""
    print("\n" + "="*60)
    print("方案比较分析")
    print("="*60)
    
    print("方案A (扩展关键词):")
    print(f"  ✅ 优势: 召回率从{current_recall:.1f}%提升到{extended_recall:.1f}% (+{extended_recall-current_recall:.1f}%)")
    print(f"  ✅ 优势: 不损失样本，保持{missing_count+filled_count:,}个终止试验")
    print(f"  ⚠️ 风险: 可能引入误报（将非安全性原因误判为安全性）")
    
    print("\n方案B (剔除缺失数据):")
    print(f"  ✅ 优势: 数据质量更高，避免未知风险")
    print(f"  ✅ 优势: 标签定义更严谨")
    print(f"  ⚠️ 风险: 损失{missing_count:,}个样本 ({missing_count/(missing_count+filled_count)*100:.1f}%)")
    print(f"  ⚠️ 风险: 可能漏掉真正因安全性终止但未记录原因的试验")
    
    print("\n混合方案 (A+B):")
    print(f"  ✅ 优势: 扩展关键词 + 剔除缺失数据")
    print(f"  ✅ 优势: 既提高召回率又保证数据质量")
    print(f"  ⚠️ 风险: 样本量进一步减少")

def generate_feasibility_report(current_recall, extended_recall, missing_count, filled_count, safety_words):
    """生成可行性报告"""
    print("\n" + "="*60)
    print("可行性报告")
    print("="*60)
    
    # 创建结果目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    report_content = f"""# 标签定义方案可行性报告

## 数据概览
- 终止状态试验总数: {missing_count+filled_count:,}
- 有why_stopped信息的试验: {filled_count:,} ({filled_count/(missing_count+filled_count)*100:.1f}%)
- 缺失why_stopped信息的试验: {missing_count:,} ({missing_count/(missing_count+filled_count)*100:.1f}%)

## 方案A分析 (扩展安全性关键词)

### 当前关键词效果
- 当前关键词召回率: {current_recall:.1f}%
- 扩展关键词召回率: {extended_recall:.1f}%
- 召回率提升: +{extended_recall-current_recall:.1f}%

### 发现的安全性关键词 (前20个)
"""
    
    # 添加关键词列表
    for i, (word, count) in enumerate(safety_words[:20]):
        percent = count / filled_count * 100
        report_content += f"- {word}: {count}次 ({percent:.1f}%)\n"
    
    report_content += f"""
## 方案B分析 (处理缺失数据)

### 样本影响
- 缺失why_stopped试验数: {missing_count:,}
- 如果剔除，样本损失比例: {missing_count/(missing_count+filled_count)*100:.1f}%
- 剩余样本数: {filled_count:,}

## 方案比较

### 方案A (推荐)
**优势**:
- 显著提高召回率 (+{extended_recall-current_recall:.1f}%)
- 不损失任何样本
- 实施简单，只需修改关键词列表

**风险**:
- 可能引入少量误报
- 需要人工审核扩展的关键词

### 方案B
**优势**:
- 数据质量更高
- 标签定义更严谨

**风险**:
- 损失{missing_count:,}个样本 ({missing_count/(missing_count+filled_count)*100:.1f}%)
- 可能漏掉真正因安全性终止但未记录原因的试验

## 推荐方案

**推荐采用方案A (扩展关键词)**，原因如下：

1. **召回率提升显著**: 从{current_recall:.1f}%提升到{extended_recall:.1f}%
2. **样本完整性**: 保持所有{missing_count+filled_count:,}个终止试验
3. **实施成本低**: 只需修改关键词列表
4. **风险可控**: 误报风险可通过关键词优化控制

### 实施建议
1. 采用扩展后的关键词列表
2. 对扩展的关键词进行人工审核
3. 在标签定义说明文档中记录优化过程
4. 后续可基于实际效果进一步优化关键词

## 结论

方案A在召回率和样本完整性方面具有明显优势，是当前最可行的优化方案。建议立即实施方案A，并持续监控模型性能。
"""
    
    # 保存报告
    report_file = os.path.join(results_dir, 'label_scheme_feasibility_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 可行性报告已保存: {report_file}")

def main():
    print("="*80)
    print("标签定义方案可行性分析")
    print("="*80)
    
    # 加载数据
    df = load_studies_data()
    if df is None:
        return
    
    # 分析方案A
    current_recall, extended_recall, safety_words = extract_safety_keywords(df)
    
    # 分析方案B
    missing_count, filled_count = analyze_missing_why_stopped(df)
    
    # 比较方案
    compare_schemes(current_recall, extended_recall, missing_count, filled_count)
    
    # 生成可行性报告
    generate_feasibility_report(current_recall, extended_recall, missing_count, filled_count, safety_words)
    
    print("\n" + "="*80)
    print("✅ 可行性分析完成!")
    print("="*80)

if __name__ == "__main__":
    main()