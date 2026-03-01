#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析临床试验适应症分布

功能：
1. 分析预处理数据中的疾病条件分布
2. 识别主要适应症类别
3. 统计各适应症的比例

作者：系统管理员
创建日期：2026-02-21
"""

import pandas as pd
import numpy as np
import os
import sys
from collections import Counter
import re

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_conditions_distribution():
    """分析疾病条件分布"""
    
    print("开始分析适应症分布...")
    
    # 加载预处理数据
    processed_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'processed_data', 'preprocessed_data.csv'
    )
    
    if not os.path.exists(processed_file):
        print("❌ 预处理数据不存在，请先运行01_数据预处理.py")
        return
    
    df = pd.read_csv(processed_file)
    
    print(f"总试验数: {len(df)}")
    print(f"有疾病信息的试验: {df[df['conditions_text'] != ''].shape[0]}")
    
    # 提取所有疾病条件
    all_conditions = []
    for conditions_text in df['conditions_text']:
        if pd.notna(conditions_text) and conditions_text != '':
            # 按空格分割疾病名称
            conditions = str(conditions_text).split()
            all_conditions.extend(conditions)
    
    print(f"\n总疾病条件词数: {len(all_conditions)}")
    
    # 统计词频
    word_freq = Counter(all_conditions)
    
    print("\n=== 前50个最常见疾病条件词 ===")
    for word, count in word_freq.most_common(50):
        percentage = (count / len(all_conditions)) * 100
        print(f"{word}: {count}次 ({percentage:.2f}%)")
    
    # 定义主要适应症类别
    condition_categories = {
        '心血管疾病': ['coronary', 'heart', 'cardiac', 'hypertension', 'blood pressure', 'stroke'],
        '肿瘤': ['cancer', 'tumor', 'neoplasm', 'carcinoma', 'leukemia', 'lymphoma'],
        '神经系统疾病': ['alzheimer', 'dementia', 'parkinson', 'epilepsy', 'migraine', 'stroke'],
        '代谢疾病': ['diabetes', 'obesity', 'metabolic', 'lipid', 'cholesterol'],
        '呼吸系统疾病': ['asthma', 'copd', 'pulmonary', 'respiratory', 'lung'],
        '消化系统疾病': ['liver', 'hepatic', 'gastrointestinal', 'ibd', 'crohn'],
        '感染性疾病': ['infection', 'sepsis', 'hiv', 'hepatitis', 'tuberculosis'],
        '精神疾病': ['depression', 'anxiety', 'schizophrenia', 'bipolar', 'mental'],
        '风湿免疫疾病': ['arthritis', 'rheumatoid', 'lupus', 'autoimmune', 'inflammatory'],
        '肾脏疾病': ['renal', 'kidney', 'nephrology', 'dialysis'],
        '骨科疾病': ['fracture', 'osteoporosis', 'bone', 'orthopedic'],
        '眼科疾病': ['eye', 'ocular', 'retinal', 'glaucoma'],
        '皮肤疾病': ['psoriasis', 'eczema', 'dermatology', 'skin'],
        '血液疾病': ['anemia', 'hemophilia', 'blood', 'coagulation']
    }
    
    # 统计每个类别的试验数量
    category_counts = {}
    
    for category, keywords in condition_categories.items():
        count = 0
        for conditions_text in df['conditions_text']:
            if pd.notna(conditions_text):
                text_lower = str(conditions_text).lower()
                if any(keyword in text_lower for keyword in keywords):
                    count += 1
        category_counts[category] = count
    
    print("\n=== 主要适应症类别分布 ===")
    total_trials = len(df)
    
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_trials) * 100
        print(f"{category}: {count}个试验 ({percentage:.2f}%)")
    
    # 分析GLP-1相关试验
    print("\n=== GLP-1相关试验分析 ===")
    
    # 定义GLP-1相关关键词
    glp1_keywords = [
        'glp-1', 'glp1', 'glucagon-like peptide-1', 'semaglutide', 'liraglutide',
        'dulaglutide', 'exenatide', 'lixisenatide', 'albiglutide', 'tirzepatide'
    ]
    
    glp1_trials = []
    for idx, row in df.iterrows():
        text_to_search = f"{row['brief_title']} {row['official_title']} {row['conditions_text']}".lower()
        if any(keyword in text_to_search for keyword in glp1_keywords):
            glp1_trials.append(row['nct_id'])
    
    print(f"GLP-1相关试验数量: {len(glp1_trials)}")
    print(f"GLP-1试验占比: {(len(glp1_trials) / total_trials) * 100:.4f}%")
    
    # 分析肥胖和糖尿病试验
    obesity_trials = df[df['conditions_text'].str.contains('obesity|overweight|bmi', case=False, na=False)].shape[0]
    diabetes_trials = df[df['conditions_text'].str.contains('diabetes', case=False, na=False)].shape[0]
    
    print(f"\n肥胖相关试验: {obesity_trials}个 ({(obesity_trials / total_trials) * 100:.2f}%)")
    print(f"糖尿病相关试验: {diabetes_trials}个 ({(diabetes_trials / total_trials) * 100:.2f}%)")
    
    # 保存分析结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存类别分布
    category_df = pd.DataFrame({
        'category': list(category_counts.keys()),
        'count': list(category_counts.values()),
        'percentage': [count/total_trials*100 for count in category_counts.values()]
    })
    
    category_df.to_csv(os.path.join(output_dir, 'condition_category_distribution.csv'), index=False)
    
    print(f"\n✅ 分析完成!")
    print(f"结果文件: {os.path.join(output_dir, 'condition_category_distribution.csv')}")

def main():
    """主函数"""
    try:
        analyze_conditions_distribution()
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()