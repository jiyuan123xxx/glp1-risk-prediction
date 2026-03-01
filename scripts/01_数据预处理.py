#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLP-1临床试验风险预测特征构建 - 数据预处理模块

功能：
1. 加载原始AACT数据库文件
2. 数据清洗和预处理
3. 合并三个数据源
4. 处理缺失值和数据类型转换

作者：系统管理员
创建日期：2026-02-21
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_preprocess_data():
    """
    加载和预处理原始数据
    
    返回：
        pandas.DataFrame: 预处理后的数据框
    """
    print("开始数据预处理...")
    
    # 定义文件路径
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    studies_file = os.path.join(raw_data_dir, 'studies.txt')
    conditions_file = os.path.join(raw_data_dir, 'conditions.txt')
    eligibilities_file = os.path.join(raw_data_dir, 'eligibilities.txt')
    
    # 检查文件是否存在
    for file_path in [studies_file, conditions_file, eligibilities_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    print("1. 加载原始数据文件...")
    
    # 加载文件（注意分隔符为竖线 '|'）
    studies = pd.read_csv(studies_file, sep='|', dtype=str, low_memory=False)
    conditions = pd.read_csv(conditions_file, sep='|', dtype=str, low_memory=False)
    eligibilities = pd.read_csv(eligibilities_file, sep='|', dtype=str, low_memory=False)
    
    print(f"   - studies.txt: {studies.shape[0]} 行, {studies.shape[1]} 列")
    print(f"   - conditions.txt: {conditions.shape[0]} 行, {conditions.shape[1]} 列")
    print(f"   - eligibilities.txt: {eligibilities.shape[0]} 行, {eligibilities.shape[1]} 列")
    
    print("2. 选择需要的列...")
    
    # 选择需要的列
    studies_selected = studies[['nct_id', 'start_date', 'phase', 'enrollment', 
                               'brief_title', 'official_title']].copy()
    
    print("3. 处理疾病条件数据...")
    
    # 按 nct_id 分组，将所有疾病名称拼接成一个长字符串
    conditions_grouped = conditions.groupby('nct_id')['downcase_name'].apply(
        lambda x: ' '.join(x) if pd.notna(x).any() else ''
    ).reset_index()
    conditions_grouped.rename(columns={'downcase_name': 'conditions_text'}, inplace=True)
    
    print("4. 处理入排标准数据...")
    
    # eligibilities 保留 criteria 字段
    eligibilities_selected = eligibilities[['nct_id', 'criteria']].copy()
    
    print("5. 合并数据...")
    
    # 合并数据（左连接，保留所有试验）
    df = studies_selected.merge(conditions_grouped, on='nct_id', how='left')
    df = df.merge(eligibilities_selected, on='nct_id', how='left')
    
    print(f"   - 合并后数据: {df.shape[0]} 行, {df.shape[1]} 列")
    
    print("6. 处理缺失值和数据类型转换...")
    
    # 填充缺失值
    df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce').fillna(0)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['phase'] = df['phase'].fillna('').astype(str)
    df['conditions_text'] = df['conditions_text'].fillna('')
    df['criteria'] = df['criteria'].fillna('')
    df['brief_title'] = df['brief_title'].fillna('')
    df['official_title'] = df['official_title'].fillna('')
    
    print("7. 数据质量检查...")
    
    # 数据质量检查
    print(f"   - 唯一试验数量: {df['nct_id'].nunique()}")
    print(f"   - 有开始日期的试验: {df['start_date'].notna().sum()}")
    print(f"   - 有注册人数的试验: {df[df['enrollment'] > 0].shape[0]}")
    print(f"   - 有疾病信息的试验: {df[df['conditions_text'] != ''].shape[0]}")
    print(f"   - 有入排标准的试验: {df[df['criteria'] != ''].shape[0]}")
    
    # 检查重复的nct_id
    duplicate_nct = df['nct_id'].duplicated().sum()
    if duplicate_nct > 0:
        print(f"   ⚠️ 警告: 发现 {duplicate_nct} 个重复的nct_id")
        df = df.drop_duplicates(subset=['nct_id'], keep='first')
        print(f"   - 去重后数据: {df.shape[0]} 行")
    
    print("8. 保存预处理后的数据...")
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预处理后的数据
    output_file = os.path.join(output_dir, 'preprocessed_data.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ 数据预处理完成!")
    print(f"   - 输出文件: {output_file}")
    print(f"   - 最终数据维度: {df.shape}")
    
    return df

def main():
    """主函数"""
    try:
        start_time = datetime.now()
        print(f"=== GLP-1临床试验数据预处理 ===")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 执行数据预处理
        df = load_and_preprocess_data()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"处理耗时: {duration:.2f} 秒")
        print("=" * 50)
        
        # 显示数据概览
        print("\n数据概览:")
        print(f"总试验数: {len(df)}")
        print(f"特征数: {len(df.columns)}")
        print("\n前5个试验的nct_id:")
        print(df['nct_id'].head().tolist())
        
    except Exception as e:
        print(f"❌ 数据预处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()