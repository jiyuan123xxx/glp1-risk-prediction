#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLP-1临床试验风险预测特征构建 - 特征工程模块

功能：
构建18个临床驱动的核心风险预测特征，包括：
1. 试验规模与统计效力特征
2. 药物研发时代与里程碑特征  
3. 试验阶段与监管风险特征
4. 适应症与目标人群风险特征
5. 入排标准与患者选择特征
6. 安全性文本信号强度特征
7. 交互特征

作者：系统管理员
创建日期：2026-02-21
"""

import pandas as pd
import numpy as np
import re
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def filter_metabolic_trials(df):
    """
    筛选 GLP-1、糖尿病或肥胖相关的临床试验
    
    参数:
        df: 预处理后的数据框
        
    返回:
        pandas.DataFrame: 筛选后的数据框
    """
    # GLP-1 相关药物关键词（根据干预措施或疾病描述）
    glp1_keywords = [
        'glp-1', 'glucagon-like peptide', 'glp1', 'glp 1',
        'semaglutide', 'ozempic', 'rybelsus', 'wegovy',
        'liraglutide', 'victoza', 'saxenda',
        'dulaglutide', 'trulicity',
        'exenatide', 'byetta', 'bydureon',
        'lixisenatide', 'adlyxin'
    ]
    
    # 糖尿病关键词
    diabetes_keywords = [
        'diabetes', 'type 2 diabetes', 't2dm', 'type 2 dm',
        'diabetes mellitus', 'dm'
    ]
    
    # 肥胖关键词
    obesity_keywords = [
        'obesity', 'overweight', 'weight loss', 'body weight',
        'bmi', 'body mass index', 'weight management'
    ]
    
    # 合并所有关键词
    all_keywords = glp1_keywords + diabetes_keywords + obesity_keywords
    
    # 确保conditions_text列存在且为字符串
    if 'conditions_text' not in df.columns:
        raise ValueError("数据框中缺少 'conditions_text' 列，无法进行筛选")
    
    # 填充空值并转换为小写
    conditions_text = df['conditions_text'].fillna('').str.lower()
    
    # 创建筛选掩码
    mask = conditions_text.str.contains('|'.join(all_keywords), case=False, na=False)
    filtered_df = df[mask].copy()
    
    print(f"原始试验数量: {len(df)}")
    print(f"筛选后试验数量: {len(filtered_df)}")
    print(f"筛选比例: {len(filtered_df)/len(df)*100:.2f}%")
    
    return filtered_df

def build_18_clinical_features(df):
    """
    构建18个临床驱动的核心风险预测特征
    
    参数：
        df: 预处理后的数据框
        
    返回：
        pandas.DataFrame: 包含18个特征的特征矩阵
    """
    print("开始构建18个临床风险预测特征...")
    
    # 创建特征数据框的副本
    features_df = df[['nct_id']].copy()
    
    print("1. 构建试验规模与统计效力特征...")
    
    # 1. enrollment_log - 注册人数对数变换（试验规模与统计效力特征）
    features_df['enrollment_log'] = np.log1p(df['enrollment'])
    
    print("2. 构建药物研发时代与里程碑特征...")
    
    # 2. start_year - 试验开始年份（药物研发时代与里程碑特征）
    # 修复缺失日期处理：使用默认年份2000填充缺失值
    features_df['start_year'] = df['start_date'].dt.year.fillna(2000)
    
    # 3. pre_semaglutide_era - 司美格鲁肽上市前时代（药物研发时代与里程碑特征）
    features_df['pre_semaglutide_era'] = (features_df['start_year'] < 2017).astype(int)
    
    # 4. post_semaglutide_era - 司美格鲁肽上市后时代（药物研发时代与里程碑特征）
    features_df['post_semaglutide_era'] = (features_df['start_year'] >= 2017).astype(int)
    
    print("3. 构建试验阶段与监管风险特征...")
    
    phase_low = df['phase'].str.lower()
    
    # 5. phase_Unknown - 试验阶段未知（试验阶段与监管风险特征）
    features_df['phase_Unknown'] = phase_low.str.contains('na|not applicable|early phase 1', na=False).astype(int)
    
    # 6. phase_PHASE4 - IV期上市后研究（试验阶段与监管风险特征）
    features_df['phase_PHASE4'] = phase_low.str.contains('phase 4', na=False).astype(int)
    
    print("4. 构建适应症与目标人群风险特征...")
    
    # 修复文本关键词匹配：先填充空值，避免NaN转字符串'nan'的误匹配
    cond = df['conditions_text'].fillna('').str.lower()
    title = (df['brief_title'].fillna('') + ' ' + df['official_title'].fillna('')).str.lower()
    
    # 7. is_obesity - 肥胖相关试验（适应症与目标人群风险特征）
    obesity_keywords = ['obesity', 'overweight', 'bmi', 'body mass index', 'weight management']
    features_df['is_obesity'] = cond.apply(
        lambda x: 1 if any(keyword in x for keyword in obesity_keywords) and x != '' else 0
    )
    
    # 8. is_t2d - 2型糖尿病试验（适应症与目标人群风险特征）
    t2d_keywords = ['type 2 diabetes', 't2dm', 'type 2 dm', 'diabetes mellitus type 2']
    features_df['is_t2d'] = cond.apply(
        lambda x: 1 if any(keyword in x for keyword in t2d_keywords) and x != '' else 0
    )
    
    # 9. is_weight_loss - 减重为主要终点的试验（适应症与目标人群风险特征）
    weight_loss_keywords = ['weight loss', 'body weight', 'weight reduction', 'weight management']
    features_df['is_weight_loss'] = title.apply(
        lambda x: 1 if any(keyword in x for keyword in weight_loss_keywords) and x != '' else 0
    )
    
    print("5. 构建入排标准与患者选择特征...")
    
    # 确保criteria列是字符串类型
    df['criteria'] = df['criteria'].astype(str)
    crit_low = df['criteria'].str.lower()
    
    # 10. exc_count - 排除标准数量（入排标准与患者选择特征）
    def count_exclusions(crit):
        """计算排除标准数量（改进版：避免高估）"""
        if pd.isna(crit) or crit == '':
            return 0
        
        # 多种排除标准标识
        exclusion_indicators = [
            'exclusion criteria:',
            'exclusion criteria',
            'exclusion:',
            'exclusion criteria\n',
            'exclusion criteria\r\n'
        ]
        
        for indicator in exclusion_indicators:
            if indicator in crit:
                try:
                    # 提取排除标准部分
                    excl_part = crit.split(indicator)[1]
                    
                    # 如果后面有包含标准，则截断
                    inclusion_indicators = ['inclusion criteria:', 'inclusion criteria']
                    for inc_indicator in inclusion_indicators:
                        if inc_indicator in excl_part:
                            excl_part = excl_part.split(inc_indicator)[0]
                    
                    # 多种计数方法
                    counts = []
                    
                    # 方法1: 按数字编号计数（最精确）
                    numbered_items = re.findall(r'\d+\.\s', excl_part)
                    counts.append(len(numbered_items))
                    
                    # 方法2: 按换行符计数
                    line_items = [line.strip() for line in excl_part.split('\n') if line.strip()]
                    counts.append(len(line_items))
                    
                    # 方法3: 按句号计数
                    sentence_items = [s.strip() for s in excl_part.split('.') if len(s.strip()) > 10]
                    counts.append(len(sentence_items))
                    
                    # 改进：优先使用数字编号计数，如果没有则使用中位数避免极端值
                    if counts[0] > 0:  # 如果有数字编号
                        return counts[0]
                    elif counts:
                        # 使用中位数而不是最大值，避免高估
                        return int(np.median(counts)) if counts else 0
                    else:
                        return 0
                    
                except:
                    return 0  # 出错时返回0而不是1
        
        return 0
    
    features_df['exc_count'] = df['criteria'].apply(count_exclusions)
    
    # 11. criteria_total_len - 入排标准总字符数（入排标准与患者选择特征）
    features_df['criteria_total_len'] = df['criteria'].str.len()
    
    # 12. mentions_bmi - 提及BMI
    features_df['mentions_bmi'] = crit_low.str.contains('bmi|body mass index', na=False).astype(int)
    
    # 13. mentions_contraindication - 提及禁忌症
    contraindications = ['medullary thyroid', 'mtc', 'men2', 'pancreatitis', 'gallbladder', 
                         'contraindication', 'contraindicated']
    features_df['mentions_contraindication'] = crit_low.apply(
        lambda x: 1 if any(k in x for k in contraindications) else 0
    )
    
    # 14. mentions_renal_cutoff - 提及肾功能阈值
    renal_pattern = r'egfr\s*[<≤]\s*45|creatinine\s*[>≥]\s*1\.5|renal impairment|kidney function'
    features_df['mentions_renal_cutoff'] = crit_low.str.contains(renal_pattern, na=False).astype(int)
    
    print("6. 构建安全性文本信号强度特征...")
    
    # 15. high_risk_term_count - 高风险术语计数（安全性文本信号强度特征）
    high_risk_words = ['serious', 'death', 'hospitalization', 'discontinue', 'withdrawal', 
                       'severe', 'fatal', 'adverse', 'toxicity', 'safety']
    features_df['high_risk_term_count'] = crit_low.apply(
        lambda x: sum(x.count(w) for w in high_risk_words) if pd.notna(x) else 0
    )
    
    # 16. risk_ratio - 风险比率（安全性文本信号强度特征）
    low_risk_words = ['well tolerated', 'safe', 'effective', 'benefit', 'stable', 'tolerability']
    low_count = crit_low.apply(
        lambda x: sum(x.count(w) for w in low_risk_words) + 1 if pd.notna(x) else 1
    )
    features_df['risk_ratio'] = features_df['high_risk_term_count'] / low_count
    
    print("7. 构建交互特征...")
    
    # 17. year_x_enrollment - 年份 × 注册人数对数（交互特征）
    features_df['year_x_enrollment'] = features_df['start_year'] * features_df['enrollment_log']
    
    # 18. enrollment_log_x_phase_Unknown - 注册人数对数 × 阶段未知（交互特征）
    features_df['enrollment_log_x_phase_Unknown'] = features_df['enrollment_log'] * features_df['phase_Unknown']
    
    print("8. 特征质量检查...")
    
    # 检查特征完整性
    expected_features = 18
    actual_features = len([col for col in features_df.columns if col != 'nct_id'])
    
    if actual_features == expected_features:
        print(f"✅ 成功构建 {actual_features} 个特征")
    else:
        print(f"⚠️ 特征数量不匹配: 期望 {expected_features}, 实际 {actual_features}")
    
    # 检查缺失值
    missing_info = features_df.isnull().sum()
    if missing_info.sum() == 0:
        print("✅ 所有特征无缺失值")
    else:
        print(f"⚠️ 发现缺失值: {missing_info[missing_info > 0].to_dict()}")
    
    return features_df

def save_features(features_df, output_dir):
    """保存特征矩阵"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整的特征矩阵
    output_file = os.path.join(output_dir, 'glp1_18clinical_features.csv')
    features_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 保存特征描述
    feature_descriptions = {
        'enrollment_log': '注册人数对数变换',
        'start_year': '试验开始年份',
        'pre_semaglutide_era': '2017年以前（司美格鲁肽上市前）',
        'post_semaglutide_era': '2017年及以后',
        'phase_Unknown': '试验阶段未知',
        'phase_PHASE4': 'IV期上市后研究',
        'is_obesity': '肥胖相关试验',
        'is_t2d': '2型糖尿病试验',
        'is_weight_loss': '减重为主要终点的试验',
        'exc_count': '排除标准数量',
        'criteria_total_len': '入排标准总字符数',
        'mentions_bmi': '提及BMI',
        'mentions_contraindication': '提及禁忌症',
        'mentions_renal_cutoff': '提及肾功能阈值',
        'high_risk_term_count': '高风险术语计数',
        'risk_ratio': '风险比率（高风险词频/低风险词频）',
        'year_x_enrollment': '年份 × 注册人数对数',
        'enrollment_log_x_phase_Unknown': '注册人数对数 × 阶段未知'
    }
    
    desc_df = pd.DataFrame({
        'feature_name': list(feature_descriptions.keys()),
        'description': list(feature_descriptions.values())
    })
    
    desc_file = os.path.join(output_dir, 'feature_descriptions.csv')
    desc_df.to_csv(desc_file, index=False, encoding='utf-8')
    
    return output_file, desc_file

def main():
    """主函数"""
    try:
        start_time = datetime.now()
        print(f"=== GLP-1临床试验特征工程 ===")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查预处理数据是否存在
        processed_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'data', 'processed', 'preprocessed_data.csv'
        )
        
        if not os.path.exists(processed_file):
            print("❌ 预处理数据不存在，请先运行01_数据预处理.py")
            sys.exit(1)
        
        # 加载预处理数据
        print("加载预处理数据...")
        df = pd.read_csv(processed_file)
        df['start_date'] = pd.to_datetime(df['start_date'])
        
        # 新增：筛选 GLP-1、糖尿病、肥胖相关试验
        print("\n开始筛选目标试验...")
        df = filter_metabolic_trials(df)
        
        # 如果筛选后数据为空，则退出
        if len(df) == 0:
            print("❌ 筛选后无有效试验，请检查关键词或数据。")
            sys.exit(1)
        
        # 构建特征
        features_df = build_18_clinical_features(df)
        
        # 保存特征
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
        output_file, desc_file = save_features(features_df, output_dir)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ 特征工程完成!")
        print(f"完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"处理耗时: {duration:.2f} 秒")
        print(f"特征文件: {output_file}")
        print(f"特征描述: {desc_file}")
        print(f"特征矩阵维度: {features_df.shape}")
        
        # 显示特征统计
        print("\n特征统计:")
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            if feature != 'nct_id':
                stats = features_df[feature].describe()
                print(f"{feature}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 特征工程失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()