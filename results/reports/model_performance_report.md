# GLP-1 临床试验风险预测模型性能报告

**生成时间**: 2026-02-27 22:22:09

## 数据集概览
- 测试样本总数: 13499
- 高风险样本数: 30 (0.22%)
- 低风险样本数: 13469 (99.78%)

## 模型性能指标
| 指标 | 值 |
|------|-----|
| AUC | 0.6453 |
| PR-AUC | 0.0041 |
| 准确率 | 0.8416 |
| 精确率 | 0.0056 |
| 召回率 | 0.4000 |
| F1分数 | 0.0111 |

## 阈值优化
- 最优阈值（基于F1）: 0.8500
- 对应F1分数: 0.0111

## 混淆矩阵（默认阈值0.5）
```
         预测低风险  预测高风险
实际低风险    11349        2120
实际高风险       18          12
```

## 可解释性分析
- SHAP 全局解释图：`D:\projects\glp1-trial-risk-prediction\glp1-risk-prediction\results\shap_summary.png`
- LIME 局部解释（高风险样本）：`D:\projects\glp1-trial-risk-prediction\glp1-risk-prediction\results\lime_explanation.html`
- PDP 边际效应分析图：`D:\projects\glp1-trial-risk-prediction\glp1-risk-prediction\results\pdp_analysis.png`

## 解释与建议
- AUC = 0.645 表明模型具有一定的区分能力。
- 召回率 = 0.400 表示模型能识别 40.0% 的实际高风险试验。
- 精确率偏低，提示假阳性较多，可通过提高阈值牺牲部分召回率换取精确率。
- 类别极度不平衡（高风险仅 0.22%），建议持续关注召回率而非准确率。
