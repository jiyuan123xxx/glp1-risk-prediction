# studies.txt文件分析报告

## 数据概览
- 总试验数: 568,538
- overall_status字段填充率: 100.00%
- why_stopped字段填充率: 7.85%

## overall_status分析

### 状态分布
- COMPLETED: 310,933 (54.69%)
- UNKNOWN: 86,989 (15.30%)
- RECRUITING: 65,700 (11.56%)
- TERMINATED: 32,774 (5.76%)
- NOT_YET_RECRUITING: 26,038 (4.58%)
- ACTIVE_NOT_RECRUITING: 21,486 (3.78%)
- WITHDRAWN: 15,953 (2.81%)
- ENROLLING_BY_INVITATION: 4,999 (0.88%)
- SUSPENDED: 1,672 (0.29%)
- WITHHELD: 966 (0.17%)
- NO_LONGER_AVAILABLE: 511 (0.09%)
- AVAILABLE: 251 (0.04%)
- APPROVED_FOR_MARKETING: 234 (0.04%)
- TEMPORARILY_NOT_AVAILABLE: 32 (0.01%)

### 风险分类
- 高风险状态 (TERMINATED, WITHDRAWN, SUSPENDED): 50,399 (8.86%)
- 低风险状态 (COMPLETED, ACTIVE_NOT_RECRUITING, RECRUITING, NOT_YET_RECRUITING): 424,157 (74.60%)
- 其他状态: 93,982 (16.53%)

## why_stopped分析

### 填充情况
- 有why_stopped信息的试验数: 44,620
- 终止状态试验总数: 50,399
- 有终止原因的试验数: 44,620
- 终止状态试验的why_stopped填充率: 88.53%

## 结论

1. **数据质量**: overall_status字段填充良好，why_stopped字段填充率较低
2. **风险分布**: 高风险试验占比较低，符合临床试验安全性监管的实际情况
3. **终止原因**: 安全性相关的终止原因需要重点关注，这是风险预测的重要信号

