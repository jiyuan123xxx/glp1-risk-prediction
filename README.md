# 🧬 GLP-1临床试验风险预测系统

> 基于机器学习的GLP-1类药物临床试验成功率预测与风险评估平台

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](README.md)
[![Data](https://img.shields.io/badge/Data-80MB-brightgreen.svg)](data/)
[![Optimization](https://img.shields.io/badge/Optimization-94%25%20Reduction-success.svg)](README.md)

## 📖 项目简介

**GLP-1临床试验风险预测系统**是一个基于机器学习的智能分析平台，专门用于预测GLP-1受体激动剂类药物临床试验的成功风险。通过分析22,093个GLP-1相关临床试验记录，构建18个临床驱动的风险特征，为药物研发决策提供科学依据。

**数据优化成果**: 项目总数据从约1.36GB优化到约80MB（缩减94.2%），同时保持完整的分析能力。

### 🎯 核心价值

- **🔬 科学预测**: 基于真实临床试验数据的机器学习模型
- **📊 风险评估**: 识别高风险临床试验，降低研发失败率
- **💡 决策支持**: 为药物开发提供数据驱动的风险管控工具
- **🌐 开源透明**: 完整的可解释性分析和可视化展示

## 🚀 快速开始

### 环境要求

- **Python**: 3.10+
- **内存**: 推荐8GB以上（优化后要求降低）
- **系统**: Windows/Linux/macOS
- **存储**: 仅需约80MB空间

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-username/glp1-risk-prediction.git
cd glp1-risk-prediction

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows PowerShell
venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt
```

### 运行项目

#### 方式一：Jupyter Notebook（推荐用于学习和探索）

```bash
# 启动Jupyter
jupyter notebook

# 在浏览器中打开 http://localhost:8888
# 按顺序运行notebook：
# 1. 01_数据预处理_详细讲解.ipynb
# 2. 02_特征工程_详细讲解.ipynb
# 3. 03_特征分析与可视化_详细讲解.ipynb
# 4. 04_标签定义_正确方法_详细讲解.ipynb
# 5. 05_集成学习_详细讲解.ipynb
# 6. 06_仪表盘_改进版_详细讲解.ipynb
```

#### 方式二：命令行脚本（推荐用于生产环境）

```bash
# 按顺序运行处理脚本
python scripts/01_数据预处理.py
python scripts/02_特征工程.py
python scripts/03_特征分析与可视化.py
python scripts/04_标签定义_正确方法.py
python scripts/05_集成学习.py
python scripts/06_仪表盘_改进版.py
```

## 📊 项目架构

```
glp1-risk-prediction/
├── 📁 data/                    # 数据管理
│   ├── 📁 raw/                 # 原始临床试验数据 (44.5MB，优化后)
│   ├── 📁 processed/          # 处理后的结构化数据 (35.4MB，优化后)
│   └── 📁 external/           # 外部数据源（预留）
├── 📁 scripts/                 # 核心处理脚本
│   ├── 📁 analysis/           # 分析工具脚本
│   ├── 01_数据预处理.py       # 数据加载和清洗
│   ├── 02_特征工程.py         # 18个风险特征构建
│   ├── 03_特征分析与可视化.py # 特征探索分析
│   ├── 04_标签定义_正确方法.py # 风险标签定义
│   ├── 05_集成学习.py         # 模型训练和评估
│   └── 06_仪表盘_改进版.py    # 可视化仪表板
├── 📁 docs/                    # 项目文档
│   ├── 📁 api/                 # API文档（预留）
│   ├── 📁 tutorials/           # 教程文档
│   ├── architecture.md         # 系统架构说明
│   ├── feature_engineering.md # 特征工程详解
│   └── models.md              # 模型架构说明
├── 📁 results/                 # 输出结果
│   ├── 📁 models/              # 训练好的模型文件
│   ├── 📁 visualizations/      # 可视化图表 (18个图表)
│   └── 📁 reports/             # 分析报告
├── 📁 notebooks/               # Jupyter笔记本
├── 📁 assets/                  # 静态资源
├── 📄 requirements.txt         # Python依赖包列表
├── 📄 LICENSE                  # 开源许可证
└── 📄 README.md               # 项目说明（本文档）
```

## 📁 项目结构（优化版）

```
glp1-risk-prediction/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   │   ├── studies.txt     # 临床试验数据 (14.4MB，优化后)
│   │   ├── conditions.txt  # 适应症数据 (2.2MB，优化后)
│   │   └── eligibilities.txt # 入选标准数据 (27.9MB，优化后)
│   └── processed/          # 处理后的数据
│       ├── preprocessed_data.csv
│       ├── glp1_18clinical_features.csv
│       ├── glp1_18clinical_features_with_labels.csv
│       ├── feature_descriptions.csv
│       └── label_definition.json
├── scripts/                # Python脚本（15个）
│   ├── 01_数据预处理.py
│   ├── 02_特征工程.py
│   ├── 03_特征分析与可视化.py
│   ├── 04_标签定义_正确方法.py
│   ├── 05_集成学习.py
│   ├── 06_仪表盘_改进版.py
│   ├── analysis/           # 分析工具脚本
│   │   ├── analyze_conditions.py
│   │   ├── analyze_label_options.py
│   │   ├── analyze_studies_status.py
│   │   └── analyze_time_split.py
│   └── tools/              # 项目工具脚本
│       ├── check_duplicate_files.py
│       ├── cleanup_duplicates.py
│       ├── cleanup_project.py
│       ├── move_remaining_files.py
│       ├── organize_results.py
│       └── verify_filename_conventions.py
├── notebooks/              # Jupyter Notebook（6个）
│   ├── 01_数据预处理_详细讲解.ipynb
│   ├── 02_特征工程_详细讲解.ipynb
│   ├── 03_特征分析与可视化_详细讲解.ipynb
│   ├── 04_标签定义_正确方法_详细讲解.ipynb
│   ├── 05_集成学习_详细讲解.ipynb
│   └── 06_仪表盘_改进版_详细讲解.ipynb
├── models/                 # 训练好的模型（8个）
│   ├── best_model_pipeline.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── svm.pkl
│   ├── scaler.pkl
│   ├── standard_scaler.pkl
│   └── feature_names.csv
├── results/                # 分析结果（30个文件）
│   ├── visualizations/     # 可视化图表（15个）
│   │   ├── feature_correlation.png
│   │   ├── feature_distributions.png
│   │   ├── feature_importances_ensemble.png
│   │   ├── model_dashboard.png
│   │   ├── pdp_analysis_dashboard.png
│   │   ├── roc_curve_ensemble.png
│   │   ├── shap_summary_dashboard.png
│   │   ├── shap_summary_ensemble.png
│   │   └── 文件命名规范.md
│   └── reports/           # 分析报告（15个）
│       ├── model_performance_report.md
│       ├── model_performance_results.csv
│       ├── shap_analysis_results.csv
│       ├── 报告来源说明.md
│       └── 可视化图片解释说明.md
├── docs/                   # 项目文档
│   └── GLP1临床试验风险预测项目综合说明.md
├── requirements.txt        # 依赖包列表
├── README.md              # 项目说明
└── 项目综合状态报告.md    # 项目状态报告
```

### 🎯 文件组织规范

项目采用**统一文件命名规范**和**清晰目录结构**：

- **可视化文件命名**：`*_ensemble.png`（集成学习）、`*_dashboard.png`（仪表盘）
- **目录分类**：所有图片→`visualizations/`，所有报告→`reports/`
- **避免重名**：不同脚本生成的文件使用不同的命名约定

## 🔬 技术特色

### 🎯 18个临床风险特征

项目构建了18个临床驱动的风险特征，涵盖7个维度：

| 特征维度 | 特征数量 | 代表特征 | 临床意义 |
|----------|----------|----------|----------|
| **试验规模与统计效力** | 1 | `enrollment_log` | 试验规模与统计效力 |
| **药物研发时代** | 3 | `start_year`, `pre_semaglutide_era` | 研发技术水平和监管环境 |
| **试验阶段与监管风险** | 2 | `phase_Unknown`, `phase_PHASE4` | 试验阶段和监管要求 |
| **适应症与目标人群** | 3 | `is_obesity`, `is_t2d`, `is_weight_loss` | 治疗领域特异性 |
| **入排标准与患者选择** | 2 | `exc_count`, `criteria_total_len` | 患者选择严格程度 |
| **安全性文本信号** | 5 | `mentions_bmi`, `high_risk_term_count` | 安全性关注程度 |
| **交互特征** | 2 | `year_x_enrollment`, `enrollment_log_x_phase_Unknown` | 特征交互效应 |

### 🤖 集成学习模型

- **基础模型**: 逻辑回归、随机森林、XGBoost
- **集成策略**: 加权集成 + 堆栈集成
- **类别不平衡处理**: SMOTE过采样技术
- **性能指标**: AUC=0.6453，召回率=40%

### 🔍 可解释性分析

- **SHAP分析**: 全局特征重要性
- **LIME分析**: 局部样本解释
- **PDP分析**: 部分依赖图
- **中文可视化**: 完整的中文标签支持

## 📈 性能表现

### 模型性能指标

| 指标 | 随机分割 | 时间分割 | 说明 |
|------|----------|----------|------|
| **AUC** | 0.6453 | 0.6348 | 模型区分能力 |
| **召回率** | 0.4000 | 0.4286 | 高风险试验检测率 |
| **精确率** | 0.0056 | 0.0024 | 预测准确性 |
| **F1分数** | 0.0111 | 0.0047 | 综合性能 |

### 数据规模统计（优化后）

- **总试验数**: 22,093个GLP-1相关临床试验
- **高风险试验**: 73个 (0.40%)
- **特征数量**: 18个临床风险特征
- **数据总量**: 约80MB (50个文件)
- **优化效果**: 从约1.36GB缩减到80MB（缩减94.2%）

## 🛠️ 技术栈

| 类别 | 技术栈 | 版本 |
|------|--------|------|
| **编程语言** | Python | 3.10+
| **机器学习** | scikit-learn, XGBoost | 最新稳定版
| **数据处理** | pandas, numpy | 最新稳定版
| **可视化** | matplotlib, seaborn, plotly | 最新稳定版
| **可解释性** | SHAP, LIME | 最新稳定版
| **开发工具** | Jupyter, VS Code | - |

## 📚 文档资源

### 核心文档

- [📋 项目综合状态报告](项目综合状态报告.md) - 完整的项目状态评估
- [🏗️ 系统架构说明](docs/architecture.md) - 详细的技术架构设计
- [🔬 特征工程详解](docs/feature_engineering.md) - 18个风险特征说明
- [🤖 模型架构说明](docs/models.md) - 机器学习模型设计

### 分析报告

- [📊 试验状态分析报告](docs/studies_txt_analysis_report.md) - 临床试验数据分布分析
- [🎯 标签方案可行性分析](docs/label_scheme_feasibility_analysis_report.md) - 风险标签优化策略
- [⏰ 时间分割分析报告](docs/time_split_analysis_report.md) - 模型泛化能力评估

### 可视化说明

- [🖼️ 图片来源说明](results/visualizations/图片来源说明.md) - 18个可视化图表说明
- [📋 报告来源说明](results/reports/报告来源说明.md) - 分析报告生成说明

## 🎯 应用场景

### 药物研发企业
- **风险评估**: 预测GLP-1类药物临床试验成功率
- **决策支持**: 优化临床试验设计和执行策略
- **成本控制**: 降低研发失败带来的经济损失
- **快速部署**: 项目仅需80MB空间，可快速部署到任何环境

### 研究机构
- **方法研究**: 临床试验数据分析方法论
- **技术验证**: 机器学习在医疗领域的应用验证
- **学术研究**: 临床试验风险因素探索

### 监管机构
- **质量监控**: 临床试验质量和安全性评估
- **趋势分析**: 药物研发趋势和风险模式识别
- **政策制定**: 基于数据的监管政策优化

## 🤝 贡献指南

我们欢迎各种形式的贡献！请参考以下指南：

### 报告问题

如果您发现任何问题或有改进建议，请：
1. 查看[现有问题](https://github.com/jiyuan123/glp1-risk-prediction/issues)
2. 创建新的issue，详细描述问题
3. 提供复现步骤和环境信息

### 提交代码

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

### 代码规范

- 遵循PEP 8代码风格
- 添加适当的注释和文档
- 确保所有测试通过
- 更新相关文档

## 📄 许可证

本项目采用 **MIT 许可证** - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和研究人员：

- **数据提供**: AACT (Aggregate Analysis of ClinicalTrials.gov)
- **技术支持**: scikit-learn, XGBoost, SHAP等开源社区
- **研究支持**: 相关临床试验研究机构和专家

## ⚠️ 免责声明

**重要提示**: 本项目为研究用途，预测结果仅供参考，不构成医疗建议或投资建议。

- 预测结果基于历史数据，不保证未来准确性
- 使用者应自行验证结果的适用性
- 作者不对使用本项目产生的任何后果负责

---

## 📞 联系我们

如有任何问题或合作意向，请通过以下方式联系：

- **项目主页**: [GitHub Repository](https://github.com/jiyuan123/glp1-risk-prediction)
- **问题反馈**: [Issues](https://github.com/jiyuan123/glp1-risk-prediction/issues)
- **邮箱**: jiyuan@sdmcro.com

---

<div align="center">

**如果这个项目对您有帮助，请给个⭐️支持一下！**

[![Star History](https://api.star-history.com/svg?repos=jiyuan123/glp1-risk-prediction&type=Date)](https://star-history.com/#jiyuan123/glp1-risk-prediction&Date)

</div>