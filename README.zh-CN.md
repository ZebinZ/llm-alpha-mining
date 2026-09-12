# LLM Alpha Mining

[English](README.md) | **简体中文**

[![Tests](https://github.com/ZebinZ/llm-alpha-mining/actions/workflows/tests.yml/badge.svg)](https://github.com/ZebinZ/llm-alpha-mining/actions/workflows/tests.yml)

**将大语言模型提出的研究假设，转化为受约束的因子定义、符合时点语义的信号和可复现研究文件的量化研究框架。**

项目关注的是如何让自动挖因子的过程可检查：提出了什么假设、当时能看到哪些数据、为什么接受这个候选，以及能否复现它的信号。语言模型负责提出和评审候选；确定性的代码负责表达式校验、风险否决、数据对齐、评估和来源追踪。

这个仓库保留了量化研究项目中可复用的框架，提供离线示例、251 项自动化测试和完整的中英文文档。研究数据和具体因子提交文件单独保存。项目使用语言模型辅助研究，不涉及训练基础大模型。

## 快速开始

使用 Python 3.12 或 3.13。安装需要下载依赖；示例运行时无需联网或 API 密钥。

```bash
git clone https://github.com/ZebinZ/llm-alpha-mining.git
cd llm-alpha-mining
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
alpha-demo --output outputs/demo
alpha-demo --output outputs/demo --verify
python -m pytest -q
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。每次运行示例都需要新的输出目录；`--verify` 用于检查已有运行。

示例生成 40 只虚构股票和一个虚构指数的 100 个交易日数据。预设的角色响应提出三条公式，其中一条被风险角色否决，另外两条进入真实的因子计算引擎。示例覆盖缺失观测、滚动窗口预热、可交易范围变化，以及非股票证券的排除。

预期结果：接受两条因子、网络调用次数为零、文件完整性验证返回 `PASS`。输出包括信号 Parquet 文件、描述性诊断、调用账本、`report.json` 和 SHA-256 `manifest.json`。这些结果验证合成数据上的软件行为，不代表投资表现。

## 框架能力

```mermaid
flowchart LR
    A[研究假设] --> B[结构化 LLM 候选]
    B --> C[DSL 校验与角色评审]
    C --> D[符合时点语义的因子计算]
    D --> E[时间验证与成本评估]
    E --> F[筛选与可复现研究文件]
```

| 能力 | 实现位置 | 可检查的证据 |
| --- | --- | --- |
| 提案、批评、风险、仲裁等结构化角色 | [挖掘协议与 LLM 调用](src/llm_alpha_mining/mining/llm) | [结构化 LLM 测试](tests/test_structured_llm.py) |
| 公式允许列表与未来信息限制 | [DSL 解释器](src/llm_alpha_mining/mining/dsl/interpreter.py) | [DSL 测试](tests/test_safe_dsl.py) |
| 区分证券身份、可用历史与信号时点的交易范围 | [因子引擎](src/llm_alpha_mining/research/factors/engine.py) | [因子契约测试](tests/test_factor_contracts.py) |
| 标签、时间划分与因子评估 | [研究评估](src/llm_alpha_mining/research/evaluation) | [标签与验证测试](tests/test_labels_and_validation.py)、[评估测试](tests/test_evaluation.py) |
| 组合约束、换手、成本与执行时序 | [回测引擎](src/llm_alpha_mining/research/backtest/engine.py) | [回测测试](tests/test_backtest.py) |
| 冻结候选集合与多重检验修正 | [显著性评估](src/llm_alpha_mining/research/robustness/significance.py) | [显著性测试](tests/test_significance.py) |
| 预算、受限反馈、持久化状态与检查点恢复 | [批次编排](src/llm_alpha_mining/mining/orchestration) | [多轮搜索测试](tests/test_multigeneration_campaign.py)、[恢复测试](tests/test_resumable_generation_executor.py) |

示例覆盖候选评审、信号计算、描述性诊断和文件验证。正式回测与批次编排有各自的测试。接入真实模型和数据需要编写集成运行器；示例不会启动真实挖掘任务。

## 目录结构

```text
src/llm_alpha_mining/
  mining/       # 候选协议、LLM 角色、DSL、状态与搜索批次
  research/     # 数据契约、因子、评估、组合与回测
  demo/         # 使用合成数据和预设响应的可复现示例
tests/          # 按行为和研究组件命名的测试
docs/           # 一一对应的中英文指南
```

当前目录保留一套维护中的实现，不包含历史研究运行器、供应商专用面板修复工具、未使用的模型训练分支、缓存和提交归档。序列化格式和算子的版本标识仍明确保留，以维持已有记录身份的含义。

## 进一步阅读

- [架构与研究设计](docs/architecture.zh-CN.md)：职责分工、时点语义、缺失数据和恢复机制。
- [复现与接入指南](docs/reproducibility.zh-CN.md)：运行示例、接入模型、自备数据和开发检查。
- [研究范围与证据](docs/project_scope.zh-CN.md)：项目展示的能力、原始研究的工作和仍未确认的结果。

源代码公开可读。目前尚未指定开源许可证，具体见[发布与使用范围](docs/project_scope.zh-CN.md#发布与使用范围)。
