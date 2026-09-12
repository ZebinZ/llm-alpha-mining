# 复现与接入指南

[English](reproducibility.md) | **简体中文** · [README](../README.zh-CN.md)

## 运行与验证

按照 [README](../README.zh-CN.md) 安装。项目采用 `src/` 布局：即使位于仓库根目录，也需要先安装再导入 `llm_alpha_mining`。

```bash
alpha-demo --output outputs/run-a
alpha-demo --output outputs/run-b
alpha-demo --output outputs/run-a --verify
```

在软件环境和随机种子相同的情况下，两份 `report.json` 的 `scientific_signature` 应一致。调用账本包含时间戳，因此整个输出目录不要求逐字节相同。验证会拒绝被修改或缺失的文件，但不判断统计有效性。

安装后的等价模块命令是 `python -m llm_alpha_mining.demo.run`。使用 `alpha-demo --help` 查看选项。

## 检查研究工作区

工作区命令冻结协议和文件身份，并创建持久化运行登记：

```bash
alpha-workspace init --workspace outputs/research-workspace
alpha-workspace status --workspace outputs/research-workspace
alpha-workspace verify --workspace outputs/research-workspace
```

这些命令只初始化和检查工作区，不启动搜索批次。程序化编排示例见[多轮搜索测试](../tests/test_multigeneration_campaign.py)和[可恢复执行测试](../tests/test_resumable_generation_executor.py)。

## 接入真实模型

示例明确使用 `FakeTransport`。真实运行器可以构造 `LiveProviderConfig` 和 `LiveStructuredTransport`，再将传输对象注入 `StructuredCallExecutor`。[传输实现](../src/llm_alpha_mining/research/agents/live_transport.py)和[离线传输测试](../tests/test_live_llm_transport.py)提供完整配置示例。

配置包括 HTTPS 域名允许列表、凭证环境变量名、模型标识、策略绑定、响应限制和成本限制。接入前应确认供应商 API 兼容性、模型可用性和价格。示例模型名与费用是虚构的。凭证放在环境变量中，不写入候选定义或日志。离线 CI 不验证真实供应商兼容性。

## 接入自己的数据

1. 定义数据结构、频率、可用性规则和不可变快照。
2. 使用有权使用的数据源构造 `DataBatch`。
3. 提供证券身份、状态变化、信号时点的股票范围和可用时间。
4. 通过 `factor_spec_from_candidate` 绑定已接受候选，保留缺失值和预热窗口。
5. 正式评估前冻结标签、时间划分、成本和筛选规则。
6. 导出数值和清单，将独立评估隔离于搜索反馈之外。

[示例运行器](../src/llm_alpha_mining/demo/run.py)展示从数据到因子的路径，[因子契约测试](../tests/test_factor_contracts.py)展示合法和非法绑定。合成数据中的完整性声明不能为真实供应商数据背书。

## 开发与打包

修改源码时使用 `python -m pip install -e '.[test]'` 安装，再运行 `python -m pytest -q`。发布检查应构建并安装 wheel，避免仅依赖可编辑安装：

```bash
python -m pip wheel --no-deps . --wheel-dir dist
python -m pip install --force-reinstall --no-deps dist/llm_alpha_mining-0.2.0-py3-none-any.whl
python -m pytest -q
```

CI 在 Python 3.12 和 3.13 上构建并安装 wheel，运行测试，从仓库目录外运行示例并验证文件，同时检查工作区初始化。当前结果见 [Actions](https://github.com/ZebinZ/llm-alpha-mining/actions/workflows/tests.yml)。

`requirements-validated.txt` 记录本地验收使用的直接依赖版本，不是完整的传递依赖锁文件。项目声明兼容版本范围；比较科学签名时应记录环境和随机种子。

公开版使用 `llm_alpha_mining.mining` 和 `llm_alpha_mining.research`，旧包路径不再提供兼容别名。序列化格式仍可能包含历史版本标识，原因见[架构说明](architecture.zh-CN.md#包结构与文件身份)。
