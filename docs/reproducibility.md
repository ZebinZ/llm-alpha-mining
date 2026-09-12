# 复现与接入

## 离线复现

README 的安装和运行命令是维护入口。相同软件环境与随机种子下，两次新输出目录的 `report.json` 应具有相同 `scientific_signature`。调用账本包含运行时间，因此不要求整个目录逐字节相同。

`requirements-validated.txt` 记录本次本地验收使用的直接依赖版本，并不是包含所有间接依赖和哈希的锁文件。CI 使用 Python 3.12、3.13 和项目声明的兼容依赖范围；CI 只有实际运行之后才算通过。

可以不安装命令行入口，直接运行：

```bash
python -m alpha_demo.run --output outputs/another-demo
python -m alpha_demo.run --output outputs/another-demo --verify
```

## 接入真实模型

离线入口明确使用 `FakeTransport`。若要真实调用，可在自己的运行器中构造 `LiveProviderConfig` 和 `LiveStructuredTransport`，再传给 `StructuredCallExecutor`。配置需要 HTTPS 域名允许列表、环境变量名、供应商模型标识及成本和响应大小限制；具体字段见 `alpha_research/agents/live_transport.py`。

真实 API 的兼容性、当前可用模型和价格需在接入时确认。示例中的假模型标识和模拟计费不可用于真实服务。配置密钥时仅使用环境变量，不把密钥写进候选、日志或仓库。

## 接入研究数据

1. 创建数据 schema、频率、可用时间规则和不可变快照。
2. 从自己的合法数据来源实现适配器或构造 `DataBatch`。
3. 提供股票身份、状态变更、交易范围与数据可知时间。
4. 将候选绑定到数据及算子注册表后计算，保留预热期和缺失值。
5. 冻结标签、时间验证、成本和筛选规则后运行正式评估。
6. 导出工件及哈希；独立检验完成后再确定最终交付。

演示里虚拟数据的完整性声明只适用于生成的 fixture，不能复制为真实供应商数据已被验证的证明。

## 恢复原研究

公开仓库可重现框架行为和演示，不能单独重现私有数据上的全部结果。恢复原研究需要另行取得私有归档中的冻结目录、数据契约、面板、公式、运行回执和版本说明，并核对 SHA。不要把新的实验直接写进旧的冻结输出目录。
