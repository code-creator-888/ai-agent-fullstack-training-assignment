# 转账工具治理 — Tool Runtime 上的资金安全实践

在课程 week02 治理框架（`ToolRuntime`）上接入"转账"工具，演示 Agent 调用工具时的
资金安全治理：审批前置、参数绑定、一次性消费、超时治理、并发防护与审计脱敏。

> 设计方案详见 [02-design.md](02-design.md)。

## 核心特性

- **审批前置 + 参数逐字段绑定** — 转账必须先签发与参数完全一致的审批凭证，
  改一个字段（金额、账户、事由）即失效
- **审批一次性消费** — 审批在执行前核销，重放同审批被拒；超时烧审批后需重新签发
- **大额超时治理** — `amount > 40000` 触发 3s 模拟延迟，超过 2s 工具超时预算，
  返回 `TIMEOUT_UNKNOWN` 且**不自动重试**（非幂等写）
- **并发双花防护** — 框架写锁串行化 + handler 锁内余额二次校验（预检在锁外）
- **租户隔离与防枚举** — 账户按 `(tenant, account)` 复合键隔离，跨租户与不存在同响应
- **审计与脱敏** — 两阶段审计（decision + execution），邮箱脱敏，审计不记参数值只记键名

## 快速开始

```bash
# 在仓库根目录执行
uv sync
uv run python 1-2-assignment/demo.py                            # 十条治理轨迹演示
uv run python -m pytest 1-2-assignment/test_transfer.py -q      # 25 项测试
```

> `demo.py` 会真实 sleep 约 2s（大额超时场景），属预期行为。

## 演示轨迹解读

`demo.py` 输出 10 条 JSON 结果 + 1 条尾声统计：

| 轨迹 | 场景 | 结果 |
|------|------|------|
| call_01 | 读账户 | OK，邮箱脱敏为 `***@***` |
| call_02 | 未审批转账 | `APPROVAL_REQUIRED`（action=confirm），handler 零调用 |
| call_03 | 大额 45000（已审批） | `TIMEOUT_UNKNOWN`，恰好一次执行，禁止重试 |
| call_04 | 超时后对账 | 余额未变 → 确认转账未执行 |
| call_05 | 正常转账 35000（新审批） | OK，双向余额变动 50000→15000 / 1200→36200 |
| call_06 | 到账确认 | 查询转入方余额 36200 |
| call_07 | 伪造身份字段 | `INVALID_ARGUMENT`（`extra="forbid"`），结构化字段错误 |
| call_08 | 跨租户转出 | `BUSINESS_RULE_DENIED`，与不存在同响应（防枚举） |
| call_09 | 余额不足 | `BUSINESS_RULE_DENIED`（预检拦截，先于审批消耗） |
| call_10 | plan 模式写操作 | `PLAN_MODE_DENIED` |

尾声统计（副作用对账）：

```json
{"side_effects": {"transfer_executions": 2, "completed_transfers": 1}, "audit_records": 15}
```

- `transfer_executions`：handler 真实进入次数（1 次超时 + 1 次成功）
- `completed_transfers`：真正落账的转账数（仅成功那笔）
- 两口径分离回答"执行了几次"与"生效了几次"——超时路径计入前者、不计入后者

## 文件结构

```
1-2-assignment/
├── governance.py        # 治理框架（从课程 2-4 迁移；consume 文案参数化为唯一改动）
├── transfer.py          # 转账工具：参数契约 / 模拟账务 / 预检 / handler / 策略注册
├── demo.py              # 演示入口
├── test_transfer.py     # 25 项测试（全部打 ToolRuntime.invoke() 公共入口）
└── doc/
    ├── 02-design.md     # 设计方案
    └── README.md        # 本文件
```

## 注意事项

- **顺序敏感**：大额超时演示必须在成功转账之前（成功后余额不足，大额过不了预检）
- **审批签发**：`approvals.approve()` 需传入 `TransferArgs.model_validate(...)` 的
  实参模型——与调用参数逐字段绑定
- **模拟数据**：账户余额为内存常量（tenant_a: acc_1001=50000 / acc_1002=1200；
  tenant_b: acc_2001=80000），进程重启即复位
- 本作业不请求任何外部服务（含作业一的网关），全部在进程内完成
