# 作业 1-2：转账工具治理 专属规范

> 本文件只适用于 `1-2-assignment/`。**仓库级通用规则见根目录 `AGENTS.md`，先读那份。**
> 开工前两份都要读：根目录讲"怎么做事"，本文件讲"这个作业的具体事实与坑"。

---

## 0. 本作业命令与数字

```bash
uv run python -m pytest 1-2-assignment/test_transfer.py -q    # 测试（当前 25 项）
uv run python 1-2-assignment/demo.py                           # 十条治理轨迹 + 副作用统计
```

> ⚠️ 必须在**仓库根目录**执行（`demo.py` / `test_transfer.py` 均以 `sys.path` + 顶层模块名导入）。
> `demo.py` 会真实 sleep 约 2s（大额超时场景），不是卡死。

**本作业未安装** ruff / mypy。**测试数量一变，同步更新此处 + `doc/02-design.md` §7。**

---

## 1. 本作业的文件地图

| 文件 | 职责 |
|------|------|
| `governance.py` | 治理框架核心，从课程 `week02/2-4/tool_governance_demo.py` 迁移；**唯一改动是 `consume()` 错误文案参数化**（基线硬编码"退款"文案） |
| `transfer.py` | 转账工具四件套（`TransferArgs` / `check_transfer` / `transfer` / 策略注册）+ 模拟账务 + `build_governance()` 组合根 |
| `demo.py` | 十条治理轨迹演示 + 尾声副作用统计 |
| `test_transfer.py` | 25 项测试，全部打 `ToolRuntime.invoke()` 公共入口 |
| `doc/02-design.md`、`doc/README.md` | 设计与使用文档，**随代码同步** |

**迁移边界**：课程基线的订单域工具（`get_order` / `create_refund` / `run_shell`）不迁移，框架组件全量保留。

---

## 2. 本作业踩过的坑（详细记录，不要重复踩）

### 2.1 `from import` 不可变全局是快照，不是引用

`demo.py` 曾用 `from transfer import TRANSFER_EXECUTIONS` 打印尾声统计——**显示 0 而非 2**。
int 不可变，导入时绑定的是当时值的副本；handler 里 `global TRANSFER_EXECUTIONS; += 1`
改的是 `transfer` 模块命名空间，demo 的本地绑定**永不更新**。
（`TRANSFERS` 是 dict，可变共享引用，所以 `len(TRANSFERS)` 恰好没错——两处一起统计时更具迷惑性。）

**正确形态**：`import transfer as transfer_module`，读 `transfer_module.TRANSFER_EXECUTIONS`。
`test_transfer.py` 的 fixture 重置同理（monkeypatch 模块属性）。

### 2.2 测"审批一次性"必须让第二笔过得了预检

门禁顺序是 **precheck 在 approval consume 之前**。测审批重放时若第一笔转走 35000
（余额 50000→15000），第二笔重放会在 precheck 就被 `BUSINESS_RULE_DENIED` 拦截，
**根本走不到审批核销**，测不到一次性语义。要用小额（如 10000）保证第一笔成功后
余额仍覆盖第二笔。写测试前先想清楚"哪道门先拦"。

### 2.3 大额超时演示的顺序约束

`demo.py` 里 `call_03`（45000 超时演示）**必须在 `call_05`（35000 成功）之前**——
`call_05` 成功后余额只剩 15000，45000 过不了余额预检，到不了超时模拟分支，
超时演示直接变成预检拒绝。

### 2.4 测试 fixture 重置三件套

`ACCOUNTS` 必须**深拷贝**（handler 原地 mutate `balance`，浅拷贝跨用例污染）、
`TRANSFERS` 清空、`TRANSFER_EXECUTIONS` 归零——三者都经 `monkeypatch.setattr(transfer_module, ...)`
重置，不要 `from import` 后直接改（见 2.1）。

### 2.5 precheck 在锁外，handler 内二次校验是实际防线

框架锁只包 handler（`_execute`），`before_tool_call` 里的 precheck 在锁外。**结论分两态，别混着说**：

- **shipped 代码（模拟实现，handler 无挂起点）下竞态不可达**：第二笔总被锁外 precheck
  拦下（`executions == 1`，审计为 decision 阶段 deny），锁内二检一次都不执行。
- **handler 有挂起点（扣款前真实 I/O，接支付渠道后必然出现）即可达**：探针实测
  （`gather` 两笔 30000 + 扣款前 `sleep(0.05)`）第二笔 precheck 读旧余额放行 → 等锁 →
  锁内二检拦截，`executions == 2`；删掉二检则两笔都成功、余额 -10000。

**变异测试教训**：删二检 / 删写锁在旧版测试（只断"恰一笔成功"）下**全绿**——该断言被
串行路径同样满足。现测试 #18 以注入挂起点钉住两处护栏（删二检→红、删锁→`max_active == 2` 红），
但"handler 有挂起点"仍是测试构造的前提，改 handler 时须同步评估。详见 `doc/02-design.md` §5.9。

### 2.6 并发测试：gather 必要但不充分，测竞态必须注入挂起点

两层缺一不可：
- 顺序 `await` 两笔不构成并发（第一笔完整结束后第二笔才开始）——必须 `asyncio.gather` 驱动；
- **仅 gather 仍不构成并发**：shipped handler 无挂起点时，第一笔在第二笔 precheck 前
  完整跑完，gather 与顺序执行行为无异——旧版 #18 就是这样"绿灯通过却没测到任何并发性质"
  （把 gather 改回顺序 await 结果不变）。要真实打开竞态窗口，须在**扣款前**注入挂起点
  （真实 I/O 的等价形态），见 #18 的 `suspended_transfer` 包装写法（复用 shipped handler，不复制逻辑）。

### 2.7 断言数值前先验算——差点改坏正确文档

首轮跑测试时场景 2 断言 `acc_1002 == 36500.00` 失败（实际 **36200.00**）：
`1200 + 35000 = 36200`，写测试时把 36500 算错了；同时误判"设计文档算错"，
准备去"修"文档——幸好文档一直是对的，未遭修改。
**测试失败时先独立手算/拆解验算哪边正确，不要预设"文档错"或"代码错"**；
pytest 的 actual 值就是证据，先算清再动任何一边。

---

## 3. 本作业的语义契约与实测参考值

改 `transfer.py` / `governance.py` 的门禁、计数、超时逻辑后，逐项实测：

| 契约 | 期望 | 实测参考值 |
|------|------|-----------|
| 大额（>40000）超时恰好执行一次 | `TIMEOUT_UNKNOWN`，非幂等不盲重试 | `executions=1, completed=0`，审计 `duration_ms ≥ 1900` |
| 阈值边界（== 40000） | 严格大于才触发模拟延迟 | `OK`，`executions=1` |
| 误配 `max_retries=3` 的非幂等写 | 框架压回 0 | handler 恰好被调 1 次 |
| 并发双花（#18：注入挂起点 + gather 两笔 30000） | 恰一笔成功 + 写锁串行 | `["BUSINESS_RULE_DENIED", "OK"]`，余额 20000，`executions=2`，`max_active=1` |
| 审批一次性 | 重放同审批被拒 | 第二笔 `APPROVAL_REQUIRED`（见 2.2 的构造要求） |
| 审批四绑定（user/tenant/tool/TTL）+ 超时烧审批 | 盗用/过期/跨租户/签错工具均拒；超时后原参数重放拒 | 全部 `APPROVAL_REQUIRED` 且 handler 零调用（#21–#25，变异自检均红；tool_name 删除后 digest 兜底属有意纵深） |
| demo 尾声统计 | 三口径对账 | `transfer_executions=2 / completed_transfers=1 / audit_records=15` |

**两口径计数语义**：`TRANSFER_EXECUTIONS` = handler 被进入几次（含超时/二检拒绝）；
`len(TRANSFERS)` = 副作用落地几次。审计记录数 = decision 拒绝 1 条 / 到执行 2 条。

---

## 4. 工具治理契约

门禁顺序（框架钉死，不可乱序）：参数校验 → deny/enabled → plan → 白名单/RBAC →
precheck → approval consume → confirmation → allow。

| 工具 | effect / risk | 审批 | timeout | retries | 幂等 | 锁粒度（canonical_target） |
|------|--------------|------|---------|---------|------|--------------------------|
| `transfer` | WRITE / HIGH | 必须，绑定全部参数，一次性 | 2.0s | 0 | 否 | `tenant:from_account` |
| `get_account` | READ / LOW | 不需要 | 1s | 2 | 是 | `tenant:account_id` |

**超时模拟规则（作业指定）**：`amount > 40000` → `await asyncio.sleep(3.0)`，
配合 `timeout_seconds=2.0` 必然 `TIMEOUT_UNKNOWN`。三个数字是闭环，改一个要连着看另外两个。

---

## 5. 本作业提交前 Checklist

- [ ] `uv run python -m pytest 1-2-assignment/test_transfer.py -q` 全绿（当前 25 项）
- [ ] `uv run python 1-2-assignment/demo.py` 十条轨迹 + 尾声统计与 `doc/02-design.md` §6 一致
- [ ] 计数两口径语义未被破坏（`executions` vs `completed`，见第 3 节）
- [ ] 可变全局状态的读写都经 `transfer_module.`（见 2.1），无 `from import` 快照
- [ ] 改 handler 签名后 grep mock：`grep -rn "def fake_\|def mock_" 1-2-assignment/`
- [ ] 护栏代码（写锁 / 锁内二检 / `from == to` / consume 四绑定 + TTL）改动后做**变异自检**：
      删掉护栏跑测试，#18 / #20 / #21 / #23 / #25 必须变红（不变红=护栏无测试保护，见 §2.5）
- [ ] `doc/02-design.md`（模型代码块、测试项数、已知限制）+ `doc/README.md` + 本文件已同步
- [ ] `git diff` 自查：无残留调试 `print` / 注释掉的代码

---

## 6. 本作业已知遗留

- **全内存态**：账务、审批、审计、锁均为进程内数据结构，多 worker / 重启不共享，
  生产需换数据库。详见 `doc/02-design.md` 第 10 节。
- **等锁时间不占超时预算**：`asyncio.timeout` 在锁内（基线结构），锁排队久时
  handler 实际可用预算被压缩。接真实 I/O 时需在锁外重算 deadline。
- **超时烧审批**：`TIMEOUT_UNKNOWN` 后审批已核销（consume 先于执行），人工对账确认
  未执行后须重新签发审批——`demo.py` 的 `call_05` 正是用新审批 `approval_02`。
- **handler 二检与 precheck 非共享实现**：有意取舍（框架锁结构不动，基线 precheck 在锁外），
  代价是余额校验逻辑两处维护。详见 `doc/02-design.md` §5.9 与第 10 节限制 #8。
- **`consume()` 文案参数化是 `governance.py` 与课程基线的唯一差异**：对照基线排查问题时要记得这一点。
- **`status != "active"` 状态机分支无覆盖**：三个模拟账户均为 active、未构造冻结账户，
  删除该预检行 25 项仍绿（变异实测）。属"未启用分支"而非盲区，接真实账户体系时需补冻结/止付用例。
