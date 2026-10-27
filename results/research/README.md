# 冻结模型的研究诊断与固定线性参考

根据多因子研究问题，补充因子、模型、风险和执行证据。原LightGBM、配置、冻结预测、基础账户和3bp账户均保持不变。新Ridge只作为参考，不替换最终模型。**2016—2025是复用开发期，无新增独立样本外证明。**

学习入口：[研究流程与证据](../../docs/研究流程与证据.md)、[图中15题与项目讲解](../../docs/项目问答与讲解.md)、[158项表达式字典](../../docs/alpha158_dictionary.csv)。

## 主要结果

| 模型/成本 | 平均年度超额IR | 全期超额IR | 十年年化收益 | 最大绝对回撤 |
| --- | ---: | ---: | ---: | ---: |
| 原LightGBM，无滑点 | 1.532 | 1.489 | 18.01% | −30.50% |
| 原LightGBM，每边3bp成交价滑点 | **1.378** | **1.344** | **16.66%** | **−29.47%** |
| 固定Ridge，每边3bp成交价滑点 | 0.429 | 0.486 | 9.36% | −43.15% |

原始账户逐日2430天；因子诊断仅2424个五日标签成熟的信号日。两模型使用同样训练/验证特征、标签和交易规则，Ridge均值填补、标准化只拟合训练段，λ=1、z截断±5事先固定。LightGBM保留自身原处理流程。本轮比较支持保留现有模型，不能推广为树模型普遍胜出。

![同规则账户曲线](report/model_comparison.png)

## 因子有效性、冗余与不确定性

- 158项特征每日成对有效值PearsonIC、RankIC、非年化ICIR、分年、覆盖率与七项预先固定示例的五组前瞻均值；模型分数另列。原始符号保留，不根据全期结果翻向交易。
- LightGBM平均RankIC约0.0583，Ridge约0.0514；单因子CORR20约−0.0371。IC、ICIR、账户超额IR三者口径不同。
- 20日固定圆形块Bootstrap，19,999次，seed=20261027；158项双侧均值RankIC检验做BY/BHY校正，5%阈值下32项拒绝。只覆盖当前检验族，**不校正过去所有策略尝试，也不证明没有过拟合**。
- LightGBM3bp全期IR的块抽样95%区间约[0.780,1.925]。配对抽样的LightGBM减Ridge全期IR差值约0.858，区间[0.420,1.299]。这些是开发样本的描述性区间。
- 120个月末横截面相关矩阵显示多组近乎完全重复/相反的方向占比特征。报告相关性，未据此重新筛选或训练最终模型。

![日IC与滚动稳定性](report/factor_ic.png)

![五组前瞻均值，不是可交易收益曲线](report/factor_quintiles.png)

![158项月度相关矩阵](report/factor_correlation.png)

## 风险、成本、容量与失效状态

- 以实际LightGBM3bp订单和持仓分析：双边换手=买卖额之和/前净值，年均约44.86倍。年均日费用占比加总约4.48%，成交价滑点损失归因约1.35%；是归因加总，不是CAGR损失。
- 同一已成交路径2倍放大，约15.86%成交额所在股票日突破5%日量/额约束；5倍约47.48%。这是静态压力测试，**不是重跑放大资金账户，也不估计容量上限**。
- 使用前收盘持仓和前一日可知的beta、60日波动、252−21日动量、20日成交额对数；后三项参考当日等权沪深300横截面z分数。暴露覆盖不足98%留空；2,429个可对齐交易日中，beta/波动/动量/流动性有效日分别2,412/1,885/1,368/1,853。图中的60日均值要求至少40个有效日，覆盖率另画，缺口没有填0。
- 缺少完整历史行业、市值和指数权重，因此这些是风险代理，不是完整Barra或相对真实指数权重的行业风格归因。单市场描述性回归组合beta约1.061，不宣称截距是纯Alpha。
- 用前一日200日均线位置及60日波动25%阈值报告条件表现。各子集非连续、长度不同，不能据此生成虚构的子策略净值或直接修改择时。

![换手、持仓与风险暴露及覆盖](report/execution_risk.png)

## 文件与复算

| 文件 | 用途 |
| --- | --- |
| ../../research_config.json | 新诊断、Bootstrap和固定Ridge的参数；禁止自动晋级 |
| factors/daily_ic.csv、daily_rank_ic.csv、daily_pair_counts.csv | 每日统计及有效样本数 |
| factors/quintile_forward_means.csv | 原始日组均值、组内样本数；五日区间重叠，不能连乘 |
| factors/mean_*_correlation.csv | 月末样本平均相关矩阵 |
| factors/risk_proxy_panel.parquet | 当日历史成员的可观测代理和beta回退标记 |
| linear_reference/*_ridge.json、training.json | 两个固定线性模型及训练记录 |
| linear_reference/frozen_ridge_forecast.parquet | 与原729,016行索引一致的参考预测 |
| linear_reference/daily_report.csv、orders.parquet、account_checks.json | 实际3bp参考账户和核对记录 |
| execution/holdings.parquet | 实际LightGBM3bp日终持仓，用于前一日暴露及集中度 |
| report/ | 可重算表格与五张图；含配对差值、条件状态及容量压力 |
| source_provenance.json | 输入指纹、原训练特征指纹、800条原存储标签交叉核对 |

无原始行情也能重新计算以上报告：

```bash
python run.py check
python -m unittest discover -s tests
python run.py research-report --output runs/research-report
```

重新模拟固定Ridge的所有成交，需要原始Qlib行情：

```bash
python run.py research-backtest --data-dir /path/to/qlib_data --output runs/ridge-3bp
```

重新构造IC与风险输入还需要本地特征缓存：

```bash
python run.py research-build --data-dir /path/to/qlib_data --early-features /path/to/early.parquet --late-features /path/to/late.pkl --output runs/rebuilt-factors
```

early输入为第一折73,200行预测特征，late输入为第二折655,816行预测特征，均有158个原顺序列、`datetime,instrument`多重索引，数据指纹见source_provenance.json。只加载自己信任的本地pickle；大体积特征缓存和原始训练样本未随仓库分发。`research-build`构造统计至指定新目录；默认`research-report`读取随仓库冻结的factors输入，不会自动替换它们。完整重新取数、复建历史训练集并训练，不属于仅靠本仓库附件即可一键完成的范围。

src/linear_reference.py保留固定Ridge的流式训练和预测函数；模型JSON记录训练均值、标准差、系数及截距。训练样本行数、标签成熟截断和验证误差见training.json，原样本/特征指纹逐段核对；额外抽取800条原保存残差标签与新计算实现交叉核对，最大绝对误差小于3×10⁻¹⁷。Ridge完整账户另核对滑后成交价、费用、日量额上限、日现金、首日损失及信号领先一日。

方法依据与解释见学习文档。本轮未接入另类数据、未部署实盘、未新增完整风险优化器；这些内容仅以原理及下一轮数据/验证要求说明。
