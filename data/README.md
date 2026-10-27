# 输入数据

- `frozen_forecast.parquet`：最终两个模型原有的729016条历史预测，索引为datetime/instrument，列为score。
- `official_index_levels.csv`：本项目保存的官方沪深300价格/全收益指数水平。评价只使用H00300全收益列。

原始行情不随仓库分发。回测和新预测需要本地Qlib数据目录，包含`features/`、`calendars/day.txt`、`instruments/csi300.txt`。历史成分区间只用于确定当天股票池。

字段需要close/open/high/low/vwap/volume/amount/change/factor。Alpha158使用供应商复权价格，factor只用于Qlib交易单位换算，不重建分红或新增股份日程。本项目数据的volume单位为经复权转换的百股单位、amount为千元；换用其他来源前须先统一这些单位和复权定义。不要直接把adata的原始输出文件夹传给Qlib。

Qlib计算滚动特征需要请求日期以前的历史；预测输入应保留至少60个交易日的特征回看数据。模型不使用需要额外估计β的动态推理处理器。
