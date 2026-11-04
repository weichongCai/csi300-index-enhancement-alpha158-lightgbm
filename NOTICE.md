# 来源

框架与Alpha158表达式：[Microsoft Qlib](https://github.com/microsoft/qlib)，MIT许可；学习器：[LightGBM](https://github.com/microsoft/LightGBM)，MIT许可。本仓库通过依赖引用它们。

研究思路曾参考[QuantsPlaybook](https://github.com/hugo2046/QuantsPlaybook)。这里只保留最终自有研究实现，不包含研报PDF，也不声称完整复现其中任一研报。

基准为本项目保存的中证指数H00300历史全收益指数水平。原始股票行情、公司公告、终端数据及原研究的大型缓存不在仓库中分发。

产品对照的单位净值与现金分红来自公开基金历史净值，身份与年度收益参考管理人官网/年报；具体链接、采集时间和口径见[data/peers/README.md](data/peers/README.md)及sources.json。原始年报PDF未随仓库分发。

行业风格风险实验使用[noblechang/a-share-data FREE_DATA_v0](https://github.com/noblechang/a-share-data/releases/tag/FREE_DATA_v0)中的历史行业归档，文件哈希及可得时点规则见[实验报告](results/style_turnover_control/README.md)。原始第三方归档不随仓库分发；模型不是MSCI官方Barra。
