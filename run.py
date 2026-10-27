"""Check, report, compare products, predict and backtest the frozen model."""
from pathlib import Path
import argparse
import hashlib
import json
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs/.matplotlib'))


def verify_files():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'File fingerprint mismatch: {name}')
    print(f"Frozen release verified: {len(manifest['files'])} files.")


def main():
    parser = argparse.ArgumentParser(description='冻结模型检查、预测与复权价格回测')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='核对随仓库提供的模型、代码与结果文件')
    report_parser = sub.add_parser('report', help='从已保存账户重算指标与曲线，不需要原始行情')
    report_parser.add_argument('--output', type=Path, required=True)
    peers_parser = sub.add_parser('peers', help='与五只沪深300增强基金比较，不需要原始股票行情')
    peers_parser.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('predict', help='使用本地Qlib数据生成预测')
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('backtest', help='重放2016—2025固定预测；不训练或调参')
    b.add_argument('--data-dir', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify_files()
    if args.command == 'check':
        return
    if args.command == 'peers':
        from src.peers import save_comparison
        metrics = save_comparison(ROOT, args.output)
        print(metrics.to_string(index=False))
        print(f'Comparison saved: {args.output}')
        return
    if args.command == 'report':
        if args.output.exists():
            raise FileExistsError('Output already exists.')
        import pandas as pd
        from src.metrics import save_results
        report = pd.read_csv(ROOT / 'results/daily_report.csv', index_col=0, parse_dates=True)
        metrics = save_results(ROOT, report, args.output)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    from src.model import init_data
    data = init_data(args.data_dir)
    if args.command == 'predict':
        if args.output.exists():
            raise FileExistsError('Output already exists.')
        from src.model import predict
        result = predict(ROOT, args.start, args.end)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output)
        print(f'Prediction saved: {len(result)} rows, {args.output}')
    else:
        from src.backtest import run
        from src.metrics import save_results
        report = run(ROOT, data, args.output)
        metrics = save_results(ROOT, report, args.output)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
