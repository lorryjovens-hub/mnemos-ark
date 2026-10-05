"""LongMemEval-V2 检索层评测 CLI。

用法：
    python scripts/run_longmemeval.py --dataset path/to/longmemeval.jsonl \\
        [--strategy lexical|hybrid] [--k 5] [--limit 50]

真实数据集：https://xiaowu0162.github.io/longmemeval-v2/ （下载 JSONL 后传入）。
仓库自带 tests/fixtures/longmemeval_sample.jsonl 可先跑通流程。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos_ark.benchmark import evaluate, load_longmemeval  # noqa: E402
from mnemos_ark.memory import DLSMemory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="LongMemEval JSONL 路径")
    parser.add_argument("--strategy", default="lexical", choices=["lexical", "hybrid"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="只评前 N 条（0=全部）")
    args = parser.parse_args()

    questions = load_longmemeval(args.dataset)
    if args.limit:
        questions = questions[:args.limit]

    with tempfile.TemporaryDirectory() as tmp:
        dls = DLSMemory(home=Path(tmp) / "dls")
        result = evaluate(dls, questions, k=args.k, strategy=args.strategy)
        dls.close()

    print("\n# LongMemEval 检索层评测（Mnemos Ark）\n")
    print(f"问题数: {result['questions']} | 策略: {result['strategy']} "
          f"| k={result['k']}")
    print(f"hit@{result['k']}: {result['hit_at_k']:.1%} | MRR: {result['mrr']:.3f}")
    print(f"token: 全量注入 {result['tokens_full_injection']} → "
          f"检索 {result['tokens_strategy']}（{result['token_ratio']:.1%}）")
    if result["by_type"]:
        print("\n| 题型 | 数量 | 命中 |")
        print("|---|---|---|")
        for qtype, bucket in sorted(result["by_type"].items()):
            print(f"| {qtype} | {bucket['n']} | {bucket['hit']} |")
    if result["degraded"]:
        print("\n[降级] hybrid 不可用时已回退词法（详见引擎降级事件）")
    print("\n" + json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
