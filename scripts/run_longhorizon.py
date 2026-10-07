"""长时程基准 CLI — 连续任务压力协议（对比 full-history 与 status-first）。

用法：
    python scripts/run_longhorizon.py [--tasks 50] [--noise 2] [--k 5]
                                      [--window 128000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos_ark.longrun import run_long_horizon, synthesize_long_run  # noqa: E402
from mnemos_ark.memory import DLSMemory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--noise", type=int, default=2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--window", type=int, default=128000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = synthesize_long_run(n_tasks=args.tasks, noise_per_task=args.noise,
                                seed=args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        dls = DLSMemory(home=Path(tmp) / "dls")
        result = run_long_horizon(dls, tasks, k=args.k,
                                  window_tokens=args.window)
        dls.close()

    full = result["tokens_full_history"]
    sf = result["tokens_status_first"]
    print("\n# 长时程基准（连续任务压力协议）\n")
    print(f"任务数: {result['tasks']} | 检索 k={args.k} | "
          f"窗口: {result['window_tokens']} tok\n")
    print("| 策略 | 累计 token | 末任务上下文 | 峰值上下文 |")
    print("|---|---|---|---|")
    print(f"| full-history | {full['total']} | {full['end']} | {full['max']} |")
    print(f"| status-first | {sf['total']} | {sf['end']} | {sf['max']} |")
    overflow = result["window_overflow_at_task"]
    print(f"\nstatus-first 检索命中率: {result['hit_rate_status_first']:.1%}")
    print(f"token 消耗比（status-first / full-history）: "
          f"{result['savings_ratio']:.1%}")
    print(f"full-history 窗口溢出于第 {overflow} 个任务"
          if overflow else "full-history 未溢出窗口")
    print("\nfull-history 轨迹（采样）:", full["trajectory_sampled"])
    print("status-first 轨迹（采样）:", sf["trajectory_sampled"])
    print("\n" + json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
