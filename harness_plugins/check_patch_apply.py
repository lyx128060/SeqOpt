#!/usr/bin/env python3
"""单节点自检：这台机器能不能把 agent 的 C 补丁打到 HMMER 源码上？

为什么需要单独一个脚本：真正编译的是 **Slurm 计算节点**，而计算节点与登录节点的软件栈
并不一样 —— 目标集群上 `/usr/bin/git` 只有登录节点有，计算节点没有。所以"登录节点能过"
不等于"计算节点能过"。

脚本只做一件事：在**当前节点**上，用真实的补丁应用代码（`run_hmmer_cli._apply_patches`）
把内置样例补丁打到那个真实 HMMER 源文件的副本上，然后报告走了哪条路径。
不需要集群、不需要 LLM、不编译，几秒钟出结果。**退出码 0 = 补丁能打上。**

用法：

    # 登录节点
    python3 harness_plugins/check_patch_apply.py

    # 计算节点 —— 这才是关键的一次（$DEPLOY 是部署根，由登录节点展开）
    # 注意 srun 没有 --wrap（那是 sbatch 的选项），直接在选项后面跟命令
    srun --job-name=seqalign-patchcheck --time=2:00 --cpus-per-task=1 \\
         "$DEPLOY/.venv/bin/python" "$DEPLOY/harness_plugins/check_patch_apply.py"

    # 想留日志就换 sbatch：
    sbatch --job-name=seqalign-patchcheck --time=2:00 --cpus-per-task=1 \\
           --output=$DEPLOY/patchcheck-%j.out \\
           --wrap "$DEPLOY/.venv/bin/python $DEPLOY/harness_plugins/check_patch_apply.py"

输出里 `git = （本节点没有 git）` 且 `applier = python` 就是预期结果：
说明这台节点上编译不会再因为缺 git 而失败。
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

from seq_align import run_hmmer_cli  # noqa: E402
from seq_align.assembler import AssembledConfig  # noqa: E402
from seq_align.components.base import ComponentKind, ComponentSpec  # noqa: E402
from seq_align.components.c_kernels import ACTIVE_C_KERNEL_CONTRACTS  # noqa: E402

DEFAULT_PROPOSAL = ROOT / "examples" / "inner_proposal.example.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="检查当前节点能否应用 C 补丁")
    parser.add_argument("--proposal", default=str(DEFAULT_PROPOSAL),
                        help="含 unified_diff 的提案 JSON（默认用内置样例）")
    parser.add_argument("--hmmer-root", default=str(run_hmmer_cli.DEFAULT_HMMER_ROOT),
                        help="HMMER 源码树（默认就是 Slurm 路径上写死的那个位置）")
    args = parser.parse_args()

    proposal_path = Path(args.proposal)
    if not proposal_path.is_file():
        print(f"FAIL: 找不到提案文件 {proposal_path}")
        return 2
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))

    name = str(proposal.get("component_name") or "")
    contract = ACTIVE_C_KERNEL_CONTRACTS.get(name)
    if contract is None:
        print(f"FAIL: 提案里的 component_name={name!r} 不在活跃契约表里，"
              f"可选：{sorted(ACTIVE_C_KERNEL_CONTRACTS)}")
        return 2
    diff = str(proposal.get("unified_diff") or "")
    if not diff.strip():
        print("FAIL: 提案里没有 unified_diff")
        return 2

    source = Path(args.hmmer_root) / "src" / contract.c_target_file
    if not source.is_file():
        print(f"FAIL: 找不到 {source}（HMMER 源码树放对了吗？见部署说明 §1）")
        return 2

    print(f"host      = {socket.gethostname()}")
    print(f"python    = {sys.executable}  ({sys.version.split()[0]})")
    print(f"git       = {shutil.which('git') or '（本节点没有 git）'}")
    print(f"component = {name}  target={contract.c_target_file}")
    print(f"proposal  = {proposal_path}")
    print(f"source    = {source}")

    staging = Path(tempfile.mkdtemp(prefix="seqalign-patchcheck-"))
    try:
        build_root = staging / "hmmer-src"
        target_file = build_root / "src" / contract.c_target_file
        target_file.parent.mkdir(parents=True)
        shutil.copy2(source, target_file)
        before = target_file.read_bytes()

        cfg = AssembledConfig()
        cfg.c_patches = [
            ComponentSpec(
                contract_name=name,
                kind=ComponentKind.C_PATCH,
                c_patch=diff,
                c_target_file=contract.c_target_file,
                c_target_functions=list(contract.c_target_functions or []),
            )
        ]
        cfg.needs_recompile = True
        used = run_hmmer_cli._apply_patches(build_root, cfg)
        after = target_file.read_bytes()
    except Exception as exc:  # noqa: BLE001 - 自检脚本要把失败原因原样打出来
        print(f"FAIL: 补丁应用抛异常 —— {type(exc).__name__}: {exc}")
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"applier   = {used}   （git = 走 git apply；python = 走内置应用器）")
    print(f"size      = {len(before)} -> {len(after)} 字节")
    if after == before:
        print("FAIL: 补丁应用后文件内容没变")
        return 1
    print("OK: 这台节点能把 C 补丁打到 HMMER 源码上")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
