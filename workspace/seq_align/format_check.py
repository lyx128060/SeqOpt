import json
import sys
import os
import re
import argparse

# 基线默认过滤阈值（和run_jack内置默认值保持一致）
BASE_F1, BASE_F2, BASE_F3 = 0.05, 0.015, 0.007

def check_param_file(file_path: str, skip_dup_check: bool) -> int:
    # 1. 校验文件名格式，仅允许 opt_config_round数字_second.json
    filename = os.path.basename(file_path)
    pattern = r"opt_config_round\d+_second\.json"
    if not re.fullmatch(pattern, filename):
        print(f"[ERROR] 文件名称不合法：{filename}，仅支持 opt_config_roundX_second.json")
        return 1

    # 2. 判断文件是否存在
    if not os.path.exists(file_path):
        print(f"[ERROR] 参数文件不存在：{file_path}")
        return 1

    # 3. 解析JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] JSON解析失败：{e}")
        return 1

    # 4. 必须包含filters字段（run_jack读取过滤参数）
    if "filters" not in cfg_data:
        print("[ERROR] 配置缺失filters字段，无法覆盖默认过滤阈值")
        return 1
    filters = cfg_data["filters"]
    required_filter_keys = ["F1", "F2", "F3"]
    missing = [k for k in required_filter_keys if k not in filters]
    if missing:
        print(f"[ERROR] filters缺失必填阈值：{missing}")
        return 1

    # 5. 阈值数值合法性校验（0~1之间浮点数）
    for key in ["F1", "F2", "F3"]:
        val = filters[key]
        if not isinstance(val, float) or not (0.0 <= val <= 1.0):
            print(f"[ERROR] {key} 必须是0~1之间浮点数，当前值：{val}")
            return 1

    # 新增：仅非首轮才拦截和基线完全相同的参数
    if not skip_dup_check:
        curr_f1 = filters["F1"]
        curr_f2 = filters["F2"]
        curr_f3 = filters["F3"]
        if curr_f1 == BASE_F1 and curr_f2 == BASE_F2 and curr_f3 == BASE_F3:
            print(f"[ERROR] 过滤参数与基线默认值完全相同(F1={BASE_F1},F2={BASE_F2},F3={BASE_F3})，禁止复用基线，请生成差异化阈值")
            return 1

    # F4可选，存在则校验
    if "F4" in filters and filters["F4"] is not None:
        f4_val = filters["F4"]
        if not isinstance(f4_val, float) or not (0.0 <= f4_val <= 1.0):
            print(f"[ERROR] F4 必须是0~1之间浮点数，当前值：{f4_val}")
            return 1

    print(f"[PASS] {filename} 参数格式校验全部通过")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_dup_check", action="store_true", help="首轮跳过基线参数重复拦截")
    parser.add_argument("file_path", help="待校验json配置文件路径")
    args = parser.parse_args()
    exit_code = check_param_file(args.file_path, args.skip_dup_check)
    sys.exit(exit_code)
