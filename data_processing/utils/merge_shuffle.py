#!/usr/bin/env python3
import json
import random
import sys
from pathlib import Path
import glob

def expand_paths(inputs):
    paths = []
    for p in inputs:
        if p == "-":
            paths.append(p)
            continue
        # glob expand
        matches = glob.glob(p, recursive=True)
        if matches:
            for m in matches:
                mp = Path(m)
                if mp.is_dir():
                    # find json/jsonl files recursively
                    for f in mp.rglob("*"):
                        if f.suffix.lower() in (".json", ".jsonl"):
                            paths.append(str(f))
                else:
                    paths.append(str(mp))
        else:
            # direct path (may not exist; let later check handle)
            paths.append(p)
    return paths

def load_json_file(path):
    if path == "-":
        data = json.load(sys.stdin)
        return data
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    # choose loader: for .jsonl treat each line as json object -> return list
    if p.suffix.lower() == ".jsonl":
        items = []
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Invalid json on line {i} in {path}: {e}")
        return items
    else:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

# 新增：将主要逻辑封装为可直接调用的函数
def run_merge(input_paths, out_path, jsonl=False, seed=None):
    in_paths = expand_paths(input_paths)
    if not in_paths:
        print("No input files found.", file=sys.stderr)
        return 2

    # Check if all files exist before processing
    for p in in_paths:
        if p != "-" and not Path(p).exists():
            print(f"Error: Input file not found: {p}", file=sys.stderr)
            return 1

    merged = []
    for p in in_paths:
        try:
            data = load_json_file(p)
        except Exception as e:
            print(f"Error loading {p}: {e}", file=sys.stderr)
            return 3
        
        count = 0
        if isinstance(data, list):
            count = len(data)
            merged.extend(data)
        else:
            count = 1
            merged.append(data)
        print(f"Loaded {count} items from {p}")

    if seed is not None:
        random.seed(seed)
    random.shuffle(merged)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if jsonl:
        with out_path.open("w", encoding="utf-8") as f:
            for item in merged:
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    return 0

# main: 在此处直接写入输入输出配置（无需命令行）
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge and shuffle JSON/JSONL files")
    parser.add_argument("inputs", nargs="+", help="Input files/directories/globs")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    exit_code = run_merge(args.inputs, args.output, jsonl=args.jsonl, seed=args.seed)
    if exit_code == 0:
        print(f"Written merged output to {args.output}")
    else:
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
