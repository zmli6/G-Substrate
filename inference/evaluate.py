#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Evaluation Router (GraphAGI)
-----------------------------------
Supported tasks:
  - graph_search      → eval_graph_search
  - mol               → eval_molecular
  - event_graph       → evaluate_event_graph_original_metrics
  - ssg / scene_graph → PCIs + SGCls (via sgg_utils, PGSG-style)

Supported Formats:
  - Standard Task-Specific Formats (e.g., [obj]...[rel]...)
  - UGS (Unified Graph Structure): <graph><nodes>...</nodes><edges>...</edges></graph>
    * Automatically detected and converted for SGG and Event Graph tasks.
    * Robust parsing handles truncated or malformed XML/JSON.

For SGG:
  - 使用 vg150_gt.pkl + vg_metadata.json 加载 VG150 GT
  - VLM 生成 [obj]/[rel] 文本 → sgg_utils.process_text_prediction_item
    * 解析 triplets
    * SentenceTransformer closed-set 映射
    * 映射到 GT index 空间 (sub_idx, obj_idx, pred_id)
  - 通过 sgg_utils.eval_pcis / eval_sgcls 计算:
    * R@50 / R@100
    * mR@50 / mR@100
  - 完全对齐 PGSG 在 PCls / SGCls setting 下的评估协议
"""

import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import re
import os
import sys

# ========= SGG utils (PCIs + SGCls) =========
from evaluators.sgg_eval.vg_sgg_eval import (
    attach_vg_gt_to_record,
    process_text_prediction_item,
    eval_pcis,
    eval_sgcls,
    eval_pcis_original_class_level,
)

# ========= Non-SGG task evaluators =========
from evaluators.graph_search_eval import eval_graph_search
from evaluators.mol_eval import eval_molecular
from evaluators.event_graph_eval import evaluate_event_graph_original_metrics


SUBTASK_KEYWORDS = {
    "connectivity": ["path between", "is there a path between"],
    "cycle": [" cycle", " circuit", "is there a cycle", "contains a cycle", "has a cycle"],
    "shortest_path": ["shortest path", "shortest", "minimum distance", "min distance", "minimum cost", "with weight"],
    "max_flow": ["max flow", "maximum flow", "flow from", "capacity", "bottleneck"],
    "bipartite_matching": [
        "matching", "maximum matching", "perfect matching",
        "assign", "assignment", "assigned to",
        "one-to-one", "one to one",
        "can be assigned",
        "host is interested in task",
        "each host can solve only one task",
        "each task can be resolved by just one host",
        "job", "job applicant",
        "task assignment", "job assignment", "worker",
    ],
    "hamiltonian": [
        "hamilton", "hamiltonian",
        "visits every node exactly once",
        "visit each node exactly once",
        "visit every node",
    ],
}


EVENT_METRIC_KEYS = ["precision", "recall", "f1"]



# =========================================================
# UGS Parsing Helpers
# =========================================================
def is_ugs_format(text):
    """
    Check if the text contains UGS format indicators.
    Checks for <graph>, <nodes>, or <edges>.
    """
    if not isinstance(text, str):
        return False
    return "<graph>" in text or "<nodes>" in text or "<edges>" in text


def is_graph_facts_format(text):
    """
    Check if the text contains Graph Facts format indicators.
    Checks for [Graph Facts], entity(...), event(...), or relation(...).
    """
    if not isinstance(text, str):
        return False
    return "[Graph Facts]" in text or "entity(" in text or "event(" in text or "relation(" in text


def extract_json_objects(text):
    """
    Robustly extract JSON objects from text.
    Handles:
    - Multiple objects
    - Single quotes (by replacing with double quotes)
    - Flat structures primarily (using regex for robustness against broken nesting)
    """
    objs = []
    # Regex for flat JSON objects: { ... } without nested braces
    # This covers the standard UGS node/edge format
    candidates = re.findall(r'({[^{}]+})', text)
    for c in candidates:
        try:
            objs.append(json.loads(c))
        except:
            try:
                # Try fixing single quotes
                c_fixed = c.replace("'", '"')
                objs.append(json.loads(c_fixed))
            except:
                pass
    return objs


def parse_ugs_content(text):
    """
    Parses the UGS (Unified Graph Structure) XML-like format.
    Returns a list of nodes (dicts) and a list of edges (dicts).
    
    Strategy:
    1. Scan the entire text for JSON-like objects { ... }.
    2. Classify each object as Node or Edge based on keys.
       - Edge: has "src" and "dst"
       - Node: has "id" (and not src/dst)
    This ignores XML tags (<nodes>, <edges>) which makes it robust to:
    - Missing tags
    - Truncated tags
    - Wrong nesting
    """
    nodes = []
    edges = []
    
    all_objs = extract_json_objects(text)
    
    for obj in all_objs:
        # Classify based on keys
        if "src" in obj and "dst" in obj:
            edges.append(obj)
        elif "id" in obj:
            nodes.append(obj)
    
    # Deduplicate nodes by id just in case
    unique_nodes = {}
    for n in nodes:
        if "id" in n:
            unique_nodes[n["id"]] = n
    nodes = list(unique_nodes.values())

    return nodes, edges


def convert_ugs_to_sgg_text(text):
    """
    Converts UGS format to SGG text format: "sub [obj] rel [rel] obj [obj]"
    """
    nodes, edges = parse_ugs_content(text)
    
    # Map node id to label/name
    id_to_label = {}
    for n in nodes:
        if "id" in n:
            # If label exists use it, else use id as label (SGG case often uses id as class name)
            id_to_label[n["id"]] = n.get("label", n["id"])
            
    triplets = []
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        rel = e.get("relation")
        
        if src in id_to_label and dst in id_to_label and rel:
            sub_lbl = id_to_label[src]
            obj_lbl = id_to_label[dst]
            # Construct: "sub [obj] rel [rel] obj [obj]"
            t_str = f"{sub_lbl} [obj] {rel} [rel] {obj_lbl} [obj]"
            triplets.append(t_str)
            
    return " , ".join(triplets)


def convert_ugs_to_event_graph_text(text):
    """
    Converts UGS format to Event Graph text format.
    """
    nodes, edges = parse_ugs_content(text)
    
    lines = ["<event_graph>", "<Node>"]
    for n in nodes:
        nid = n.get("id")
        lbl = n.get("label")
        if nid and lbl:
            # Format: (E1, "co-headline|Cause_to_amalgamate")
            lines.append(f'({nid}, "{lbl}")')
    lines.append("</Node>")
    
    lines.append("<Edge>")
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        rel = e.get("relation")
        if src and dst and rel:
            # Format: (E10, BEFORE, E1)
            lines.append(f"({src}, {rel}, {dst})")
    lines.append("</Edge>")
    lines.append("</event_graph>")
    
    return "\n".join(lines)


def parse_graph_facts_sgg(text):
    """
    Parses SGG Graph Facts:
    entity(1, light)
    relation(1, on, 3)
    Returns SGG text format: "light [obj] on [rel] board [obj]"
    
    Robustness:
    - Handles incomplete lines or missing closing parenthesis.
    - Extracts as many valid entities and relations as possible.
    """
    id_to_label = {}
    # Parse entities: entity(id, label)
    # Regex: entity\s*\(\s*([^,]+)\s*,\s*([^)\n]+)
    # Matches "entity(1, light" even if ")" is missing
    ent_pattern = re.compile(r'entity\s*\(\s*([^,]+)\s*,\s*([^)\n]+)')
    for match in ent_pattern.finditer(text):
        nid, label = match.groups()
        # Clean up potential trailing chars if regex over-matched due to missing paren
        label = label.split(')')[0].strip()
        id_to_label[nid.strip()] = label
        
    triplets = []
    # Parse relations: relation(src, rel, dst)
    # Regex: relation\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)\n]+)
    rel_pattern = re.compile(r'relation\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)\n]+)')
    for match in rel_pattern.finditer(text):
        src, rel, dst = match.groups()
        src = src.strip()
        rel = rel.strip()
        # Clean up potential trailing chars
        dst = dst.split(')')[0].strip()
        
        if src in id_to_label and dst in id_to_label:
            sub_lbl = id_to_label[src]
            obj_lbl = id_to_label[dst]
            triplets.append(f"{sub_lbl} [obj] {rel} [rel] {obj_lbl} [obj]")
            
    return " , ".join(triplets)


def parse_graph_facts_event(text):
    """
    Parses Event Graph Facts:
    event(E1, "trigger|type")
    relation(E1, BEFORE, E11)
    Returns Event Graph text format (XML-like representation used in eval).
    
    Robustness:
    - Handles incomplete lines or missing closing parenthesis.
    - Extracts as many valid events and relations as possible.
    """
    lines = ["<event_graph>", "<Node>"]
    
    # Parse events: event(id, "content")
    # Regex: event\s*\(\s*([^,]+)\s*,\s*"?([^")\n]+)
    # Matches "event(E1, "trigger|type" even if closing quote/paren missing
    evt_pattern = re.compile(r'event\s*\(\s*([^,]+)\s*,\s*"?([^")\n]+)')
    for match in evt_pattern.finditer(text):
        nid, content = match.groups()
        # Clean up potential trailing quote/paren
        content = content.split('"')[0].split(')')[0].strip()
        lines.append(f'({nid.strip()}, "{content}")')
        
    lines.append("</Node>")
    lines.append("<Edge>")
    
    # Parse relations: relation(src, rel, dst)
    # Regex: relation\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)\n]+)
    rel_pattern = re.compile(r'relation\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)\n]+)')
    for match in rel_pattern.finditer(text):
        src, rel, dst = match.groups()
        src = src.strip()
        rel = rel.strip()
        # Clean up potential trailing chars
        dst = dst.split(')')[0].strip()
        lines.append(f"({src}, {rel}, {dst})")
        
    lines.append("</Edge>")
    lines.append("</event_graph>")
    
    return "\n".join(lines)


def parse_bracket_prefix(prompt):
    """
    Parse prefixes like:
      [graph_search:shortest_path]
      [event_graph:maven_ere]
      [scene_graph:scene_graph_generation]
      [molecule:molecule_description]
    Returns (task_normalized, subtask_raw) or (None, None) if not present.

    NOTE: changed to search anywhere in the text (not only at the start) to handle
    prompts that contain headers like "system\n" or other prefixes before the tag.
    """
    if not isinstance(prompt, str):
        return None, None
    # Use search so tag can appear anywhere in the prompt (not just at start)
    m = re.search(r'\[([^\]:]+)(?::([^\]]+))?\]', prompt)
    if not m:
        return None, None
    task_token = m.group(1).strip().lower()
    subtask_token = m.group(2).strip() if m.group(2) else None

    # Map common aliases to internal category names
    alias_map = {
        "graph_search": "graph_search",
        "graph-search": "graph_search",
        "event_graph": "event_graph",
        "event-graph": "event_graph",
        "scene_graph": "ssg",
        "scene-graph": "ssg",
        "ssg": "ssg",
        "molecule": "mol",
        "mol": "mol",
        "molecular": "mol",
    }
    task_norm = alias_map.get(task_token, None)
    return task_norm, subtask_token


def extract_answer_content(text):
    """
    If the text contains '###', extract the part after it.
    Commonly used to separate reasoning from the final answer.
    """
    if isinstance(text, str) and "###" in text:
        return text.split("###", 1)[1].strip()
    return text


# =========================================================
# SGG: per-record processing
# =========================================================
def process_one_ssg_record(r):
    """
    单条 SGG record 的标准处理流程：
      1) 解析 / 填补 image_id
      2) attach_vg_gt_to_record: 补齐 VG150 GT + metadata
         - gt_boxes, gt_classes, gt_relations
         - ind_to_classes, ind_to_predicates
      3) 若 predict 是 [obj]/[rel] 文本，则：
         - process_text_prediction_item:
             * parse 文本 triplets
             * SentenceTransformer closed-set 映射
             * 映射到 GT index 空间，生成 pred_rel_triplets
      4) 返回增强后的 record，供批量评估使用
    """
    # ---- image_id ----
    # Try to extract image_id if not present, to help attach_vg_gt_to_record
    if "image_id" not in r:
        if "images" in r and isinstance(r["images"], list) and len(r["images"]) > 0:
            m = re.search(r"(\d+)\.[a-zA-Z]+$", r["images"][0])
            if m:
                try:
                    r["image_id"] = int(m.group(1))
                except ValueError:
                    pass

    # ---- GT / metadata ----
    # Mimic test_sgg_logic: Always try to attach GT to ensure we have boxes/classes/relations
    # This ensures gt_boxes are filled as requested.
    try:
        r = attach_vg_gt_to_record(r)
    except Exception as e:
        # [MODIFIED] Don't fail completely, just warn. 
        # This allows parsing predictions even if GT/Metadata is missing.
        # print(f"[WARN] attach_vg_gt_to_record failed: {e}", file=sys.__stdout__)
        pass

    # ---- VLM 文本输出 → pred_rel_triplets ----
    text_pred = r.get("predict", "")

    # [NEW] Extract content after ### if present
    text_pred = extract_answer_content(text_pred)
    r["predict"] = text_pred

    # [NEW] Handle Graph Facts format for SGG
    if is_graph_facts_format(text_pred):
        try:
            converted_text = parse_graph_facts_sgg(text_pred)
            r["predict"] = converted_text
            text_pred = converted_text
        except Exception as e:
            print(f"[WARN] Failed to convert Graph Facts SGG: {e}")

    # [NEW] Handle UGS format for SGG (Auto-detect)
    if is_ugs_format(text_pred):
        try:
            converted_text = convert_ugs_to_sgg_text(text_pred)
            r["predict"] = converted_text
            text_pred = converted_text
        except Exception as e:
            print(f"[WARN] Failed to convert UGS SGG: {e}")

    # ---- VLM 文本输出 → pred_rel_triplets ----
    # Mimic test_sgg_logic: Always try to process prediction
    try:
        r = process_text_prediction_item(r)
    except Exception as e:
        # print(f"[WARN] process_text_prediction_item failed: {e}")
        # Ensure defaults exist if failed
        r.setdefault("parsed_triplets", [])
        r.setdefault("pred_rel_triplets", [])

    return r


# =========================================================
# SGG: batch evaluation (PCIs + SGCls)
# =========================================================
def evaluate_ssg_batch(records):
    """
    对一组 SGG records 进行评估。
    返回 dict:
      {
        "pcis": { "R@50": ..., "R@100": ..., "mR@50": ..., "mR@100": ... },
        "sgcls": { 同上 },
        "parsed_triplets": { image_id_str: [ (s,p,o), ... ], ... }  // 便于 debug
      }
    """
    if not records:
        return {}

    print(f"[INFO] SSG evaluating {len(records)} samples")

    processed = []
    for r in tqdm(records, desc="SSG-Prep", ncols=80):
        item = process_one_ssg_record(r)
        if item is not None:
            processed.append(item)

    if not processed:
        print("[WARN] No valid SSG samples after processing.")
        return {}

    # ----- PCIs (Instance-Level Broadcasting) -----
    print("[INFO] Evaluating PCIs ...")
    pcis_metrics = eval_pcis(processed)

    # ----- SGCls (Strict Instance-Level 1-to-1) -----
    print("[INFO] Evaluating SGCls ...")
    sgcls_metrics = eval_sgcls(processed)

    # Extract primary metrics at default threshold (0.6)
    pcis_primary = pcis_metrics.get("0.6", {})
    sgcls_primary = sgcls_metrics.get("0.6", {})

    final = {
        "pcis": {
            "R@50": pcis_primary.get("R@50", 0.0),
            "R@100": pcis_primary.get("R@100", 0.0),
            "mR@50": pcis_primary.get("mR@50", 0.0),
            "mR@100": pcis_primary.get("mR@100", 0.0),
        },
        "sgcls": {
            "R@50": sgcls_primary.get("R@50", 0.0),
            "R@100": sgcls_primary.get("R@100", 0.0),
            "mR@50": sgcls_primary.get("mR@50", 0.0),
            "mR@100": sgcls_primary.get("mR@100", 0.0),
        },
    }
    

    # 收集 parsed_triplets，方便人工检查映射效果
    # triplet_map = {}
    # for r in processed:
    #     imgid = r.get("image_id")
    #     if imgid is not None and r.get("parsed_triplets"):
    #         triplet_map[str(imgid)] = r["parsed_triplets"]

    # if triplet_map:
    #     final["parsed_triplets"] = triplet_map

    return final


# =========================================================
# Top-level router: all tasks
# =========================================================
def classify_record_category(record):
    prompt = record.get("prompt", "")
    # 1) Try bracket prefix method first (now searches anywhere)
    task_from_prefix, subtask_from_prefix = parse_bracket_prefix(prompt if isinstance(prompt, str) else "")
    if task_from_prefix:
        # attach normalized task/subtask back to record for downstream use
        record["task"] = task_from_prefix
        if subtask_from_prefix:
            # normalize common event subtask tokens when task is event_graph
            if task_from_prefix == "event_graph":
                # use detect_event_subtask to normalize casing/labels
                record["subtask"] = detect_event_subtask({"subtask": subtask_from_prefix})
            else:
                record["subtask"] = subtask_from_prefix.strip()
        # Map to the same category names used elsewhere
        if task_from_prefix == "graph_search":
            return "graph_search"
        if task_from_prefix == "event_graph":
            return "event_graph"
        if task_from_prefix == "ssg":
            return "ssg"
        if task_from_prefix == "mol":
            return "mol"

    # If not found at top-level prompt, try other fields (predict/label) as fallback
    for field in ("predict", "label"):
        if record.get(field):
            t, s = parse_bracket_prefix(record.get(field))
            if t:
                record["task"] = t
                if s:
                    if t == "event_graph":
                        record["subtask"] = detect_event_subtask({"subtask": s})
                    else:
                        record["subtask"] = s.strip()
                if t == "graph_search":
                    return "graph_search"
                if t == "event_graph":
                    return "event_graph"
                if t == "ssg":
                    return "ssg"
                if t == "mol":
                    return "mol"

    # 2) Fallback to existing heuristics
    if record.get("task") == "graph_search":
        return "graph_search"
    if isinstance(prompt, str):
        if "visual relationships" in prompt:
            return "ssg"
        if "SMILES:" in prompt or "smiles:" in prompt:
            return "mol"
        if "event graph" in prompt:
            # If mentions event graph, default to event_graph
            # and try to normalize subtask from any surrounding text
            pf = " ".join([str(record.get("prompt", "")), str(record.get("predict", "")), str(record.get("label", ""))])
            record["subtask"] = detect_event_subtask(pf)
            return "event_graph"

        # Detect explicit Graph Facts / molecule-like content anywhere in prompt/predict/label
        pf = " ".join([str(record.get("prompt", "")), str(record.get("predict", "")), str(record.get("label", ""))]).lower()
        if "atom(" in pf or "bond(" in pf or "graph facts" in pf or is_graph_facts_format(pf):
            return "mol"

        # Detect event-graph indicators (events/relations/Matres/Ere tokens) anywhere
        if any(tok in pf for tok in ("event(", "relation(", "<event_graph>", "matres", "maven", "ere")):
            # normalize subtask using the joined fields
            record["subtask"] = detect_event_subtask(pf)
            return "event_graph"

        # [Added] Auto-detect graph search based on keywords or content
        if detect_graph_subtask(record) != "unknown":
            return "graph_search"
        if "graph" in prompt.lower() and ("node" in prompt.lower() or "edge" in prompt.lower()):
            return "graph_search"

    return None


def detect_graph_subtask(record):
    # 优先使用由 prefix 填入的 subtask（如果存在且非 unknown）
    if record is not None:
        pref = record.get("subtask")
        if isinstance(pref, str) and pref and pref.lower() != "unknown":
            return pref
    prompt = record.get("prompt", "")
    if not isinstance(prompt, str):
        return "unknown"
    text = prompt.lower()
    for subtask, keywords in SUBTASK_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return subtask
    return "unknown"


def detect_event_subtask(record):
    # Accept either a record dict or a plain string for convenience
    if isinstance(record, str):
        prompt = record
        pref = None
    else:
        pref = record.get("subtask")
        prompt = record.get("prompt", "")

    # If explicit pref is provided, normalize it
    if isinstance(pref, str) and pref:
        s = pref.strip().lower()
        if s in ("ere", "maven_ere", "maven-ere", "mavenere"):
            return "ERE"
        if s in ("matres", "maters", "matres?"):
            return "MATRES"
        if s in ("hievent", "hievents", "hie", "hie-event"):
            return "Hievent"
        # return pref normalized (preserve reasonable casing if unknown)
        return pref

    # Fallback: inspect prompt/predict/label text
    if not isinstance(prompt, str):
        return "Hievent"
    text = prompt.lower()
    if "ere" in text or "maven" in text:
        return "ERE"
    if "matres" in text or "maters" in text:
        return "MATRES"
    if "hievent" in text or "hie" in text:
        return "Hievent"
    return "Hievent"


def evaluate_all(jsonl_file):
    """
    统一入口：读取 JSONL，每条记录包含 `task` 字段：
      - "graph_search"
      - "mol"
      - "ssg" or "scene_graph"
      - "event_graph"
    返回一个 results dict，可直接 dump 成 json。
    """
    with open(jsonl_file, "r") as f:
        records = [json.loads(line) for line in f]
    print(f"[INFO] Loaded {len(records)} records")

    graph, mol, ssg, event = [], [], [], []
    for r in records:
        category = classify_record_category(r)
        if category == "graph_search":
            graph.append(r)
        elif category == "mol":
            mol.append(r)
        elif category == "ssg":
            ssg.append(r)
        elif category == "event_graph":
            event.append(r)

    result = {}
    print(f"[INFO] Records by category: graph_search={len(graph)}, mol={len(mol)}, ssg={len(ssg)}, event_graph={len(event)}")
    # ---------- Graph Search ----------
    if graph:
        print(f"[INFO] Evaluating Graph Search on {len(graph)} samples")
        accs = []
        subtask_scores = defaultdict(list)
        for r in graph:
            r["predict"] = extract_answer_content(r.get("predict", ""))
            acc = eval_graph_search(r)
            accs.append(acc)
            subtask = detect_graph_subtask(r)
            subtask_scores[subtask].append(acc)
        overall = sum(accs) / len(accs)
        by_subtask = {
            subtask: sum(scores) / len(scores)
            for subtask, scores in subtask_scores.items()
        }
        result["graph_search"] = {
            "accuracy": overall,
            "by_subtask": by_subtask,
        }
    print(f"[INFO] Graph Search accuracy: {result.get('graph_search', {})}")
    # ---------- Molecular ----------
    if mol:
        print(f"[INFO] Evaluating Molecular on {len(mol)} samples")
        bleu_sum = 0.0
        rouge_sum = 0.0
        for r in mol:
            r["predict"] = extract_answer_content(r.get("predict", ""))
            m = eval_molecular(r)
            bleu_sum += m.get("bleu", 0.0)
            rouge_sum += m.get("rouge_l", 0.0)
        n = len(mol)
        result["molecular"] = {
            "bleu": bleu_sum / n,
            "rouge_l": rouge_sum / n,
        }
    print(f"[INFO] Molecular metrics: {result.get('molecular', 'N/A')}")
    
    # ---------- SSG (PCIs + SGCls) ----------
    if ssg:
        result["ssg"] = evaluate_ssg_batch(ssg)
    print(f"[INFO] SSG metrics: {result.get('ssg', 'N/A')}")
    
    # ---------- Event Graph ----------
    if event:
        print(f"[INFO] Evaluating Event Graph on {len(event)} samples")
        overall_metrics = []
        subtask_metrics = defaultdict(list)
        for r in event:
            subtask = detect_event_subtask(r)
            r["subtask"] = subtask

            r["predict"] = extract_answer_content(r.get("predict", ""))

            # [NEW] Handle Graph Facts / UGS format for Event Graph (Auto-detect)
            # 1. Convert Prediction
            pred_text = r.get("predict", "")
            if is_graph_facts_format(pred_text):
                try:
                    r["predict"] = parse_graph_facts_event(pred_text)
                except Exception as e:
                    print(f"[WARN] Failed to convert Graph Facts Event Graph Predict: {e}")
            elif is_ugs_format(pred_text):
                try:
                    r["predict"] = convert_ugs_to_event_graph_text(pred_text)
                except Exception as e:
                    print(f"[WARN] Failed to convert UGS Event Graph Predict: {e}")

            # 2. Convert Label (Critical if GT is also in UGS/Graph Facts format)
            label_text = r.get("label", "")
            if is_graph_facts_format(label_text):
                try:
                    r["label"] = parse_graph_facts_event(label_text)
                except Exception as e:
                    print(f"[WARN] Failed to convert Graph Facts Event Graph Label: {e}")
            elif is_ugs_format(label_text):
                try:
                    r["label"] = convert_ugs_to_event_graph_text(label_text)
                except Exception as e:
                    print(f"[WARN] Failed to convert UGS Event Graph Label: {e}")

            m = evaluate_event_graph_original_metrics(r)
            overall_metrics.append(m)
            subtask_metrics[subtask].append(m)
        result["event_graph"] = {
            "overall": {
                key: sum(m.get(key, 0.0) for m in overall_metrics) / len(overall_metrics)
                for key in EVENT_METRIC_KEYS
            }
            if overall_metrics else {key: 0.0 for key in EVENT_METRIC_KEYS},
            "by_subtask": {
                subtask: {
                    key: sum(m.get(key, 0.0) for m in metrics) / len(metrics)
                    for key in EVENT_METRIC_KEYS
                }
                for subtask, metrics in subtask_metrics.items()
            },
        }

    return result


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Where to save results JSON")
    args = parser.parse_args()

    print(f"[INFO] Evaluating file: {args.input}")
    res = evaluate_all(args.input)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print("[DONE] Saved results to", args.output)


if __name__ == "__main__":
    main()
