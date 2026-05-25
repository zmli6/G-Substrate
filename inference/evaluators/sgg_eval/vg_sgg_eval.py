#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGG Utilities (PGSG-aligned)
============================

Implements:
  - Loading VG150 GT & metadata
  - Parsing VLM "[obj]/[rel]" textual triplets
  - SentenceTransformer closed-set mapping (entity/predicate)
  - Mapping mapped triplets → GT index space (sub_idx, obj_idx, pred_id)
  - PCIs / PCls evaluation (R@K / mR@K)   [class-level]
  - SGCls evaluation (R@K / mR@K)         [instance-level]

No Detectron. No BoxList. Pure Python PGSG-style evaluation.
"""

import json
import os
import re
import numpy as np
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer, util
import nltk

# ============================================================
# NLTK / WordNet Setup
# ============================================================
try:
    from nltk.corpus import wordnet as wn
except ImportError:
    wn = None

def check_nltk_resources():
    if wn is None:
        return
    try:
        # Try to access a common synset to check if loaded
        wn.synsets('entity')
    except LookupError:
        print("[INFO] Downloading WordNet resources...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')

check_nltk_resources()


# ============================================================
# Paths (relative to this sgg_utils.py file)
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VG_META_PATH = os.path.join(THIS_DIR, "vg_metadata.json")
VG_GT_PATH = os.path.join(THIS_DIR, "vg150_gt.pkl")

VG_META_CACHE = None
VG_GT_CACHE = None


# ============================================================
# Load VG metadata & GT
# ============================================================
def load_vg_metadata():
    global VG_META_CACHE
    if VG_META_CACHE is not None:
        return VG_META_CACHE
    with open(VG_META_PATH, "r") as f:
        VG_META_CACHE = json.load(f)
    return VG_META_CACHE


def load_vg_gt():
    global VG_GT_CACHE
    if VG_GT_CACHE is not None:
        return VG_GT_CACHE
    import pickle
    with open(VG_GT_PATH, "rb") as f:
        VG_GT_CACHE = pickle.load(f)
    return VG_GT_CACHE


def attach_vg_gt_to_record(r):
    """
    Ensure record r contains:
      - image_id
      - gt_boxes (xyxy)
      - gt_classes
      - gt_relations (sub_idx, obj_idx, pred_id)
      - ind_to_classes
      - ind_to_predicates
    """
    meta = load_vg_metadata()
    gt_all = load_vg_gt()

    # image_id
    if "image_id" in r:
        image_id = int(r["image_id"])
    else:
        img_path = r["images"][0]
        basename = os.path.basename(img_path)
        image_id = int(os.path.splitext(basename)[0])
        r["image_id"] = image_id

    if image_id not in gt_all:
        raise ValueError(f"No GT for image {image_id}")

    g = gt_all[image_id]

    # xywh -> xyxy
    raw_boxes = np.array(g["boxes"], dtype=np.float32)
    xyxy = []
    for x, y, w, h in raw_boxes:
        xyxy.append([float(x), float(y), float(x + w), float(y + h)])

    r["gt_boxes"] = xyxy
    r["gt_classes"] = list(g["labels"])
    r["gt_relations"] = [list(t) for t in g["relation_tuple"]]

    r["ind_to_classes"] = meta["ind_to_classes"]
    r["ind_to_predicates"] = meta["ind_to_predicates"]
    return r


# ============================================================
# Triplet parse: "man [obj] on [rel] ski [obj]"
# ============================================================
def parse_generated_triplets(text):
    """
    Parse VLM output:
      1. "a [obj] on [rel] b [obj], c [obj] in [rel] d [obj] [SEP]"
      2. "(a, on, b), (c, in, d)"
    → [(a, on, b), (c, in, d)]
    """
    triplets = []
    if not isinstance(text, str):
        return triplets

    # 1. Try tuple format: (s, p, o)
    # Regex looks for content inside parens, separated by commas.
    tuple_pattern = r'\(\s*([^,()]+)\s*,\s*([^,()]+)\s*,\s*([^,()]+)\s*\)'
    tuple_matches = re.findall(tuple_pattern, text)
    
    if tuple_matches:
        for s, p, o in tuple_matches:
            # Clean up whitespace and quotes
            s_clean = s.strip().strip('"\'')
            p_clean = p.strip().strip('"\'')
            o_clean = o.strip().strip('"\'')
            triplets.append((s_clean, p_clean, o_clean))
        
        # If we found tuples, return immediately to avoid mixed parsing
        if triplets:
            return triplets

    segments = re.split(r',[ ]*|\[SEP\]', text)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # greedy form: (.*?) [obj] (.*?) [rel] (.*?) [obj]
        m = re.search(
            r'^(.*?)\s*\[obj\]\s*(.*?)\s*\[rel\]\s*(.*?)\s*\[obj\]',
            seg,
            flags=re.IGNORECASE
        )
        if m:
            s, p, o = m.groups()
            triplets.append((s.strip(), p.strip(), o.strip()))
    return triplets


# ============================================================
# Closed-set mapping using WordNet + SentenceTransformer
# ============================================================
EMB_MODEL = SentenceTransformer("all-mpnet-base-v2")
ENTITY_EMB_CACHE = None
PRED_EMB_CACHE = None
ENTITY_SYNSETS_CACHE = None
PRED_SYNSETS_CACHE = None
ENTITY_MAP_CACHE = {}
PRED_MAP_CACHE = {}


def get_synsets(text):
    """Helper to get synsets for a word, handling spaces/underscores."""
    if wn is None:
        return set()
    # Try original
    ss = set(wn.synsets(text))
    # Try underscore replacement for multi-word terms
    if not ss and " " in text:
        ss = set(wn.synsets(text.replace(" ", "_")))
    return ss


def build_entity_emb(ind_to_classes):
    global ENTITY_EMB_CACHE
    if ENTITY_EMB_CACHE is not None:
        return ENTITY_EMB_CACHE
    names = [
        ind_to_classes[str(i)].lower()
        for i in range(1, len(ind_to_classes) + 1)
    ]
    emb = EMB_MODEL.encode(names, convert_to_tensor=True)
    ENTITY_EMB_CACHE = (names, emb)
    return ENTITY_EMB_CACHE


def build_entity_synsets(ind_to_classes):
    global ENTITY_SYNSETS_CACHE
    if ENTITY_SYNSETS_CACHE is not None:
        return ENTITY_SYNSETS_CACHE
    cache = {}
    for i in range(1, len(ind_to_classes) + 1):
        name = ind_to_classes[str(i)].lower()
        cache[name] = get_synsets(name)
    ENTITY_SYNSETS_CACHE = cache
    return ENTITY_SYNSETS_CACHE


def build_pred_emb(ind_to_predicates):
    global PRED_EMB_CACHE
    if PRED_EMB_CACHE is not None:
        return PRED_EMB_CACHE
    names = [
        ind_to_predicates[str(i)].lower()
        for i in range(1, len(ind_to_predicates) + 1)
    ]
    emb = EMB_MODEL.encode(names, convert_to_tensor=True)
    PRED_EMB_CACHE = (names, emb)
    return PRED_EMB_CACHE


def build_pred_synsets(ind_to_predicates):
    global PRED_SYNSETS_CACHE
    if PRED_SYNSETS_CACHE is not None:
        return PRED_SYNSETS_CACHE
    cache = {}
    for i in range(1, len(ind_to_predicates) + 1):
        name = ind_to_predicates[str(i)].lower()
        cache[name] = get_synsets(name)
    PRED_SYNSETS_CACHE = cache
    return PRED_SYNSETS_CACHE


def map_to_closed(raw, closed_names, closed_emb, cache, closed_synsets_map=None):
    """
    Return the closest closed-set class name.
    Strategy:
      0. Exact string match (in closed_names).
      1. Exact string match (via cache check).
      2. WordNet Synset overlap (Logical Alignment).
      3. Sentence-BERT Cosine Similarity > 0.8 (Semantic Patch).
    """
    key = raw.lower().strip()
    if key in cache:
        return cache[key]

    # --- Stage 0: Exact Match in List ---
    # Prioritize exact match before Synset alignment to avoid polysemy issues
    # (e.g. 'sign' matching 'house' via zodiac sense)
    if key in closed_names:
        cache[key] = key
        return key

    # --- Stage 1: WordNet Synset Alignment ---
    if closed_synsets_map:
        raw_ss = get_synsets(key)
        if raw_ss:
            # Check if raw word shares any synset with any target class
            # We iterate closed_names to prioritize order if needed, or just to find *a* match
            for cname in closed_names:
                tgt_ss = closed_synsets_map.get(cname)
                if tgt_ss and not raw_ss.isdisjoint(tgt_ss):
                    cache[key] = cname
                    return cname

    # --- Stage 2: Sentence-BERT Semantic Patch ---
    raw_emb = EMB_MODEL.encode(raw, convert_to_tensor=True)
    scores = util.cos_sim(raw_emb, closed_emb)[0]
    idx = int(torch.argmax(scores))
    score = float(scores[idx])

    # Threshold check (0.8)
    if score >= 0.8:
        cache[key] = closed_names[idx]
        return cache[key]

    # No valid mapping found
    cache[key] = None
    return None


def map_entity(ent, ind_to_classes):
    names, emb = build_entity_emb(ind_to_classes)
    synsets = build_entity_synsets(ind_to_classes)
    return map_to_closed(ent, names, emb, ENTITY_MAP_CACHE, synsets)


def map_predicate(pred, ind_to_predicates):
    names, emb = build_pred_emb(ind_to_predicates)
    synsets = build_pred_synsets(ind_to_predicates)
    return map_to_closed(pred, names, emb, PRED_MAP_CACHE, synsets)


# ============================================================
# Map triplets → GT index space (sub_idx, obj_idx, pred_id)
# ============================================================

# [SOFT EVALUATION]
SOFT_MATCH_THRESHOLDS = [0.2, 0.4, 0.6, 0.8, 1.0]

def get_class_similarity(cls_id1, cls_id2, ind_to_classes):
    """
    Compute SBERT similarity between two class IDs.
    """
    if cls_id1 == cls_id2:
        return 1.0
    
    # Use cached embeddings if possible
    names, emb = build_entity_emb(ind_to_classes)
    
    # ind_to_classes keys are strings "1", "2"...
    # names list is 0-indexed, so class "1" is at index 0.
    idx1 = int(cls_id1) - 1
    idx2 = int(cls_id2) - 1
    
    if idx1 < 0 or idx1 >= len(emb) or idx2 < 0 or idx2 >= len(emb):
        return 0.0
        
    sim = util.cos_sim(emb[idx1], emb[idx2]).item()
    return sim

def find_soft_matching_instances(target_cls_id, gt_classes, ind_to_classes, threshold):
    matches = []
    for i, c in enumerate(gt_classes):
        if get_class_similarity(str(target_cls_id), str(c), ind_to_classes) >= threshold:
            matches.append(i)
    return matches

def is_soft_match(gt_trip, pred_trip, ind_to_classes, threshold):
    """
    Check if pred_trip is a soft match for gt_trip.
    Triplets are (sub_cls, pred_id, obj_cls).
    Predicate must match exactly.
    Subject and Object can soft match.
    """
    s_gt, p_gt, o_gt = gt_trip
    s_pred, p_pred, o_pred = pred_trip
    
    if p_gt != p_pred:
        return False
        
    if s_gt == s_pred and o_gt == o_pred:
        return True
        
    # Soft match check
    sim_s = get_class_similarity(str(s_gt), str(s_pred), ind_to_classes)
    sim_o = get_class_similarity(str(o_gt), str(o_pred), ind_to_classes)
    
    return sim_s >= threshold and sim_o >= threshold

def map_triplets_to_gt_indices(r, threshold):
    """
    Helper to map parsed triplets to GT indices using a specific threshold.
    Returns list of [sub_idx, obj_idx, pred_id]
    """
    mapped = r.get("parsed_triplets", [])
    ind_to_classes = r["ind_to_classes"]
    ind_to_predicates = r["ind_to_predicates"]
    gt_classes = r["gt_classes"]

    name_to_cls = {v.lower(): int(k) for k, v in ind_to_classes.items()}
    name_to_pred = {v.lower(): int(k) for k, v in ind_to_predicates.items()}

    pred_rel_triplets = []
    for s, p, o in mapped:
        s_id = name_to_cls.get(s.lower())
        p_id = name_to_pred.get(p.lower())
        o_id = name_to_cls.get(o.lower())
        if s_id is None or p_id is None or o_id is None:
            continue

        # match GT instance indices that have the same class
        sub_idxs = [i for i, c in enumerate(gt_classes) if c == s_id]
        obj_idxs = [i for i, c in enumerate(gt_classes) if c == o_id]

        # [SOFT MATCH ADDITION]
        # If no exact match found, try soft match
        if not sub_idxs:
            sub_idxs = find_soft_matching_instances(s_id, gt_classes, ind_to_classes, threshold)
        if not obj_idxs:
            obj_idxs = find_soft_matching_instances(o_id, gt_classes, ind_to_classes, threshold)

        for si in sub_idxs:
            for oi in obj_idxs:
                if si != oi:
                    pred_rel_triplets.append([si, oi, p_id])
    return pred_rel_triplets

def process_text_prediction_item(r):
    """
    Produces:
      r["parsed_triplets"]   = list of (s,p,o) after closed-set mapping
      r["pred_rel_triplets"] = list of [sub_idx, obj_idx, pred_id] (using default threshold 0.2)
    """
    text = r.get("predict", "")
    raw_triplets = parse_generated_triplets(text)

    ind_to_classes = r["ind_to_classes"]
    ind_to_predicates = r["ind_to_predicates"]
    
    # closed-set mapping
    mapped = []
    for s, p, o in raw_triplets:
        s2 = map_entity(s, ind_to_classes)
        p2 = map_predicate(p, ind_to_predicates)
        o2 = map_entity(o, ind_to_classes)
        
        # Only keep if all parts mapped successfully (not None)
        if s2 is not None and p2 is not None and o2 is not None:
            mapped.append((s2, p2, o2))

    r["parsed_triplets"] = mapped
    
    # Default threshold 0.2 for backward compatibility in single-record usage
    r["pred_rel_triplets"] = map_triplets_to_gt_indices(r, threshold=0.2)
    return r


# ============================================================
# Recall metrics: instance-level (SGCls) & class-level (PCIS)
# ============================================================



def compute_recall_instance_level(records, ks=(50, 100), threshold=0.2):
    """
    Instance-level triplet recall (SGCls-style) with 1-to-1 Matching.

    Previous implementation broadcasted one prediction to ALL matching GT instances.
    This version enforces that one prediction can only match ONE GT instance.
    This aligns better with standard SGG where missing instances penalizes recall.
    """
    if not records:
        return {}

    # find predicate count
    first = next(r for r in records if "ind_to_predicates" in r)
    num_pred = len(first["ind_to_predicates"])

    results = {}

    for K in ks:
        total_hit = 0
        total_gt = 0

        class_hit = defaultdict(int)
        class_total = defaultdict(int)

        for r in records:
            gt_rels = r.get("gt_relations", [])     # [(si, oi, pid), ...] indices
            gt_classes = r.get("gt_classes", [])    # [cls_id, ...]
            ind_to_classes = r.get("ind_to_classes", {})
            ind_to_predicates = r.get("ind_to_predicates", {})

            # 1. Build GT Pool: List of (s_cls, pid, o_cls)
            # Use a list to keep track of instances (duplicates allowed/expected)
            gt_pool = []
            for si, oi, pid in gt_rels:
                s_cls = int(gt_classes[int(si)])
                o_cls = int(gt_classes[int(oi)])
                pid = int(pid)
                gt_pool.append({
                    "triple": (s_cls, pid, o_cls),
                    "matched": False
                })
                total_gt += 1
                class_total[pid] += 1

            # 2. Build Pred List: List of (s_cls, pid, o_cls) from Parsed Text
            # Map names to IDs on the fly
            name_to_cls = {v.lower(): int(k) for k, v in ind_to_classes.items()}
            name_to_pred = {v.lower(): int(k) for k, v in ind_to_predicates.items()}

            pred_parsed = r.get("parsed_triplets", [])
            # Take Top K from the parsed text list
            # Note: parsed_triplets are usually ordered by generation order
            
            curr_pred_triples = []
            for s, p, o in pred_parsed:
                if len(curr_pred_triples) >= K:
                    break
                
                sid = name_to_cls.get(s.lower())
                oid = name_to_cls.get(o.lower())
                pid_val = name_to_pred.get(p.lower())
                
                if sid is not None and oid is not None and pid_val is not None:
                     curr_pred_triples.append((sid, pid_val, oid))
            
            # 3. Perform 1-to-1 Matching (Greedy)
            # For each prediction, try to consume one GT instance.
            for (ps, pp, po) in curr_pred_triples:
                # Find first unmatched GT that fits
                match_idx = -1
                for idx, item in enumerate(gt_pool):
                    if item["matched"]:
                        continue
                        
                    gs, gp, go = item["triple"]
                    
                    if gp != pp:
                        continue
                    
                    # Entity Check (incorporating Soft Match logic)
                    is_match = False
                    if gs == ps and go == po:
                        is_match = True
                    else:
                        # Use Soft Match helper
                        # is_soft_match takes (gt_sub_cls, gt_pred, gt_obj_cls) vs (pred...)
                        if is_soft_match((gs, gp, go), (ps, pp, po), ind_to_classes, threshold):
                            is_match = True
                    
                    if is_match:
                        match_idx = idx
                        break
                
                if match_idx != -1:
                    gt_pool[match_idx]["matched"] = True
                    total_hit += 1
                    class_hit[pp] += 1

        # overall R@K
        Rk = total_hit / total_gt if total_gt > 0 else 0.0

        # mean recall mR@K
        per_class_recalls = [
            class_hit[p] / class_total[p]
            for p in range(num_pred)
            if class_total[p] > 0
        ]
        mRk = (
            sum(per_class_recalls) / len(per_class_recalls)
            if per_class_recalls else 0.0
        )

        results[f"R@{K}"] = Rk
        results[f"mR@{K}"] = mRk

    return results


def compute_recall_instance_level_broadcasting(records, ks=(50, 100), threshold=0.2):
    """
    Instance-level Broadcasting Recall (Approximation for PredCls without Boxes).
    
    Definition:
    - Denominator: ALL GT instances (Instance-Level).
    - Numerator: If model predicts concept (s, p, o), it hits ALL GT instances of (s, p, o).
    
    Why this makes sense for PredCls in Text-to-Graph:
    - It simulates the "Upper Bound" where we assume the model correctly grounds
      all instances of a class, and we only judge if it retrieved the correct relation type.
    - This ensures PCIs > SGCIs while keeping the denominator Instance-Level.
    """
    if not records:
        return {}

    # find predicate count
    first = next(r for r in records if "ind_to_predicates" in r)
    num_pred = len(first["ind_to_predicates"])

    results = {}

    for K in ks:
        total_hit = 0
        total_gt = 0

        class_hit = defaultdict(int)
        class_total = defaultdict(int)

        for r in records:
            gt_rels = r.get("gt_relations", [])     # [(si, oi, pid), ...]
            gt_classes = r.get("gt_classes", [])
            ind_to_classes = r.get("ind_to_classes", {})
            ind_to_predicates = r.get("ind_to_predicates", {})

            # 1. Build GT List (Instance Level)
            # stored as (s_cls, pid, o_cls) tuples
            gt_instances = []
            for si, oi, pid in gt_rels:
                s_cls = int(gt_classes[int(si)])
                o_cls = int(gt_classes[int(oi)])
                pid = int(pid)
                gt_trip = (s_cls, pid, o_cls)
                
                gt_instances.append(gt_trip)
                total_gt += 1
                class_total[pid] += 1

            # 2. Build Pred Set (Unique Concepts)
            # Map names to IDs on the fly
            name_to_cls = {v.lower(): int(k) for k, v in ind_to_classes.items()}
            name_to_pred = {v.lower(): int(k) for k, v in ind_to_predicates.items()}

            pred_parsed = r.get("parsed_triplets", [])
            
            pred_concepts = set()
            for idx, (s, p, o) in enumerate(pred_parsed):
                if idx >= K: break
                
                sid = name_to_cls.get(s.lower())
                oid = name_to_cls.get(o.lower())
                pid_val = name_to_pred.get(p.lower())
                
                if sid is not None and oid is not None and pid_val is not None:
                     pred_concepts.add((sid, pid_val, oid))

            # 3. Broadcasting Match
            # For every GT instance, check if its TYPE is in pred_concepts
            for gt_trip in gt_instances:
                pid = gt_trip[1]
                
                # Check Exact Match
                if gt_trip in pred_concepts:
                    total_hit += 1
                    class_hit[pid] += 1
                else:
                    # Check Soft Match against all pred concepts
                    # (This is slower but necessary if soft match is enabled)
                    matched = False
                    for pred_trip in pred_concepts:
                         if is_soft_match(gt_trip, pred_trip, ind_to_classes, threshold):
                             matched = True
                             break
                    
                    if matched:
                        total_hit += 1
                        class_hit[pid] += 1

        # overall R@K
        Rk = total_hit / total_gt if total_gt > 0 else 0.0

        # mean recall mR@K
        per_class_recalls = [
            class_hit[p] / class_total[p]
            for p in range(num_pred)
            if class_total[p] > 0
        ]
        mRk = (
            sum(per_class_recalls) / len(per_class_recalls)
            if per_class_recalls else 0.0
        )

        results[f"R@{K}"] = Rk
        results[f"mR@{K}"] = mRk

    return results



# ============================================================
# Legacy Function for Class-Level Recall (Original PCIs)
# ============================================================
def compute_recall_class_level(records, ks=(50, 100), threshold=0.2):
    """
    Class-level triplet recall (Old PCIs / PCls-style):
    - Denom: Unique GT class triplets.
    - Num: Set intersection of Pred class triplets and GT class triplets.
    """
    if not records:
        return {}

    # find predicate count
    first = next(r for r in records if "ind_to_predicates" in r)
    num_pred = len(first["ind_to_predicates"])
    ind_to_classes = first["ind_to_classes"]
    ind_to_predicates = first["ind_to_predicates"]

    results = {}

    for K in ks:
        total_hit = 0
        total_gt = 0

        class_hit = defaultdict(int)
        class_total = defaultdict(int)

        for r in records:
            gt_rel = r.get("gt_relations", [])
            gt_classes = r.get("gt_classes", [])

            # ---- 1) GT: index-triplet -> class-triplet ----
            gt_class_triples = set()
            for si, oi, pid in gt_rel:
                si = int(si)
                oi = int(oi)
                pid = int(pid)
                if (
                    si < 0 or oi < 0 or
                    si >= len(gt_classes) or oi >= len(gt_classes)
                ):
                    continue
                s_cls = int(gt_classes[si])
                o_cls = int(gt_classes[oi])
                gt_class_triples.add((s_cls, pid, o_cls))

            # ---- 2) Pred: index-triplet -> class-triplet ----
            # We use parsed_triplets directly to avoid dependency on map_triplets_to_gt_indices
            name_to_cls = {v.lower(): int(k) for k, v in ind_to_classes.items()}
            name_to_pred = {v.lower(): int(k) for k, v in ind_to_predicates.items()}
            if "ind_to_predicates" in r:
                 name_to_pred = {v.lower(): int(k) for k, v in r["ind_to_predicates"].items()} # Safe lookup

            pred_parsed = r.get("parsed_triplets", [])
            pred_class_triples = set()
            for idx, (s, p, o) in enumerate(pred_parsed):
                if idx >= K: break
                sid = name_to_cls.get(s.lower())
                oid = name_to_cls.get(o.lower())
                pid_val = name_to_pred.get(p.lower())
                if sid is not None and oid is not None and pid_val is not None:
                     pred_class_triples.add((sid, pid_val, oid))


            # ---- 3) Class-Space Matching (Soft Match) ----
            for gt_trip in gt_class_triples:
                s_cls, pid, o_cls = gt_trip
                total_gt += 1
                class_total[pid] += 1

                # Check if ANY pred triplet matches this GT triplet (softly)
                match_found = False
                for pred_trip in pred_class_triples:
                    # is_soft_match helper
                    if is_soft_match(gt_trip, pred_trip, ind_to_classes, threshold):
                        match_found = True
                        break
                
                if match_found:
                    total_hit += 1
                    class_hit[pid] += 1

        # overall R@K
        Rk = total_hit / total_gt if total_gt > 0 else 0.0

        # mean recall mR@K
        per_class_recalls = [
            class_hit[p] / class_total[p]
            for p in range(num_pred)
            if class_total[p] > 0
        ]
        mRk = (
            sum(per_class_recalls) / len(per_class_recalls)
            if per_class_recalls else 0.0
        )

        results[f"R@{K}"] = Rk
        results[f"mR@{K}"] = mRk

    return results

# ============================================================
# EXPORTED EVALUATORS
# ============================================================

def eval_pcis(records):
    """
    [NEW STANDARD] PCIS / PCls-style evaluation:
      - Instance-level Denominator
      - Broadcasting Numerator (Simulates PredCls upper bound)
    """
    final_results = {}
    for th in SOFT_MATCH_THRESHOLDS:
        final_results[str(th)] = compute_recall_instance_level_broadcasting(records, ks=(50, 100), threshold=th)
    return final_results

def eval_sgcls(records):
    """
    [NEW STANDARD] SGCls-style evaluation:
      - Instance-level Denominator
      - 1-to-1 Matching (Simulates SGCls strict)
    """
    final_results = {}
    for th in SOFT_MATCH_THRESHOLDS:
        final_results[str(th)] = compute_recall_instance_level(records, ks=(50, 100), threshold=th)
    return final_results

def eval_pcis_original_class_level(records):
    """
    [OLD/LEGACY] Original PCIs (Class-Level Denom).
    """
    final_results = {}
    for th in SOFT_MATCH_THRESHOLDS:
        final_results[str(th)] = compute_recall_class_level(records, ks=(50, 100), threshold=th)
    return final_results



# ============================================================
# PCIS / PCls evaluation (class-level)
# ============================================================
def eval_pcis(records):
    """
    [NEW STANDARD] PCIS / PCls-style evaluation (Instance-Level Denom, Broadcasting).
    """
    final_results = {}
    for th in SOFT_MATCH_THRESHOLDS:
        final_results[str(th)] = compute_recall_instance_level_broadcasting(records, ks=(50, 100), threshold=th)
    return final_results



# ============================================================
# SGCls evaluation (instance-level)
# ============================================================
def eval_sgcls(records):
    """
    SGCls-style evaluation:
      - 在 (sub_idx, obj_idx, pred_id) 空间度量
      - 要求具体 subject / object 实例 index 匹配
      - [Modified] Uses 1-to-1 matching to prevent broadcasting/inflation.
    """
    final_results = {}
    for th in SOFT_MATCH_THRESHOLDS:
        # Note: We now perform matching inside compute_recall_instance_level
        # so we don't need to explode pred_rel_triplets here, but we keep it for debug.
        for r in records:
             r["pred_rel_triplets"] = map_triplets_to_gt_indices(r, th)
            
        final_results[str(th)] = compute_recall_instance_level(records, ks=(50, 100), threshold=th)
    return final_results

# ============================================================
# PredCls evaluation (Standard Instance-level)
# ============================================================
def eval_predcls_instance(records):
    """
    Standard Instance-level PredCls evaluation.
    In the context of Text-to-Graph (without provided GT boxes), this is mathematically
    equivalent to Instance-level SGCls (1-to-1 matching).
    
    Returns: Same as eval_sgcls
    """
    return eval_sgcls(records)
