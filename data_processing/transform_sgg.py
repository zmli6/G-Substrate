import json
import os
import argparse
import concurrent.futures
import re
from typing import List

############################################
# 2. Extract Scene Graph Triplets
############################################

def extract_scene_graph_text(assistant_content: str) -> str:
    """
    Extract text after ':' if present.
    """
    if ":" in assistant_content:
        return assistant_content.split(":", 1)[1].strip()
    return assistant_content.strip()

############################################
# 3. Rule-based Generation (Replaces vLLM)
############################################

def generate_graph_facts_rule_based(text: str) -> str:
    """
    Parses triplets from text and generates comma-separated triplets.
    """
    # Regex to find triplets (s, p, o)
    triplets = re.findall(r'\(([^,()]+),([^,()]+),([^,()]+)\)', text)
    
    if not triplets:
        return ""

    edges = []

    for s, p, o in triplets:
        s = s.strip()
        p = p.strip()
        o = o.strip()
        
        edges.append(f"({s}, {p}, {o})")

    # Join with comma
    return ", ".join(edges)

############################################
# 4. Validation
############################################

def validate_scene_graphfacts(text: str) -> bool:
    return "(" in text and ")" in text and "," in text

############################################
# 5. Build Graph Module (Facts only)
############################################

def build_graph_module(graph_facts: str) -> str:
    return graph_facts.strip()

############################################
# 6. Process One train_sft.json
############################################

def process_single_item(item: dict, idx: int, json_path: str) -> bool:
    """
    Process a single item. Modifies item in-place.
    Returns True if successful, False otherwise.
    """
    messages = item.get("messages", [])

    # 1. Get scene graph triplets from assistant
    scene_graph_text = None
    assistant_msg = None
    for msg in messages:
        if msg.get("role") == "assistant":
            scene_graph_text = extract_scene_graph_text(msg["content"])
            assistant_msg = msg
            break

    if scene_graph_text is None:
        return False

    # 2. Convert to Graph Facts (Rule-based)
    try:
        graph_facts = generate_graph_facts_rule_based(scene_graph_text)
        if not validate_scene_graphfacts(graph_facts):
             print(f"[Warn] Invalid scene graph facts generated in {os.path.basename(json_path)} item {idx}.")
             return False
    except Exception as e:
        print(f"[Error] Generation error in {os.path.basename(json_path)} item {idx}: {e}")
        return False

    graph_module = build_graph_module(graph_facts)

    # 3. Insert Graph Facts into ASSISTANT content
    if assistant_msg:
        # Replace original content entirely as it is redundant
        assistant_msg["content"] = graph_module

    # 4. Add tag to USER content
    for msg in messages:
        if msg.get("role") == "user":
            msg["content"] = "[scene_graph:scene_graph_generation]\n<image>\n\nGenerate a scene graph for this image."
            break
    
    return True

def process_one_file(json_path: str, workers: int = 32):
    print(f"[Start] Processing {json_path}")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to open {json_path}: {e}")
        return

    total_items = len(data)
    print(f"[{os.path.basename(json_path)}] Total items: {total_items}")

    # Determine output path: replace .json with _gskel.json
    dir_name = os.path.dirname(json_path)
    base_name = os.path.basename(json_path)
    name, ext = os.path.splitext(base_name)
    out_filename = f"{name}_gskel{ext}"
    out_path = os.path.join(dir_name, out_filename)

    # Parallelize item processing
    item_workers = workers
    valid_indices = set()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=item_workers) as executor:
        futures = {executor.submit(process_single_item, item, idx, json_path): idx for idx, item in enumerate(data)}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            completed_count += 1
            if completed_count % 100 == 0:
                remaining = total_items - completed_count
                print(f"[{os.path.basename(json_path)}] Processed: {completed_count}/{total_items}, Remaining: {remaining}")
            
            try:
                success = future.result()
                if success:
                    valid_indices.add(idx)
            except Exception as e:
                print(f"[Error] Thread exception for item in {json_path}: {e}")

            # Save intermediate results every 1000 items
            if completed_count % 1000 == 0:
                current_filtered_data = [data[i] for i in sorted(list(valid_indices))]
                print(f"[{os.path.basename(json_path)}] Saving intermediate result ({len(current_filtered_data)} items) to {out_path}...")
                with open(out_path, "w") as f:
                    json.dump(current_filtered_data, f, indent=2)

    # Filter data to keep only successfully processed items
    filtered_data = [data[i] for i in sorted(list(valid_indices))]
    print(f"[{os.path.basename(json_path)}] Filtered items: {len(filtered_data)}/{total_items}")

    with open(out_path, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"[OK] Generated {out_path}")

############################################
# 7. Test Function
############################################

def run_tests():
    print("=" * 80)
    print("RUNNING TEST CASE")
    print("=" * 80)

    test_item = { 
        "messages": [ 
            { "role": "user", "content": "<image>\n\nGenerate a scene graph for this image." }, 
            { "role": "assistant", "content": "The scene contains the following triplets: (laptop, on, bed); (pillow, on, bed); (bed, with, pillow); (chair, in, bed); (laptop, near, pillow)." } 
        ], 
        "images": [ "images/VG_100K/2355346.jpg" ] 
    }

    messages = test_item.get("messages", [])

    # 1. Get scene graph triplets from assistant
    scene_graph_text = None
    assistant_msg = None
    for msg in messages:
        if msg.get("role") == "assistant":
            scene_graph_text = extract_scene_graph_text(msg["content"])
            assistant_msg = msg
            break
    
    print(f"Extracted Scene Graph Text: {scene_graph_text}")

    if scene_graph_text:
        # 2. Convert to Graph Facts
        print("\nGenerating Graph Facts (Rule-based)...")
        
        try:
            graph_facts = generate_graph_facts_rule_based(scene_graph_text)
            print("\n--- Generated Graph Facts ---")
            print(graph_facts)

            if validate_scene_graphfacts(graph_facts):
                print("\n[Validation] PASSED")
                graph_module = build_graph_module(graph_facts)

                # 3. Insert Graph Facts into ASSISTANT content
                if assistant_msg:
                    # Replace original content entirely as it is redundant
                    assistant_msg["content"] = graph_module

                # 4. Add tag to USER content
                for msg in messages:
                    if msg.get("role") == "user":
                        msg["content"] = "[scene_graph:scene_graph_generation]\n<image>\n\nGenerate a scene graph for this image."
                        break
                
                print("\n--- Final JSON Output ---")
                print(json.dumps(test_item, indent=2))
            else:
                print("\n[Validation] FAILED")
        except Exception as e:
            print(f"Error: {e}")

############################################
# 8. Main
############################################

def main(paths: List[str], test_mode: bool = False, workers: int = 32):
    if test_mode:
        run_tests()
        return

    if not paths:
        print("No file paths provided.")
        return

    # Use ThreadPoolExecutor for concurrent processing
    # Reduced file-level workers to balance with item-level workers
    max_workers = min(len(paths), 4)
    print(f"Starting processing of {len(paths)} files with {max_workers} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_file, path, workers): path for path in paths}
        
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[Error] Exception processing {path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test case")
    parser.add_argument("--workers", type=int, default=64, help="Number of concurrent worker threads")
    args = parser.parse_args()

    if args.test:
        main([], test_mode=True)
    else:
        parser.add_argument("paths", nargs="+", help="JSON files to process")
        args = parser.parse_args()
        main(args.paths, workers=args.workers)
