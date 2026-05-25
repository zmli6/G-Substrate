import json
import os
import argparse
import concurrent.futures
import re
from typing import List, Tuple

############################################
# 1. Mol Graph Schema
############################################

MOL_GRAPH_SCHEMA = """

Edges: Represent bonds as triples in the format (node_i_id:node_i_element, bond_type, node_j_id:node_j_element). The bond_type occupies the central relationship slot (e.g., single, double, aromatic).
"""

############################################
# 2. Build Mol Graph-Fact Prompt
############################################

############################################
# 3. Extract Molecular Graph Text
############################################

def parse_mol_content(user_content: str) -> Tuple[str, str, str]:
    """
    Parses the user content into SMILES (optional), Graph Text, and Question.
    Returns: (smiles_part, graph_text, question_part)
    """
    # 1. Extract Question
    # Heuristic: The question usually starts with "What can you tell me" in this dataset
    question_marker = "What can you tell me"
    if question_marker in user_content:
        # Split by the last occurrence to be safe
        parts = user_content.rsplit(question_marker, 1)
        question_part = question_marker + parts[1]
        context_part = parts[0]
    else:
        question_part = ""
        context_part = user_content

    # 2. Extract Molecular Graph
    # Heuristic: "Molecular Graph:" separates SMILES (if any) from the graph description
    graph_marker = "Molecular Graph:"
    if graph_marker in context_part:
        parts = context_part.split(graph_marker)
        smiles_part = parts[0].strip()
        graph_text = parts[1].strip()
    else:
        smiles_part = ""
        graph_text = context_part.strip()

    return smiles_part, graph_text, question_part

def extract_mol_graph_text(user_content: str) -> str:
    """
    Legacy wrapper for backward compatibility if needed, 
    or just used to get the graph text for prompting.
    """
    _, graph_text, _ = parse_mol_content(user_content)
    return graph_text

############################################
# 4. Deterministic Parsing (Replaces vLLM Call)
############################################

def parse_graph_text_deterministic(graph_text: str) -> str:
    """
    Parses the molecular graph text using regex instead of an LLM.
    Input format: "Atom 1: Carbon (C). Neighbors: Atom 2 (single)."
    Returns edges only: (1:C, single, 2:O)
    """
    id_to_element = {}
    edges = set()

    # Regex for Atom line: "Atom 1: Carbon (C). Neighbors: ..."
    # Handles "Carbon (C)" and "Carbon (C, aromatic)"
    # Group 1: ID, Group 2: Element (before comma/paren), Group 3: Neighbors string
    atom_pattern = re.compile(r"Atom\s+(\d+):\s+[^(\n]*\(([^),]+)(?:,.*)?\)\.\s*Neighbors:\s*(.*)", re.IGNORECASE)
    
    # Regex for Neighbors: "Atom 2 (single)"
    neighbor_pattern = re.compile(r"Atom\s+(\d+)\s+\(([^)]+)\)")

    lines = graph_text.split('\n')
    
    # First pass: Parse all lines to populate id_to_element and collect edges
    for line in lines:
        line = line.strip()
        if not line: continue
        
        match = atom_pattern.search(line)
        if match:
            atom_id = int(match.group(1))
            element = match.group(2).strip()
            neighbors_str = match.group(3)
            
            id_to_element[atom_id] = element
            
            neighbor_matches = neighbor_pattern.findall(neighbors_str)
            for n_id_str, bond_type in neighbor_matches:
                n_id = int(n_id_str)
                u, v = atom_id, n_id
                # Store canonical edge (min, bond, max) to deduplicate
                if u < v:
                    edges.add((u, bond_type, v))
                elif v < u:
                    edges.add((v, bond_type, u))

    # Format Edges: Sort by IDs
    sorted_edges = sorted(list(edges), key=lambda x: (x[0], x[2]))
    edges_lines = []
    for u, bond, v in sorted_edges:
        u_elem = id_to_element.get(u, "?")
        v_elem = id_to_element.get(v, "?")
        edges_lines.append(f"({u}:{u_elem}, {bond}, {v}:{v_elem})")
    
    return ", ".join(edges_lines)

############################################
# 5. Validate Graph Facts (Lightweight)
############################################

def validate_mol_graphfacts(text: str) -> bool:
    # Relaxed validation: Check if there is at least one edge parenthesis
    return "(" in text

############################################
# 6. Build Graph Module
############################################

def build_graph_module(graph_facts: str) -> str:
    return f"""
The following is a molecular graph.

[Graph Schema]
{MOL_GRAPH_SCHEMA.strip()}

[Graph Facts]
{graph_facts.strip()}
""".strip()

############################################
# 7. Process One train_sft.json
############################################

def process_single_item(item: dict, idx: int, json_path: str) -> bool:
    """
    Process a single item. Modifies item in-place.
    Returns True if the item should be kept, False if it should be discarded.
    """
    try:
        for msg in item.get("messages", []):
            if msg.get("role") != "user":
                continue

            original_content = msg["content"]

            smiles_part, graph_text, question_part = parse_mol_content(original_content)
            
            # Skip if extraction seems empty
            if not graph_text:
                continue

            # Use deterministic parsing instead of LLM
            graph_facts = parse_graph_text_deterministic(graph_text)
            
            if not validate_mol_graphfacts(graph_facts):
                print(f"[Warn] Invalid molecular Graph Facts generated in {os.path.basename(json_path)} item {idx}. Discarding item.")
                return False

            graph_module = build_graph_module(graph_facts)

            # Reconstruct content: 
            # 1. [molecule:molecule_description] + Graph Module
            # 2. SMILES (if present)
            # 3. Question
            # We remove the original NL graph text to avoid duplication.
            
            new_content_parts = []
            new_content_parts.append(f"[molecule:molecule_description]\n{graph_module}")

            if smiles_part:
                new_content_parts.append(smiles_part)
            
            if question_part:
                new_content_parts.append(question_part)
            
            msg["content"] = "\n\n".join(new_content_parts)
        
        return True

    except Exception as e:
        print(f"[Error] Processing item {idx} in {json_path}: {e}")
        return False

def process_one_file(json_path: str, workers: int):
    print(f"[Start] Processing {json_path} with {workers} workers")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to open {json_path}: {e}")
        return

    total_items = len(data)
    print(f"[{os.path.basename(json_path)}] Total items: {total_items}")

    # Determine output path: replace .json with _gskel.json
    # e.g. smiles_graph.json -> smiles_graph_gskel.json
    dir_name = os.path.dirname(json_path)
    base_name = os.path.basename(json_path)
    name, ext = os.path.splitext(base_name)
    out_filename = f"{name}_gskel{ext}"
    out_path = os.path.join(dir_name, out_filename)

    # Parallelize item processing
    # Use the passed workers argument
    
    # Keep track of successful indices
    keep_indices = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_item, item, idx, json_path): idx for idx, item in enumerate(data)}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            completed_count += 1
            if completed_count % 100 == 0:
                remaining = total_items - completed_count
                print(f"[{os.path.basename(json_path)}] Processed: {completed_count}/{total_items}, Remaining: {remaining}")
            
            # Ensure exceptions are caught
            try:
                should_keep = future.result()
                if should_keep:
                    keep_indices.add(idx)
            except Exception as e:
                print(f"[Error] Thread exception for item in {json_path}: {e}")

            # Save checkpoint every 1000 items
            if completed_count % 1000 == 0:
                print(f"[{os.path.basename(json_path)}] Saving checkpoint at {completed_count} items to {out_path}...")
                current_filtered_data = [data[i] for i in range(total_items) if i in keep_indices]
                with open(out_path, "w") as f:
                    json.dump(current_filtered_data, f, indent=2)

    # Filter data based on keep_indices
    filtered_data = [data[i] for i in range(total_items) if i in keep_indices]
    discarded_count = total_items - len(filtered_data)
    print(f"[{os.path.basename(json_path)}] Finished. Kept: {len(filtered_data)}, Discarded: {discarded_count}")

    with open(out_path, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"[OK] Generated {out_path}")

############################################
# 8. Test Function
############################################

def run_tests():
    print("=" * 80)
    print("RUNNING TEST CASE (Deterministic)")
    print("=" * 80)

    test_content = """SMILES:
C/C=C/c1ccc2oc(-c3ccc(Oc4cc(-c5oc6ccc(/C=C/C)cc6c5C)ccc4O)cc3)c(C)c2c1

Molecular Graph:
Atom 1: Carbon (C). Neighbors: Atom 2 (single).
Atom 2: Carbon (C). Neighbors: Atom 1 (single), Atom 3 (double).
Atom 3: Carbon (C). Neighbors: Atom 2 (double), Atom 4 (single).
Atom 4: Carbon (C, aromatic). Neighbors: Atom 3 (single), Atom 5 (aromatic), Atom 40 (aromatic).
Atom 5: Carbon (C, aromatic). Neighbors: Atom 4 (aromatic), Atom 6 (aromatic).
Atom 6: Carbon (C, aromatic). Neighbors: Atom 5 (aromatic), Atom 7 (aromatic).
Atom 7: Carbon (C, aromatic). Neighbors: Atom 6 (aromatic), Atom 8 (aromatic), Atom 39 (aromatic).
Atom 8: Oxygen (O, aromatic). Neighbors: Atom 7 (aromatic), Atom 9 (aromatic).
Atom 9: Carbon (C, aromatic). Neighbors: Atom 8 (aromatic), Atom 10 (single), Atom 37 (aromatic).
Atom 10: Carbon (C, aromatic). Neighbors: Atom 9 (single), Atom 11 (aromatic), Atom 36 (aromatic).
Atom 11: Carbon (C, aromatic). Neighbors: Atom 10 (aromatic), Atom 12 (aromatic).
Atom 12: Carbon (C, aromatic). Neighbors: Atom 11 (aromatic), Atom 13 (aromatic).
Atom 13: Carbon (C, aromatic). Neighbors: Atom 12 (aromatic), Atom 14 (single), Atom 35 (aromatic).
Atom 14: Oxygen (O). Neighbors: Atom 13 (single), Atom 15 (single).
Atom 15: Carbon (C, aromatic). Neighbors: Atom 14 (single), Atom 16 (aromatic), Atom 33 (aromatic).
Atom 16: Carbon (C, aromatic). Neighbors: Atom 15 (aromatic), Atom 17 (aromatic).
Atom 17: Carbon (C, aromatic). Neighbors: Atom 16 (aromatic), Atom 18 (single), Atom 31 (aromatic).
Atom 18: Carbon (C, aromatic). Neighbors: Atom 17 (single), Atom 19 (aromatic), Atom 29 (aromatic).
Atom 19: Oxygen (O, aromatic). Neighbors: Atom 18 (aromatic), Atom 20 (aromatic).
Atom 20: Carbon (C, aromatic). Neighbors: Atom 19 (aromatic), Atom 21 (aromatic), Atom 28 (aromatic).
Atom 21: Carbon (C, aromatic). Neighbors: Atom 20 (aromatic), Atom 22 (aromatic).
Atom 22: Carbon (C, aromatic). Neighbors: Atom 21 (aromatic), Atom 23 (aromatic).
Atom 23: Carbon (C, aromatic). Neighbors: Atom 22 (aromatic), Atom 24 (single), Atom 27 (aromatic).
Atom 24: Carbon (C). Neighbors: Atom 23 (single), Atom 25 (double).
Atom 25: Carbon (C). Neighbors: Atom 24 (double), Atom 26 (single).
Atom 26: Carbon (C). Neighbors: Atom 25 (single).
Atom 27: Carbon (C, aromatic). Neighbors: Atom 23 (aromatic), Atom 28 (aromatic).
Atom 28: Carbon (C, aromatic). Neighbors: Atom 27 (aromatic), Atom 29 (aromatic), Atom 20 (aromatic).
Atom 29: Carbon (C, aromatic). Neighbors: Atom 28 (aromatic), Atom 30 (single), Atom 18 (aromatic).
Atom 30: Carbon (C). Neighbors: Atom 29 (single).
Atom 31: Carbon (C, aromatic). Neighbors: Atom 17 (aromatic), Atom 32 (aromatic).
Atom 32: Carbon (C, aromatic). Neighbors: Atom 31 (aromatic), Atom 33 (aromatic).
Atom 33: Carbon (C, aromatic). Neighbors: Atom 32 (aromatic), Atom 34 (single), Atom 15 (aromatic).
Atom 34: Oxygen (O). Neighbors: Atom 33 (single).
Atom 35: Carbon (C, aromatic). Neighbors: Atom 13 (aromatic), Atom 36 (aromatic).
Atom 36: Carbon (C, aromatic). Neighbors: Atom 35 (aromatic), Atom 10 (aromatic).
Atom 37: Carbon (C, aromatic). Neighbors: Atom 9 (aromatic), Atom 38 (single), Atom 39 (aromatic).
Atom 38: Carbon (C). Neighbors: Atom 37 (single).
Atom 39: Carbon (C, aromatic). Neighbors: Atom 37 (aromatic), Atom 40 (aromatic), Atom 7 (aromatic).
Atom 40: Carbon (C, aromatic). Neighbors: Atom 39 (aromatic), Atom 4 (aromatic).

What can you tell me about this molecule?"""

    print("Extracting graph text...")
    smiles_part, graph_text, question_part = parse_mol_content(test_content)
    print(f"SMILES Part Length: {len(smiles_part)} chars")
    print(f"Graph Text Length: {len(graph_text)} chars")
    print(f"Question Part Length: {len(question_part)} chars")
    
    print("\nParsing graph text deterministically...")

    try:
        graph_facts = parse_graph_text_deterministic(graph_text)
        print("\n--- Generated Graph Facts ---")
        print(graph_facts)
        
        if validate_mol_graphfacts(graph_facts):
            print("\n[Validation] PASSED")
            graph_module = build_graph_module(graph_facts)
            
            new_content_parts = []
            new_content_parts.append(f"[molecule:molecule_description]\n{graph_module}")
            if smiles_part:
                new_content_parts.append(smiles_part)
            if question_part:
                new_content_parts.append(question_part)
            
            final_content = "\n\n".join(new_content_parts)
            
            print("\n--- Final Content Preview (First 500 chars) ---")
            print(final_content)
        else:
            print("\n[Validation] FAILED")

    except Exception as e:
        print(f"Error: {e}")

############################################
# 9. Main
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
    max_file_workers = min(len(paths), 4)
    print(f"Starting processing of {len(paths)} files with {max_file_workers} file-threads and {workers} item-threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_file_workers) as executor:
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
    parser.add_argument("--workers", type=int, default=32, help="Number of concurrent worker threads per file")
    args = parser.parse_args()

    if args.test:
        main([], test_mode=True)
    else:
        parser.add_argument("paths", nargs="+", help="JSON files to process")
        args = parser.parse_args()
        main(args.paths, workers=args.workers)
