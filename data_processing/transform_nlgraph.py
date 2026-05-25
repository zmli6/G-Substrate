import json
import os
import sys
import glob
import re
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

############################################
# 1. Graph Schema Definitions
############################################

UNIGRASP_SCHEMA = """
Edges: Represent each connection as a triple in the format (node_i, -, node_j). The - symbol acts as a universal placeholder for "connectivity" to maintain slot-consistency with other graph types.
"""

GRAPH_SCHEMAS = {
    "undirected": UNIGRASP_SCHEMA,

    "flow": """
Edges: Represent each connection as a triple in the format (source_node, capacity: <value>, target_node).
""",

    "weighted": """
Edges: Represent each connection as a triple in the format (node_i, weight: <value>, node_j).
""",

    "matching": """
Edges: Represent each interest as a triple in the format (ApplicantID:Applicant, -, JobID:Job).
"""
}

############################################
# 2. Task → Prompt Mapping (Schema-aware)
############################################

BASE_INSTRUCTION = """
Instruction:
1. Analyze the graph description carefully.
2. Extract every edge.
3. Convert them into the specified UniGrasp format.
4. Do NOT output the original question or any reasoning.
5. Do NOT solve the problem.
6. Output ONLY the facts starting with Edges:.
7. Do not repeat the output.
"""

# Task-specific ICL Examples
EX_UNDIRECTED = """
Example:
[Input]
"The nodes are numbered from 0 to 8, and the edges are: (0,2) (0,4) (1,2) (1,8) (1,4) (1,3) (3,8) (3,4) (4,8) (5,7) (5,6) (6,7)."

[Output]
Edges: (0, -, 2), (0, -, 4), (1, -, 2), (1, -, 8), (1, -, 4), (1, -, 3), (3, -, 8), (3, -, 4), (4, -, 8), (5, -, 7), (5, -, 6), (6, -, 7)
"""

EX_FLOW = """
Example:
[Input]
"In a directed graph, edge from node 0 to node 1 with capacity 10. Node 2 is connected to node 1 with capacity 5."

[Output]
Edges: (0, capacity: 10, 1), (2, capacity: 5, 1)
"""

EX_WEIGHTED = """
Example:
[Input]
"Edge between node 0 and node 1 with weight 5. Edge between node 1 and node 2 with weight 3."

[Output]
Edges: (0, weight: 5, 1), (1, weight: 3, 2)
"""

EX_MATCHING = """
Example:
[Input]
"Applicant 0 is interested in job 1. Applicant 0 is interested in job 2. Applicant 1 is interested in job 2."

[Output]
Edges: (0:Applicant, -, 1:Job), (0:Applicant, -, 2:Job), (1:Applicant, -, 2:Job)
"""

TASK_CONFIG = {
    "connectivity": {
        "schema": GRAPH_SCHEMAS["undirected"],
        "instruction": BASE_INSTRUCTION,
        "example": EX_UNDIRECTED
    },

    "cycle": {
        "schema": GRAPH_SCHEMAS["undirected"],
        "instruction": BASE_INSTRUCTION,
        "example": EX_UNDIRECTED
    },

    "hamilton": {
        "schema": GRAPH_SCHEMAS["undirected"],
        "instruction": BASE_INSTRUCTION,
        "example": EX_UNDIRECTED
    },

    "flow": {
        "schema": GRAPH_SCHEMAS["flow"],
        "instruction": BASE_INSTRUCTION,
        "example": EX_FLOW
    },

    "shortest_path": {
        "schema": GRAPH_SCHEMAS["weighted"],
        "instruction": BASE_INSTRUCTION,
        "example": EX_WEIGHTED
    },

    "matching": {
        "schema": GRAPH_SCHEMAS["matching"],
        "instruction": BASE_INSTRUCTION,
        "example": EX_MATCHING
    }
}

############################################
# 3. Task Detection
############################################

def detect_task_from_content(content: str) -> str:
    """
    Detects the graph task based on keywords in the user's question/description.
    """
    content_lower = content.lower()
    
    # Priority based matching to avoid overlaps
    if "maximum flow" in content_lower:
        return "flow"
    if "shortest path" in content_lower:
        return "shortest_path"
    if "assignment of jobs" in content_lower or "interested in job" in content_lower or "interested in task" in content_lower or "hosts" in content_lower:
        return "matching"
    if "visits every node exactly once" in content_lower or "hamilton" in content_lower:
        return "hamilton"
    if "cycle" in content_lower:
        return "cycle"
    # Connectivity usually asks "Is there a path..." or "path between"
    # We check this last as "path" is a common word
    if "path between" in content_lower or "connected" in content_lower:
        return "connectivity"
        
    return "unknown"

def detect_task_from_path(path: str) -> str:
    lower = path.lower()
    # Check for exact matches or substrings in config keys
    for task in TASK_CONFIG:
        if task in lower:
            return task
    # Fallback for specific naming conventions if necessary
    if "topology" in lower: return "connectivity"
    
    raise ValueError(f"[ERROR] Cannot detect task from path: {path}")

############################################
# 4. Extract Graph Description & Question
############################################

def split_user_content(user_content: str) -> Tuple[str, str]:
    """
    Splits user content into (graph_description, question).
    Handles 'Q:' separator or implicit questions at the end.
    """
    # 1. Explicit Q: separator
    if "Q:" in user_content:
        parts = user_content.split("Q:", 1)
        graph_text = parts[0].strip()
        question_text = "Q:" + parts[1]
        return graph_text, question_text
    
    # 2. Implicit question at the end (heuristic based on common starts)
    # Look for the last newline
    last_newline_idx = user_content.rfind('\n')
    if last_newline_idx != -1:
        potential_question = user_content[last_newline_idx+1:].strip()
        # Common question starters in this dataset
        question_starters = ["Is there", "What is", "Give the", "Find"]
        if any(potential_question.startswith(q) for q in question_starters):
            graph_text = user_content[:last_newline_idx].strip()
            question_text = potential_question
            return graph_text, question_text

    # 3. Fallback: assume everything is graph description if no question found
    return user_content.strip(), ""

############################################
# 5. Build Graph-Fact Conversion Prompt
############################################

def build_graphfact_prompt(task: str, graph_text: str) -> str:
    cfg = TASK_CONFIG[task]

    prompt = f"""
{cfg["instruction"]}

{cfg["schema"]}

{cfg["example"]}

---
Graph Description:
{graph_text}
---

new graph:
""".strip()

    return prompt

############################################
# 7. Graph Fact Validation & Verification
############################################

def validate_syntax(graph_facts: str) -> bool:
    """
    Lightweight syntax check.
    """
    keywords = ["node(", "edge(", "applicant(", "job(", "interested(", "Edges:"]
    return any(k in graph_facts for k in keywords)

############################################
# 8. Assemble Graph Module (Schema + Facts)
############################################

def build_graph_module(task: str, schema: str, graph_facts: str) -> str:
    graph_type_map = {
        "connectivity": "undirected",
        "cycle": "undirected",
        "hamilton": "undirected",
        "shortest_path": "weighted",
        "flow": "flow",
        "matching": "matching"
    }
    
    graph_type = graph_type_map.get(task, task)

    return f"""
Here is a description of a {graph_type} graph.

[Graph Schema]
{schema.strip()}

[Graph Facts]
{graph_facts.strip()}
""".strip()

############################################
# 8.5 Rule-Based Extraction (No LLM)
############################################

def extract_graph_rule_based(task: str, text: str) -> Optional[str]:
    """
    Attempts to extract graph facts using regex patterns tailored to NLGraph/GVLQA datasets.
    Returns formatted graph facts string or None if extraction fails.
    """
    # nodes set is used for internal tracking if needed, but not output
    edges_list = []
    nodes = set()
    
    # Helper to add range nodes
    range_match = re.search(r'numbered from (\d+) to (\d+)', text)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        for i in range(start, end + 1):
            nodes.add(str(i))

    if task in ["connectivity", "cycle", "hamilton"]:
        # Pattern 1: (0,2) (0,4) ... with optional spaces
        matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
        for u, v in matches:
            edges_list.append(f"({u}, -, {v})")
            
        # Pattern 2: node i connected to node j
        matches_conn = re.findall(r'node\s+(\d+)\s+(?:is\s+)?connected\s+to\s+node\s+(\d+)', text, re.IGNORECASE)
        for u, v in matches_conn:
            edges_list.append(f"({u}, -, {v})")

        # Pattern 3: edge between node i and node j
        matches_edge = re.findall(r'edge\s+between\s+node\s+(\d+)\s+and\s+node\s+(\d+)', text, re.IGNORECASE)
        for u, v in matches_edge:
            edges_list.append(f"({u}, -, {v})")
            
    elif task == "flow":
        # Pattern: edge from node X to node Y with capacity Z
        matches = re.findall(r'edge\s+from\s+node\s+(\d+)\s+to\s+node\s+(\d+)\s+with\s+capacity\s+(\d+)', text, re.IGNORECASE)
        for u, v, cap in matches:
            edges_list.append(f"({u}, capacity: {cap}, {v})")

    elif task == "shortest_path":
        # Pattern: edge between node X and node Y with weight Z
        matches = re.findall(r'edge\s+between\s+node\s+(\d+)\s+and\s+node\s+(\d+)\s+with\s+weight\s+(\d+)', text, re.IGNORECASE)
        for u, v, w in matches:
            edges_list.append(f"({u}, weight: {w}, {v})")

    elif task == "matching":
        # Pattern: Applicant X is interested in job Y / Host X is interested in task Y
        matches = re.findall(r'(Applicant|Host)\s+(\d+)\s+is\s+interested\s+in\s+(job|task)\s+(\d+)', text, re.IGNORECASE)
        for subj_type, subj_id, obj_type, obj_id in matches:
            # Format: ID:Label
            s_node = f"{subj_id}:{subj_type.capitalize()}"
            o_node = f"{obj_id}:{obj_type.capitalize()}"
            edges_list.append(f"({s_node}, -, {o_node})")

    else:
        return None

    if not edges_list:
        return None

    # Deduplicate and sort edges for deterministic output
    unique_edges = sorted(list(set(edges_list)))
    
    edges_str = "Edges: " + ", ".join(unique_edges)
    
    return edges_str

############################################
# 9. Process One train_sft.json
############################################

def process_single_item(item: dict, i: int, json_path: str) -> List[str]:
    """
    Process a single item from the dataset.
    Modifies 'item' in-place. Returns a list of log messages.
    """
    logs = []
    
    # Normalize ShareGPT to OpenAI format
    if "conversations" in item:
        item["messages"] = []
        for c in item["conversations"]:
            role = "user" if c["from"] == "human" else "assistant"
            item["messages"].append({
                "role": role,
                "content": c["value"]
            })
        del item["conversations"]

    if "messages" not in item:
        return logs

    msgs = item["messages"]
    
    for msg in msgs:
        if msg.get("role") != "user":
            continue

        original_content = msg["content"]
        # Skip if already processed
        if "[Graph Facts]" in original_content:
            continue

        # 1) Detect Task per item
        task = detect_task_from_content(original_content)
        if task == "unknown":
            # Fallback to path detection if content detection fails
            try:
                task = detect_task_from_path(json_path)
            except ValueError:
                # logs.append(f"  [Warn] Item {i}: Unknown task. Skipping.")
                continue
        
        schema = TASK_CONFIG[task]["schema"]

        graph_text, question_text = split_user_content(original_content)

        # 2) Try Rule-Based Extraction Only
        graph_facts = extract_graph_rule_based(task, graph_text)
        
        # 3) If Rule-Based fails, skip
        if not graph_facts:
            # logs.append(f"  [Warn] Item {i} (Task: {task}): Rule-based extraction failed. Skipping.")
            continue

        # 4) Validate Syntax (Applies to both methods)
        if not validate_syntax(graph_facts):
            # logs.append(f"  [Warn] Item {i} (Task: {task}): Invalid syntax in generated facts.")
            continue
        
        # 5) Build Graph Module
        graph_module = build_graph_module(task, schema, graph_facts)

        # 6) Replace user content
        # Format: [graph_search:{task}] \n {schema+facts} \n\n {question}
        new_content = f"[graph_search:{task}]\n{graph_module}\n\n{question_text}".strip()
        msg["content"] = new_content
            
    return logs

def process_one_file(json_path: str) -> Tuple[str, List[str]]:
    logs = []
    # print(f"Processing {json_path}...") 

    with open(json_path, "r") as f:
        data = json.load(f)

    total_items = len(data)
    
    # Parallelize item processing within the file
    # Adjust max_workers based on your vLLM server capacity and file-level concurrency in main()
    # If main() has 4 workers and here we have 8, total concurrency is 32.
    item_workers = 8 
    
    with ThreadPoolExecutor(max_workers=item_workers) as executor:
        # Submit all items
        futures = {executor.submit(process_single_item, item, i, json_path): i for i, item in enumerate(data)}
        
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            if completed_count % 1000 == 0:
                tqdm.write(f"[{os.path.basename(json_path)}] Processed: {completed_count}/{total_items}")
            
            try:
                item_logs = future.result()
                logs.extend(item_logs)
            except Exception as e:
                logs.append(f"  [Error] Unhandled exception in item thread: {e}")

    # Determine output path: *graph_search*.json -> *graph_search*_gskel.json
    base, ext = os.path.splitext(json_path)
    output_path = f"{base}_gskel{ext}"

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # print(f"[OK] Generated: {output_path}")
    return output_path, logs

############################################
# 12. Task Detection Analysis (New)
############################################

def analyze_json_tasks(json_path: str):
    """
    Analyzes a JSON file to determine task distribution and check for ambiguities.
    """
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
    
    stats = {k: 0 for k in TASK_CONFIG.keys()}
    stats["unknown"] = 0
    multi_label_count = 0
    
    print(f"Analyzing tasks in: {json_path}")
    print("-" * 60)

    for i, item in enumerate(data):
        # Normalize ShareGPT to OpenAI format for analysis
        if "conversations" in item:
            msgs = []
            for c in item["conversations"]:
                role = "user" if c["from"] == "human" else "assistant"
                msgs.append({
                    "role": role,
                    "content": c["value"]
                })
        elif "messages" in item:
            msgs = item["messages"]
        else:
            continue

        user_msg = next((m for m in msgs if m["role"] == "user"), None)
        if not user_msg:
            continue
            
        content = user_msg["content"]
        content_lower = content.lower()
        
        # Check for all potential matches to detect ambiguity
        matched_tasks = []
        if "maximum flow" in content_lower: matched_tasks.append("flow")
        if "shortest path" in content_lower: matched_tasks.append("shortest_path")
        if "assignment of jobs" in content_lower or "interested in job" in content_lower or "interested in task" in content_lower or "hosts" in content_lower: matched_tasks.append("matching")
        if "visits every node exactly once" in content_lower or "hamilton" in content_lower: matched_tasks.append("hamilton")
        if "cycle" in content_lower: matched_tasks.append("cycle")
        
        # Connectivity check
        if "path between" in content_lower or "connected" in content_lower: 
            matched_tasks.append("connectivity")
        
        # Determine the primary task using the strict function
        primary_task = detect_task_from_content(content)
        stats[primary_task] = stats.get(primary_task, 0) + 1
        
        # Report ambiguity if multiple keywords are found
        if len(matched_tasks) > 1:
            multi_label_count += 1
            # print(f"[Ambiguity] Item {i} matches keywords for: {matched_tasks} -> Resolved to: {primary_task}")

    print("-" * 60)
    print("Task Distribution (Resolved):")
    for task, count in stats.items():
        if count > 0:
            print(f"  {task}: {count}")
    print(f"  unknown: {stats['unknown']}")
    print("-" * 60)
    print(f"Total items: {len(data)}")
    print(f"Items with multiple keyword matches: {multi_label_count}")
    print("=" * 60)

############################################
# 10. Test Cases (from User)
############################################

TEST_DATA = {
    "connectivity": [
        # JSON 1
        """Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an undirected edge.
Graph: (0,8) (0,1) (0,6) (0,2) (0,3) (0,7) (0,5) (1,8) (1,6) (1,2) (1,3) (1,7) (1,5) (2,8) (2,6) (2,3) (2,7) (2,5) (3,8) (3,6) (3,7) (3,5) (5,8) (5,6) (5,7) (6,8) (6,7) (7,8)
Q: Is there a path between node 8 and node 2?
A:""",
        # JSON 2
        """In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.
The nodes are numbered from 0 to 8, and the edges are: (0,2) (0,4) (1,2) (1,8) (1,4) (1,3) (3,8) (3,4) (4,8) (5,7) (5,6) (6,7)
Is there a path between node 8 and node 1 in this undirected graph?"""
    ],
    "cycle": [
        # JSON 1
        """In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.
The nodes are numbered from 0 to 24, and the edges are: (19,23) (12,23) (16,2) (2,15) (10,8) (2,20) (3,0) (2,0) (22,2) (21,11) (12,14) (4,21) (17,21) (0,14) (18,6) (3,14) (7,1) (4,24) (6,9) (9,20) (7,21) (5,9) (15,8) (2,13) (23,9) (19,1) (0,1)
Q: Is there a cycle in this graph?
A:""",
        # JSON 2
        """In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.
The nodes are numbered from 0 to 9, and the edges are: (2,0) (6,1) (8,7) (8,9) (2,6) (1,0) (4,7) (7,3) (4,5) (6,3) (0,3)<
Is there a cycle in this undirected graph?"""
    ],
    "flow": [
        # JSON 1
        """In a directed graph, the nodes are numbered from 0 to 6, and the edges are:
an edge from node 1 to node 5 with capacity 2,
an edge from node 1 to node 0 with capacity 4,
an edge from node 2 to node 5 with capacity 10,
an edge from node 2 to node 4 with capacity 5,
an edge from node 2 to node 3 with capacity 3,
an edge from node 2 to node 1 with capacity 6,
an edge from node 2 to node 6 with capacity 9,
an edge from node 3 to node 5 with capacity 10,
an edge from node 3 to node 4 with capacity 8,
an edge from node 3 to node 0 with capacity 7,
an edge from node 4 to node 2 with capacity 10,
an edge from node 4 to node 0 with capacity 1,
an edge from node 5 to node 2 with capacity 2,
an edge from node 5 to node 0 with capacity 3,
an edge from node 5 to node 3 with capacity 2,
an edge from node 6 to node 5 with capacity 10.
Q: What is the maximum flow from node 2 to node 3?
A:""",
        # JSON 2
        """In a directed graph, the nodes are numbered from 0 to 8, and the edges are:
an edge from node 0 to node 3 with capacity 7,
an edge from node 0 to node 2 with capacity 9,
an edge from node 1 to node 7 with capacity 10,
an edge from node 1 to node 6 with capacity 8,
an edge from node 1 to node 3 with capacity 4,
an edge from node 2 to node 1 with capacity 3,
an edge from node 2 to node 5 with capacity 9,
an edge from node 3 to node 5 with capacity 2,
an edge from node 3 to node 6 with capacity 4,
an edge from node 4 to node 8 with capacity 4,
an edge from node 4 to node 0 with capacity 9,
an edge from node 5 to node 6 with capacity 1,
an edge from node 7 to node 1 with capacity 1,
an edge from node 7 to node 8 with capacity 8,
an edge from node 7 to node 0 with capacity 6,
an edge from node 7 to node 2 with capacity 9,
an edge from node 8 to node 3 with capacity 9,
an edge from node 8 to node 2 with capacity 8,
What is the maximum flow from node 0 to node 6:"""
    ],
    "hamilton": [
        # JSON 1
        """In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.
The nodes are numbered from 0 to 8, and the edges are: (0,4) (0,2) (0,6) (0,7) (0,1) (1,5) (2,3) (2,6) (2,5) (3,4) (3,7) (4,7) (4,6) (5,6) (5,7) (6,8) (7,8)
Q: Is there a path in this graph that visits every node exactly once? If yes, give the path. Note that in a path, adjacent nodes must be connected with edges.
A:""",
        # JSON 2
        """In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.
The nodes are numbered from 0 to 9, and the edges are: (2,4) (0,5) (5,4) (2,7) (4,7) (3,9) (3,4) (6,7) (1,0) (5,2) (8,2) (8,1) (7,0) (6,2) (6,1) (3,8) (8,0) (8,7)
Q: Is there a path in this graph that visits every node exactly once? If yes, give the path. Note that in a path, adjacent nodes must be connected with edges."""
    ],
    "matching": [
        # JSON 1
        """There are 7 job applicants numbered from 0 to 6, and 5 jobs numbered from 0 to 4. Each applicant is interested in some of the jobs. Each job can only accept one applicant and a job applicant can be appointed for only one job.
Applicant 0 is interested in job 2.
Applicant 0 is interested in job 0.
Applicant 0 is interested in job 3.
Applicant 1 is interested in job 3.
Applicant 1 is interested in job 2.
Applicant 1 is interested in job 4.
Applicant 2 is interested in job 3.
Applicant 2 is interested in job 0.
Applicant 3 is interested in job 3.
Applicant 3 is interested in job 2.
Applicant 3 is interested in job 1.
Applicant 4 is interested in job 1.
Applicant 4 is interested in job 3.
Applicant 4 is interested in job 2.
Applicant 5 is interested in job 2.
Applicant 5 is interested in job 3.
Applicant 5 is interested in job 1.
Applicant 6 is interested in job 3.
Applicant 6 is interested in job 1.
Q: Find an assignment of jobs to applicants in such that the maximum number of applicants find the job they are interested in.
A:""",
        # JSON 2
        """There are 7 hosts numbered from 0 to 6, and 7 tasks numbered from 0 to 6. Each host has a set of tasks that it is interested in: 
Host 5 is interested in task 2.
Host 0 is interested in task 1.
Host 3 is interested in task 1.
Host 6 is interested in task 2.
Host 6 is interested in task 6.
Host 1 is interested in task 6.
Host 2 is interested in task 0.
Host 1 is interested in task 4.
Host 3 is interested in task 0.
Host 4 is interested in task 4.
Host 5 is interested in task 6.
Host 6 is interested in task 5.
Host 1 is interested in task 1.
Host 0 is interested in task 4.
Host 0 is interested in task 0.
Host 5 is interested in task 0.
Host 3 is interested in task 6.
Host 4 is interested in task 0.
Host 2 is interested in task 2.
Host 0 is interested in task 6.
Host 0 is interested in task 2.
Host 0 is interested in task 5.
Host 6 is interested in task 4. However, each host is capable of solving only one task, and similarly, each task can be resolved by just one host.
Q:  What is the maximum number of hosts that can be assigned a task they are interested in?"""
    ],
    "shortest_path": [
        # JSON 1
        """In an undirected graph, the nodes are numbered from 0 to 12, and the edges are:
an edge between node 0 and node 12 with weight 5,
an edge between node 0 and node 2 with weight 4,
an edge between node 0 and node 3 with weight 2,
an edge between node 1 and node 3 with weight 3,
an edge between node 2 and node 3 with weight 4,
an edge between node 2 and node 5 with weight 8,
an edge between node 2 and node 10 with weight 1,
an edge between node 3 and node 6 with weight 10,
an edge between node 3 and node 9 with weight 3,
an edge between node 3 and node 7 with weight 6,
an edge between node 3 and node 8 with weight 5,
an edge between node 4 and node 12 with weight 5,
an edge between node 5 and node 7 with weight 7,
an edge between node 5 and node 11 with weight 6,
an edge between node 7 and node 11 with weight 2,
an edge between node 8 and node 9 with weight 1,
an edge between node 9 and node 12 with weight 9,
an edge between node 9 and node 11 with weight 5,
an edge between node 9 and node 10 with weight 5,
an edge between node 10 and node 12 with weight 6,
an edge between node 10 and node 11 with weight 1.
Q: Give the shortest path from node 4 to node 2.
A:""",
        # JSON 2
        """In a undirected graph, the nodes are numbered from 0 to 9, and the edges are:
an edge between node 0 and node 6 with weight 2,
an edge between node 1 and node 0 with weight 2,
an edge between node 3 and node 0 with weight 3,
an edge between node 5 and node 0 with weight 3,
an edge between node 4 and node 1 with weight 1,
an edge between node 6 and node 1 with weight 1,
an edge between node 2 and node 1 with weight 2,
an edge between node 1 and node 3 with weight 3,
an edge between node 8 and node 1 with weight 1,
an edge between node 9 and node 2 with weight 1,
an edge between node 2 and node 4 with weight 2,
an edge between node 2 and node 6 with weight 1,
an edge between node 2 and node 3 with weight 2,
an edge between node 8 and node 2 with weight 3,
an edge between node 4 and node 6 with weight 2,
an edge between node 5 and node 4 with weight 3,
an edge between node 5 and node 9 with weight 4,
an edge between node 5 and node 6 with weight 4,
an edge between node 5 and node 7 with weight 3,
an edge between node 5 and node 8 with weight 3,
an edge between node 9 and node 7 with weight 3,
an edge between node 8 and node 9 with weight 4,
Q: Give the shortest path from node 9 to node 4:"""
    ]
}

def run_tests():
    print("=" * 80)
    print("RUNNING TEST CASES")
    print("=" * 80)

    for task, examples in TEST_DATA.items():
        print(f"\n>>> TEST TASK: {task}")
        print("-" * 80)
        
        for idx, content in enumerate(examples):
            print(f"\n[Example {idx+1}]")
            graph_text, question_text = split_user_content(content)
            # prompt = build_graphfact_prompt(task, graph_text)
            
            print("Attempting Rule-Based Extraction...")
            graph_facts = extract_graph_rule_based(task, graph_text)
            
            if graph_facts:
                print("[Rule-Based] Success!")
                
                # Basic validation check
                if validate_syntax(graph_facts):
                    print("[Validation] Syntax Check: PASSED")
                    
                    # Construct the final JSON block simulation
                    schema = TASK_CONFIG[task]["schema"]
                    graph_module = build_graph_module(task, schema, graph_facts)
                    
                    # Reconstruct content with header and without original graph description
                    new_content = f"[graph_search:{task}]\n{graph_module}\n\n{question_text}".strip()

                    # Create a dummy JSON structure representing the transformed data
                    # Always use "messages" format for output consistency as requested
                    final_json_block = {
                        "messages": [
                            {
                                "role": "user", 
                                "content": new_content
                            },
                            {
                                "role": "assistant",
                                "content": "(Original Answer)"
                            }
                        ]
                    }
                    
                    print("\n--- [Reconstructed JSON Block] ---")
                    print(json.dumps(final_json_block, indent=2, ensure_ascii=False))
                    
                else:
                    print("[Validation] Syntax Check: FAILED")
                    print("\n--- [Raw Output] ---")
                    print(graph_facts)
            else:
                print("[Rule-Based] Failed. Skipping.")

        print("=" * 80)

############################################
# 11. Main Entry
############################################

def main(paths: List[str], test_mode: bool = False, analyze_mode: bool = False):
    if test_mode:
        run_tests()
        return

    # 1. Collect files matching *graph_search*.json
    target_files = []
    for path in paths:
        if os.path.isdir(path):
            # Search for *graph_search*.json in the directory
            search_pattern = os.path.join(path, "*graph_search*.json")
            found = glob.glob(search_pattern)
            target_files.extend(found)
        elif os.path.isfile(path):
            if "graph_search" in os.path.basename(path) and path.endswith(".json"):
                target_files.append(path)
    
    # Exclude already generated files (*_gskel.json) to avoid loops or reprocessing output
    target_files = [f for f in target_files if not f.endswith("_gskel.json")]
    
    # Remove duplicates
    target_files = list(set(target_files))

    if not target_files:
        print("No files matching '*graph_search*.json' found in provided paths.")
        return

    print(f"Found {len(target_files)} files to process.")

    if analyze_mode:
        for f in target_files:
            analyze_json_tasks(f)
        return

    # 2. Parallel Processing
    # Use ThreadPoolExecutor because the bottleneck is likely the network call to vLLM
    # Reduced file-level workers slightly to balance with item-level workers
    max_workers = 4  
    
    print(f"Starting processing with {max_workers} file workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_one_file, f): f for f in target_files}
        
        # Use tqdm for progress bar
        with tqdm(total=len(target_files), desc="Processing Files") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    output_path, logs = future.result()
                    
                    # Print logs safely
                    if logs:
                        tqdm.write(f"--- Logs for {os.path.basename(file_path)} ---")
                        for log in logs:
                            tqdm.write(log)
                    
                    # Update postfix to show which file just finished
                    pbar.set_postfix(last_finished=os.path.basename(file_path)[:20] + "...")
                    
                except Exception as e:
                    tqdm.write(f"[Error] Failed to process {file_path}: {e}")
                finally:
                    pbar.update(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test cases")
    parser.add_argument("--analyze", action="store_true", help="Analyze task distribution in JSON files")
    parser.add_argument("paths", nargs="*", help="List of paths (files or directories) to process")
    args = parser.parse_args()

    if args.test:
        # In test mode, paths are not required
        main([], test_mode=True)
    else:
        # Default paths if none provided
        target_paths = args.paths
        if not target_paths:
            print("Please provide input paths as arguments.")
            exit(1)
        main(target_paths, test_mode=False, analyze_mode=args.analyze)
