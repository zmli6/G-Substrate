import json
import re
import random
import networkx as nx
import copy
import argparse
import os
from typing import List, Dict, Any, Tuple

class GraphParser:
    def __init__(self):
        # Regex for Scene Graph (Old)
        self.sg_entity_re = re.compile(r"entity\((\d+),\s*(.+?)\)")
        self.sg_rel_re = re.compile(r"relation\((\d+),\s*(.+?),\s*(\d+)\)")
        
        # Regex for Event Graph (Old)
        self.eg_event_re = re.compile(r"event\(([^,]+),\s*\"(.+?)\"\)")
        self.eg_rel_re = re.compile(r"relation\(([^,]+),\s*(.+?),\s*([^,]+)\)")
        
        # Regex for Tuple Format (New SG/EG)
        # Matches (a, b, c)
        self.tuple_re = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
        
        # Regex for Graph Search (NL) - Expanded based on transform-NLGraph.py
        self.gs_edge_re = re.compile(r"edge from node (\d+) to node (\d+)(?: with capacity (\d+))?")
        self.gs_triple_re = re.compile(r"\(([^,]+),\s*weight:\s*(\d+),\s*([^)]+)\)")
        
        # Regex for Structured Graph Search (Output of graph_to_search_text)
        self.gs_struct_capacity = re.compile(r"\(([^,]+),\s*capacity:\s*(\d+),\s*([^)]+)\)")
        self.gs_struct_generic = re.compile(r"\(([^,]+),\s*-,\s*([^)]+)\)")
        self.gs_struct_label = re.compile(r"\(([^,]+),\s*label:\s*\"(.+?)\",\s*([^)]+)\)")
        
        # Additional patterns from transform-NLGraph.py
        self.gs_undirected_tuple = re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)')
        self.gs_connected_to = re.compile(r'node\s+(\d+)\s+(?:is\s+)?connected\s+to\s+node\s+(\d+)', re.IGNORECASE)
        self.gs_edge_between = re.compile(r'edge\s+between\s+node\s+(\d+)\s+and\s+node\s+(\d+)', re.IGNORECASE)
        self.gs_flow = re.compile(r'edge\s+from\s+node\s+(\d+)\s+to\s+node\s+(\d+)\s+with\s+capacity\s+(\d+)', re.IGNORECASE)
        self.gs_weight = re.compile(r'edge\s+between\s+node\s+(\d+)\s+and\s+node\s+(\d+)\s+with\s+weight\s+(\d+)', re.IGNORECASE)
        self.gs_matching = re.compile(r'(Applicant|Host)\s+(\d+)\s+is\s+interested\s+in\s+(job|task)\s+(\d+)', re.IGNORECASE)
        self.gs_node_range = re.compile(r'numbered from (\d+) to (\d+)', re.IGNORECASE)

    def parse(self, text: str, mode: str) -> nx.DiGraph:
        G = nx.DiGraph()
        
        if mode == 'scene_graph':
            # Try New Format: (subj, pred, obj)
            triples = self.tuple_re.findall(text)
            if triples and "entity(" not in text:
                node_map = {} # label -> id
                next_id = 0
                for s, p, o in triples:
                    s, p, o = s.strip(), p.strip(), o.strip()
                    # Simple ID generation based on label uniqueness in local context
                    if s not in node_map:
                        node_map[s] = str(next_id)
                        G.add_node(str(next_id), label=s)
                        next_id += 1
                    if o not in node_map:
                        node_map[o] = str(next_id)
                        G.add_node(str(next_id), label=o)
                        next_id += 1
                    G.add_edge(node_map[s], node_map[o], label=p)
            else:
                # Old Format
                # Parse Entities
                for match in self.sg_entity_re.finditer(text):
                    node_id, label = match.groups()
                    G.add_node(node_id, label=label.strip())
                # Parse Relations
                for match in self.sg_rel_re.finditer(text):
                    src, rel, dst = match.groups()
                    G.add_edge(src, dst, label=rel.strip())
                
        elif mode == 'event_graph':
            # Try New Format: (E1:lbl, REL, E2:lbl)
            triples = self.tuple_re.findall(text)
            if triples and "event(" not in text:
                for s, p, o in triples:
                    s, p, o = s.strip(), p.strip(), o.strip()
                    
                    def parse_node(raw):
                        if ':' in raw:
                            parts = raw.split(':', 1)
                            return parts[0].strip(), parts[1].strip()
                        return raw, raw
                    
                    s_id, s_lbl = parse_node(s)
                    o_id, o_lbl = parse_node(o)
                    
                    if s_id not in G: G.add_node(s_id, label=s_lbl)
                    if o_id not in G: G.add_node(o_id, label=o_lbl)
                    G.add_edge(s_id, o_id, label=p)
            else:
                # Old Format
                # Parse Events
                for match in self.eg_event_re.finditer(text):
                    node_id, label_raw = match.groups()
                    # label might be "trigger|type", just keep it as is or split
                    G.add_node(node_id, label=label_raw.strip())
                # Parse Relations
                for match in self.eg_rel_re.finditer(text):
                    src, rel, dst = match.groups()
                    G.add_edge(src, dst, label=rel.strip())
        
        elif mode == 'graph_search':
            found_edges = False
            
            # 0. Node Range (to ensure isolated nodes are included)
            range_match = self.gs_node_range.search(text)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                for i in range(start, end + 1):
                    node_id = str(i)
                    if node_id not in G:
                        G.add_node(node_id, label=node_id)

            # 1. Flow (Capacity)
            matches = self.gs_flow.findall(text)
            for u, v, cap in matches:
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label=f"capacity {cap}", weight=int(cap))
                found_edges = True
                
            # 2. Weighted
            matches = self.gs_weight.findall(text)
            for u, v, w in matches:
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label=f"weight: {w}", weight=int(w))
                found_edges = True

            # 3. Matching (Applicant/Host)
            matches = self.gs_matching.findall(text)
            for subj_type, subj_id, obj_type, obj_id in matches:
                s_node = f"{subj_id}:{subj_type.capitalize()}"
                o_node = f"{obj_id}:{obj_type.capitalize()}"
                if s_node not in G: G.add_node(s_node, label=s_node)
                if o_node not in G: G.add_node(o_node, label=o_node)
                G.add_edge(s_node, o_node, label="interested_in")
                found_edges = True

            # 4. Undirected / Connectivity (Tuples)
            matches = self.gs_undirected_tuple.findall(text)
            for u, v in matches:
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label="connected")
                found_edges = True
                
            # 5. Connected to
            matches = self.gs_connected_to.findall(text)
            for u, v in matches:
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label="connected")
                found_edges = True

            # 6. Edge between
            matches = self.gs_edge_between.findall(text)
            for u, v in matches:
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label="connected")
                found_edges = True

            # 7. Fallback: Parse NL edges (original regex)
            matches = self.gs_edge_re.findall(text)
            for u, v, cap in matches:
                label = f"capacity {cap}" if cap else "edge"
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label=label)
                if cap: G[u][v]['weight'] = int(cap)
            
            # 8. Parse Weighted Triples (if input is already structured)
            matches_triple = self.gs_triple_re.findall(text)
            for u, w, v in matches_triple:
                u, v = u.strip(), v.strip()
                label = f"weight: {w}"
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label=label, weight=int(w))

            # 9. Parse Structured Capacity
            matches_cap = self.gs_struct_capacity.findall(text)
            for u, cap, v in matches_cap:
                u, v = u.strip(), v.strip()
                label = f"capacity {cap}"
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label=label, weight=int(cap))

            # 10. Parse Structured Generic (-,)
            matches_gen = self.gs_struct_generic.findall(text)
            for u, v in matches_gen:
                u, v = u.strip(), v.strip()
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label="connected")

            # 11. Parse Structured Label
            matches_lbl = self.gs_struct_label.findall(text)
            for u, l, v in matches_lbl:
                u, v = u.strip(), v.strip()
                if u not in G: G.add_node(u, label=u)
                if v not in G: G.add_node(v, label=v)
                G.add_edge(u, v, label=l)
        
        return G

    def graph_to_text(self, G: nx.DiGraph, mode: str) -> str:
        if mode == 'scene_graph':
            lines = []
            lines.append("[Graph Schema]")
            lines.append("Edges: Represent connections as (subject, predicate, object).")
            lines.append("")
            lines.append("[Graph Facts]")

            triples = []
            for u, v, data in G.edges(data=True):
                u_label = G.nodes[u].get('label', u)
                v_label = G.nodes[v].get('label', v)
                edge_label = data.get('label', 'related')
                triples.append(f"({u_label}, {edge_label}, {v_label})")
            lines.append(", ".join(triples))
            return "\n".join(lines)
                
        elif mode == 'event_graph':
            lines = []
            lines.append("[Graph Schema]")
            lines.append("Edges: Represent connections as (event1, relation, event2).")
            lines.append("")
            lines.append("[Graph Facts]")

            triples = []
            for u, v, data in G.edges(data=True):
                u_lbl = G.nodes[u].get('label', '')
                v_lbl = G.nodes[v].get('label', '')
                # Heuristic to reconstruct ID:Label
                u_str = f"{u}:{u_lbl}" if u_lbl and str(u) != str(u_lbl) else str(u)
                v_str = f"{v}:{v_lbl}" if v_lbl and str(v) != str(v_lbl) else str(v)
                edge_label = data.get('label', 'related')
                triples.append(f"({u_str}, {edge_label}, {v_str})")
            lines.append(", ".join(triples))
            return "\n".join(lines)
                
        elif mode == 'graph_search':
            lines = []
            lines.append("[Graph Facts]")
            for u, v, data in G.edges(data=True):
                lines.append(f"Edge from {u} to {v} with {data.get('label', 'edge')}.")
            return "\n".join(lines)
        
        return ""

    def graph_to_search_text(self, G: nx.DiGraph, with_intro: bool = True) -> str:
        lines = []
        if with_intro:
            lines.append("Here is a description of a directed graph.")
            lines.append("")
        lines.append("[Graph Schema]")
        
        has_weights = any('weight' in d for _, _, d in G.edges(data=True))
        has_capacity = any('capacity' in d.get('label', '') for _, _, d in G.edges(data=True))
        
        if has_capacity:
            lines.append("Edges: Represent connections as (source, capacity: <value>, target).")
        elif has_weights:
            lines.append("Edges: Represent connections as (source, weight: <value>, target).")
        else:
            lines.append("Edges: Represent connections as (source, label: <text>, target).")
            
        lines.append("")
        lines.append("[Graph Facts]")
        
        edge_list = []
        # Sort edges for deterministic output
        sorted_edges = sorted(G.edges(data=True), key=lambda x: (str(x[0]), str(x[1])))

        for u, v, d in sorted_edges:
            def fmt_node(n):
                lbl = G.nodes[n].get('label', str(n))
                lbl = str(lbl).replace('"', "'")
                if str(lbl) == str(n):
                    return str(n)
                return f"{n}: \"{lbl}\""

            u_str = fmt_node(u)
            v_str = fmt_node(v)

            if 'capacity' in d.get('label', ''):
                 val = d.get('weight', 0)
                 edge_list.append(f"({u_str}, capacity: {val}, {v_str})")
            elif 'weight' in d:
                edge_list.append(f"({u_str}, weight: {d['weight']}, {v_str})")
            else:
                l = d.get('label', 'related')
                # Use '-' placeholder for generic connections to match UniGrasp style
                if l in ["connected", "edge", "related", "interested_in"]:
                    edge_list.append(f"({u_str}, -, {v_str})")
                else:
                    l = str(l).replace('"', "'")
                    edge_list.append(f"({u_str}, label: \"{l}\", {v_str})")
        lines.append(f"Edges: {', '.join(edge_list)}")
        
        return "\n".join(lines)

class TaskGenerator:
    def __init__(self):
        self.parser = GraphParser()

    def split_user_content(self, user_content: str) -> Tuple[str, str]:
        """Splits user content into (graph_description, question)."""
        if "Q:" in user_content:
            parts = user_content.split("Q:", 1)
            return parts[0].strip(), "Q:" + parts[1]
        
        last_newline_idx = user_content.rfind('\n')
        if last_newline_idx != -1:
            potential_question = user_content[last_newline_idx+1:].strip()
            question_starters = ["Is there", "What is", "Give the", "Find"]
            if any(potential_question.startswith(q) for q in question_starters):
                return user_content[:last_newline_idx].strip(), potential_question
        
        return user_content.strip(), ""

    def generate_extraction_task(self, G: nx.DiGraph, original_text: str) -> Tuple[List[Dict], List[Dict]]:
        """Generates a task to extract graph facts from raw text."""
        if not G.edges:
            return [], []
            
        # Use the structured text as the answer, without conversational intro
        structured_text = self.parser.graph_to_search_text(G, with_intro=False)
        
        # Construct prompt using the raw description
        prompt = (
            f"[graph_search:extraction]\n"
            f"Please extract the graph facts from the following description and format them according to the standard schema.\n\n"
            f"{original_text}"
        )
        
        task = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": structured_text}
            ],
            "images": []
        }
        
        # Return same task for both simple and reasoning for now
        return [task], [task]

    def generate_traversal_task(self, G: nx.DiGraph, original_data: Dict, mode: str, max_tasks: int = 1) -> Tuple[List[Dict], List[Dict]]:
        """Generates QA pairs based on graph structure. Returns (simple_list, reasoning_list)."""
        simple_samples = []
        reasoning_samples = []
        
        nodes = list(G.nodes())
        if not nodes:
            return [], []
            
        # Helper to add both versions
        def add_task_versions(q_text, ans_simple, ans_reasoning):
            # Simple version
            simple_samples.append({
                "messages": [
                    {"role": "user", "content": q_text},
                    {"role": "assistant", "content": ans_simple}
                ],
                "images": []  # Graph search is a text-only structural reasoning task; image is not needed.
            })
            # Reasoning version
            q_reasoning = q_text + "\nProvide a step-by-step reasoning before your answer."
            reasoning_samples.append({
                "messages": [
                    {"role": "user", "content": q_reasoning},
                    {"role": "assistant", "content": ans_reasoning}
                ],
                "images": []  # Graph search is a text-only structural reasoning task; image is not needed.
            })

        # Define available task types and shuffle them to pick random ones
        # 1: Neighbor Query, 2: Direct Relation, 3: Path Existence, 4: 2-hop Path, 5: Connectivity, 6: Cycle
        task_types = [1, 2, 3, 4, 5, 6]
        random.shuffle(task_types)
        
        generated_count = 0

        for t_type in task_types:
            if generated_count >= max_tasks:
                break
            
            prev_len = len(simple_samples)

            # Determine tag and graph text
            if mode == 'scene_graph':
                prefix = "[scene_graph:graph_search]"
            elif mode == 'event_graph':
                prefix = "[event_graph:graph_search]"
            else:
                # Fallback for synthetic graph search or other modes
                tag_map = {
                    1: "neighbor_query",
                    2: "direct_relation",
                    3: "path_existence",
                    4: "two_hop_path",
                    5: "connectivity",
                    6: "cycle"
                }
                sub_tag = tag_map.get(t_type, "traversal")
                prefix = f"[graph_search:{sub_tag}]"

            graph_text = self.parser.graph_to_search_text(G)

            # 1. Neighbor Query
            if t_type == 1 and len(G.edges) > 0:
                src = random.choice(list(G.nodes()))
                neighbors = list(G.successors(src))
                src_label = G.nodes[src].get('label', src)
                
                if neighbors:
                    # Construct Answer
                    relations = []
                    reasoning_steps = [f"To find outgoing relations for node '{src_label}' (ID: {src}), I check the graph facts:"]
                    for n in neighbors:
                        edge_label = G[src][n].get('label', 'edge')
                        if 'weight' in G[src][n]: edge_label = f"weight: {G[src][n]['weight']}"
                        n_label = G.nodes[n].get('label', n)
                        # Simplified answer format
                        line = f"{edge_label} -> {n_label}"
                        relations.append(line)
                        reasoning_steps.append(f"- Found edge: ({src}) -[{edge_label}]-> ({n}) which is '{n_label}'.")
                    
                    ans_str = "\n".join(relations)
                    
                    reasoning_steps.append("Listing all identified outgoing relations:")
                    reasoning_steps.append(ans_str)
                    ans_reasoning = "\n".join(reasoning_steps)
                    
                    q_content = (
                        f"{prefix}\n{graph_text}\n\n"
                        f"Q: Given the generated graph facts, list all outgoing relations for the node '{src_label}' (ID: {src})."
                    )
                    
                    add_task_versions(q_content, ans_str, ans_reasoning)

            # 2. Direct Relation Existence (Positive & Negative)
            elif t_type == 2 and len(nodes) >= 2:
                # Randomly choose positive or negative check
                is_positive = random.choice([True, False])
                if len(G.edges) == 0: is_positive = False # Force negative if no edges
                
                # Positive: Existing edge
                if is_positive and len(G.edges) > 0:
                    u, v = random.choice(list(G.edges))
                    u_label = G.nodes[u].get('label', u)
                    v_label = G.nodes[v].get('label', v)
                    rel = G[u][v].get('label', 'edge')
                    if 'weight' in G[u][v]: rel = f"weight: {G[u][v]['weight']}"
                    
                    q_pos = (
                        f"{prefix}\n{graph_text}\n\n"
                        f"Q: Based on the graph facts, is there a direct relation '{rel}' from '{u_label}' (ID: {u}) to '{v_label}' (ID: {v})?"
                    )
                    
                    ans_simple = "Yes"
                    ans_reasoning = (
                        f"I will check the graph facts for a relation from node {u} ('{u_label}') to node {v} ('{v_label}').\n"
                        f"I found the fact connecting {u} and {v} with {rel}.\n"
                        f"Therefore, the relation exists.\nAnswer: Yes"
                    )
                    add_task_versions(q_pos, ans_simple, ans_reasoning)

                # Negative: Non-existing edge
                else:
                    non_edges = list(nx.non_edges(G))
                    if non_edges:
                        u, v = random.choice(non_edges)
                        u_label = G.nodes[u].get('label', u)
                        v_label = G.nodes[v].get('label', v)
                        
                        q_neg = (
                            f"{prefix}\n{graph_text}\n\n"
                            f"Q: Based on the graph facts, is there a direct relation from '{u_label}' (ID: {u}) to '{v_label}' (ID: {v})?"
                        )
                        
                        ans_simple = "No"
                        ans_reasoning = (
                            f"I will check the graph facts for a direct relation from node {u} ('{u_label}') to node {v} ('{v_label}').\n"
                            f"Scanning the relations, I do not find any edge starting at {u} and ending at {v}.\n"
                            f"Therefore, the relation does not exist.\nAnswer: No"
                        )
                        add_task_versions(q_neg, ans_simple, ans_reasoning)

            # 3. Path Existence (BFS)
            elif t_type == 3 and len(nodes) >= 2:
                u, v = random.sample(nodes, 2)
                try:
                    path = nx.shortest_path(G, u, v)
                    clean_path = f"({path[0]})"
                    
                    reasoning_steps = [f"I will search for a path from node {u} to node {v}."]
                    for i in range(len(path)-1):
                        p_u, p_v = path[i], path[i+1]
                        rel = G[p_u][p_v].get('label', 'edge')
                        if 'weight' in G[p_u][p_v]: rel = f"weight: {G[p_u][p_v]['weight']}"
                        clean_path += f" - {rel} -> ({p_v})"
                        reasoning_steps.append(f"- Step {i+1}: From {p_u}, take edge '{rel}' to {p_v}.")
                    
                    answer = clean_path
                    reasoning_steps.append(f"I have reached the target node {v}.")
                    reasoning_steps.append(f"Answer: {answer}")
                    ans_reasoning = "\n".join(reasoning_steps)
                    
                except nx.NetworkXNoPath:
                    answer = "No"
                    ans_reasoning = (
                        f"I will search for a path from node {u} to node {v}.\n"
                        f"I explored the graph starting from {u}, but I could not find any sequence of edges leading to {v}.\n"
                        f"Answer: No"
                    )

                q_content = (
                    f"{prefix}\n{graph_text}\n\n"
                    f"Q: Based on the graph facts, is there a directed path from node ID {u} to node ID {v}?"
                )
                
                add_task_versions(q_content, answer, ans_reasoning)

            # 4. 2-hop Path Query
            elif t_type == 4:
                two_hop_candidates = []
                # Limit search space for efficiency if graph is large, but scene graphs are usually small
                for n in nodes:
                    neighbors = list(G.successors(n))
                    for nb in neighbors:
                        nb_neighbors = list(G.successors(nb))
                        for nb2 in nb_neighbors:
                            two_hop_candidates.append((n, nb, nb2))
                
                if two_hop_candidates:
                    # Generate a positive sample
                    src, mid, dst = random.choice(two_hop_candidates)
                    rel1 = G[src][mid].get('label', 'edge')
                    if 'weight' in G[src][mid]: rel1 = f"weight: {G[src][mid]['weight']}"
                    rel2 = G[mid][dst].get('label', 'edge')
                    if 'weight' in G[mid][dst]: rel2 = f"weight: {G[mid][dst]['weight']}"
                    
                    q_content = (
                        f"{prefix}\n{graph_text}\n\n"
                        f"Q: Is there a 2-hop relationship (A->B->C) between node {src} and node {dst}? If yes, describe the path."
                    )
                    
                    # Simplified arrow format
                    ans_content = f"({src}) - {rel1} -> ({mid}) - {rel2} -> ({dst})"
                    
                    ans_reasoning = (
                        f"I am looking for a path of length 2 from node {src} to node {dst}.\n"
                        f"1. Checking neighbors of {src}: Found node {mid} via relation '{rel1}'.\n"
                        f"2. Checking neighbors of {mid}: Found node {dst} via relation '{rel2}'.\n"
                        f"A valid 2-hop path exists.\n"
                        f"Answer: {ans_content}"
                    )
                    
                    add_task_versions(q_content, ans_content, ans_reasoning)

            # 5. Connectivity (Weak Connectivity)
            elif t_type == 5 and len(nodes) >= 2:
                u, v = random.sample(nodes, 2)
                # Check weak connectivity (ignoring direction)
                undirected_G = G.to_undirected()
                try:
                    is_connected = nx.has_path(undirected_G, u, v)
                except:
                    is_connected = False
                
                q_content = (
                    f"{prefix}\n{graph_text}\n\n"
                    f"Q: Ignoring edge directions, is there a connection between node {u} and node {v}?"
                )
                ans_content = "Yes" if is_connected else "No"
                
                if is_connected:
                    ans_reasoning = (
                        f"I will check if node {u} and node {v} are connected in the underlying undirected graph.\n"
                        f"Tracing the connections from {u}, I can reach {v} (ignoring edge directions).\n"
                        f"Answer: Yes"
                    )
                else:
                    ans_reasoning = (
                        f"I will check if node {u} and node {v} are connected in the underlying undirected graph.\n"
                        f"Tracing all possible connections from {u}, I cannot reach {v}.\n"
                        f"Answer: No"
                    )
                
                add_task_versions(q_content, ans_content, ans_reasoning)

            # 6. Cycle Detection
            elif t_type == 6:
                try:
                    cycle = nx.find_cycle(G, orientation='original')
                    # cycle is list of edges [(u, v, key), ...] or [(u, v)]
                    has_cycle = True
                    # Format cycle path
                    path_nodes = [str(edge[0]) for edge in cycle]
                    path_nodes.append(str(cycle[0][0])) # Close the loop
                    cycle_path = " -> ".join(path_nodes)
                except nx.NetworkXNoCycle:
                    has_cycle = False
                    cycle_path = ""

                q_content = (
                    f"{prefix}\n{graph_text}\n\n"
                    f"Q: Is there a cycle in this graph?"
                )
                
                if has_cycle:
                    ans_simple = "Yes"
                    ans_reasoning = (
                        f"I will search for a cycle in the graph.\n"
                        f"Tracing the edges, I found a path that returns to the start: {cycle_path}.\n"
                        f"Answer: Yes"
                    )
                else:
                    ans_simple = "No"
                    ans_reasoning = (
                        f"I will search for a cycle in the graph.\n"
                        f"I explored the graph and found no path that returns to a visited node in the current traversal stack.\n"
                        f"Answer: No"
                    )
                add_task_versions(q_content, ans_simple, ans_reasoning)

            # Check if a task was successfully added
            if len(simple_samples) > prev_len:
                generated_count += 1

        return simple_samples, reasoning_samples

    def generate_consistency_task(self, G: nx.DiGraph, original_data: Dict, mode: str) -> Tuple[List[Dict], List[Dict]]:
        """Generates True/False consistency checks with perturbations. Returns (simple, reasoning)."""
        simple_samples = []
        reasoning_samples = []
        edges = list(G.edges(data=True))
        
        if mode == 'scene_graph':
            prefix = "[scene_graph:consistency]"
        elif mode == 'event_graph':
            prefix = "[event_graph:consistency]"
        else:
            prefix = f"[{mode}:consistency]"
        
        # Check if images exist to add <image> token
        has_image = len(original_data.get("images", [])) > 0
        img_token = "<image>\n" if has_image else ""
        
        def add_task_versions(q_text, ans_simple, ans_reasoning):
            # Simple version
            simple_samples.append({
                "messages": [
                    {"role": "user", "content": q_text},
                    {"role": "assistant", "content": ans_simple}
                ],
                "images": original_data.get("images", [])
            })
            # Reasoning version
            q_reasoning = q_text + "\nProvide a step-by-step reasoning before your answer."
            reasoning_samples.append({
                "messages": [
                    {"role": "user", "content": q_reasoning},
                    {"role": "assistant", "content": ans_reasoning}
                ],
                "images": original_data.get("images", [])
            })

        # Randomly choose between Positive (Original) or Negative (Perturbation)
        use_positive = random.choice([True, False])
        if not edges:
            use_positive = True

        if use_positive:
            # 1. Positive Sample (Original Graph)
            original_text = self.parser.graph_to_text(G, mode)
            prompt_pos = (
                f"{prefix}\n{img_token}{original_text}\n\n"
                f"Verify consistency. Based on the image and the graph facts above, are these facts completely accurate regarding the ground truth? Answer Yes or No. If No, identify the incorrect graph fact."
            )
            
            ans_pos_simple = "Yes"
            ans_pos_reasoning = (
                "I will verify the consistency of the graph facts against the ground truth (image/context).\n"
                "1. Checking entities: All entities appear to be correctly identified.\n"
                "2. Checking relations: All relations appear to be accurate and supported by the context.\n"
                "No inconsistencies found.\n"
                "Answer: Yes"
            )
            
            add_task_versions(prompt_pos, ans_pos_simple, ans_pos_reasoning)
        
        else:
            # 2. Negative Sample (Perturbation)
            # Perform one mutation to be precise about the error
            G_mutated = G.copy()
            mutation_desc = ""
            
            current_edges = list(G_mutated.edges(data=True))
            if current_edges:
                u, v, data = random.choice(current_edges)
                
                valid_mutations = ['mutate_label']
                # Avoid dropping the only edge to prevent empty graph text
                if len(current_edges) > 1:
                    valid_mutations.append('drop')
                # Avoid swapping self-loops
                if u != v:
                    valid_mutations.append('swap')
                
                mutation_type = random.choice(valid_mutations)
                
                u_label = G.nodes[u].get('label', u)
                v_label = G.nodes[v].get('label', v)
                
                if mutation_type == 'drop':
                    G_mutated.remove_edge(u, v)
                    mutation_desc = f"The relation between '{u_label}' and '{v_label}' is missing."
                elif mutation_type == 'swap':
                    G_mutated.remove_edge(u, v)
                    G_mutated.add_edge(v, u, **data)
                    mutation_desc = f"The relation between '{u_label}' and '{v_label}' is reversed."
                elif mutation_type == 'mutate_label':
                    fake_label = "UNRELATED_TO" if data.get('label') != "UNRELATED_TO" else "RELATED_TO"
                    G_mutated[u][v]['label'] = fake_label
                    mutation_desc = f"The relation between '{u_label}' and '{v_label}' has an incorrect label '{fake_label}'."

                mutated_text = self.parser.graph_to_text(G_mutated, mode)
                
                # Construct the prompt
                prompt_neg = (
                    f"{prefix}\n{img_token}{mutated_text}\n\n"
                    f"Verify consistency. Based on the image and the graph facts above, are these facts completely accurate regarding the ground truth? Answer Yes or No. If No, identify the incorrect graph fact."
                )
                
                # CHANGED: Enforce strict Yes/No for simple mode per requirements
                ans_neg_simple = "No"
                
                ans_neg_reasoning = (
                    f"I will verify the consistency of the graph facts against the ground truth.\n"
                    f"I detected an error: {mutation_desc}\n"
                    f"Therefore, the graph facts are not accurate.\n"
                    f"Answer: No"
                )
                
                add_task_versions(prompt_neg, ans_neg_simple, ans_neg_reasoning)

        return simple_samples, reasoning_samples

def process_dataset(input_data: List[Dict]):
    generator = TaskGenerator()
    
    # Buckets for separate task types
    buckets = {
        'consistency': {'simple': [], 'reasoning': []},
        'extraction': {'simple': [], 'reasoning': []},
        'algorithm': {'simple': [], 'reasoning': []}
    }
    
    total = len(input_data)
    print(f"Starting processing of {total} samples...")
    
    skipped_empty = 0

    for idx, entry in enumerate(input_data):
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{total} samples...")

        # Detect Type
        user_msg = entry['messages'][0]['content']
        assistant_msg = entry['messages'][1]['content']
        
        mode = None
        target_text = assistant_msg # Default source for graph
        is_raw_nl = False

        if "[scene_graph" in user_msg:
            mode = 'scene_graph'
        elif "[event_graph" in user_msg:
            mode = 'event_graph'
        elif "In a directed graph" in user_msg or "In an undirected graph" in user_msg or "The nodes are numbered" in user_msg:
            mode = 'graph_search'
            target_text = user_msg
            is_raw_nl = True
        elif "[graph_search" in user_msg:
            mode = 'graph_search'
            target_text = user_msg
        
        if not mode:
            continue

        # Removed sub-sampling for scene_graph to use all tasks
        # if mode == 'scene_graph' and random.random() > 0.33:
        #    continue

        # Parse Ground Truth Graph
        G = generator.parser.parse(target_text, mode)
        
        if G.number_of_nodes() == 0:
            skipped_empty += 1
            continue
        
        # 1. Traversal / QA (Both SG and EG) -> categorize as 'algorithm'
        # SG: 1 traversal task, EG: 1 traversal task
        # For graph_search, we only want extraction tasks, so skip traversal
        if mode != 'graph_search':
            n_traversal = 1
            s_trav, r_trav = generator.generate_traversal_task(G, entry, mode, max_tasks=n_traversal)
            buckets['algorithm']['simple'].extend(s_trav)
            buckets['algorithm']['reasoning'].extend(r_trav)
        
        # 2. Consistency / Perturbation (Only SG) -> categorize as 'consistency'
        # SG: 1 consistency task
        if mode == 'scene_graph':
            s_cons, r_cons = generator.generate_consistency_task(G, entry, mode)
            buckets['consistency']['simple'].extend(s_cons)
            buckets['consistency']['reasoning'].extend(r_cons)
            
        # 3. Extraction Task -> categorize as 'extraction'
        # Generate extraction task for all graph_search inputs
        # (unless it is already an extraction task to avoid duplication)
        if mode == 'graph_search' and "[graph_search:extraction]" not in user_msg:
            # Extract description part from user message (remove question)
            graph_desc, _ = generator.split_user_content(user_msg)
            
            # Remove existing task tag if present to avoid double tagging
            graph_desc = re.sub(r'^\[graph_search:[^\]]+\]\s*', '', graph_desc)
            
            s_ext, r_ext = generator.generate_extraction_task(G, graph_desc)
            buckets['extraction']['simple'].extend(s_ext)
            buckets['extraction']['reasoning'].extend(r_ext)

    print(f"Finished processing.")
    print(f"Consistency: Simple {len(buckets['consistency']['simple'])}, Reasoning {len(buckets['consistency']['reasoning'])}")
    print(f"Extraction:  Simple {len(buckets['extraction']['simple'])}, Reasoning {len(buckets['extraction']['reasoning'])}")
    print(f"Algorithm:   Simple {len(buckets['algorithm']['simple'])}, Reasoning {len(buckets['algorithm']['reasoning'])}")
    print(f"Skipped {skipped_empty} samples due to empty graph parsing.")
    return buckets

def test_parser():
    print("Running parser tests...")
    generator = TaskGenerator()
    parser = generator.parser
    
    # Test Scene Graph
    print("\n=== Testing Scene Graph ===")
    sg_text = "(light, on, board), (clock, on, board)"
    G_sg = parser.parse(sg_text, "scene_graph")
    print(f"SG Nodes: {G_sg.nodes(data=True)}")
    print(f"SG Edges: {G_sg.edges(data=True)}")
    assert len(G_sg.nodes) == 3, "SG should have 3 nodes (light, board, clock)"
    assert len(G_sg.edges) == 2, "SG should have 2 edges"
    
    # Generate SG Tasks
    sg_mock_data = {"messages": [{"role": "user", "content": "Generate SG"}, {"role": "assistant", "content": sg_text}], "images": ["img.jpg"]}
    s_tasks, r_tasks = generator.generate_traversal_task(G_sg, sg_mock_data, "scene_graph")
    print("--- Generated SG Traversal Simple Task Sample ---")
    if s_tasks: print(json.dumps(s_tasks[0], indent=2))
    print("--- Generated SG Traversal Reasoning Task Sample ---")
    if r_tasks: print(json.dumps(r_tasks[0], indent=2))
    
    # Generate SG Consistency Task
    s_cons, r_cons = generator.generate_consistency_task(G_sg, sg_mock_data, "scene_graph")
    print("--- Generated SG Consistency Simple Task Sample ---")
    if s_cons: print(json.dumps(s_cons[0], indent=2))
    print("--- Generated SG Consistency Reasoning Task Sample ---")
    if r_cons: print(json.dumps(r_cons[0], indent=2))
    
    # Test Event Graph
    print("\n=== Testing Event Graph ===")
    eg_text = "(E1:shortened, BEFORE, E11:abandoned), (E5:moved, PRECONDITION, E10:held)"
    G_eg = parser.parse(eg_text, "event_graph")
    print(f"EG Nodes: {G_eg.nodes(data=True)}")
    print(f"EG Edges: {G_eg.edges(data=True)}")
    assert "E1" in G_eg.nodes, "E1 should be a node"
    assert G_eg.nodes["E1"]["label"] == "shortened", "E1 label mismatch"
    assert G_eg["E1"]["E11"]["label"] == "BEFORE", "Edge label mismatch"
    
    # Generate EG Tasks
    eg_mock_data = {"messages": [{"role": "user", "content": "Generate EG"}, {"role": "assistant", "content": eg_text}], "images": []}
    s_tasks, r_tasks = generator.generate_traversal_task(G_eg, eg_mock_data, "event_graph")
    print("--- Generated EG Traversal Simple Task Sample ---")
    if s_tasks: print(json.dumps(s_tasks[0], indent=2))
    print("--- Generated EG Traversal Reasoning Task Sample ---")
    if r_tasks: print(json.dumps(r_tasks[0], indent=2))
    
    # Test Graph Search (Extraction)
    print("\n=== Testing Graph Search Extraction ===")
    gs_raw_text = "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. The nodes are numbered from 0 to 8, and the edges are: (0,2) (0,4) (1,2) (1,8) (1,4) (1,3) (3,8) (3,4) (4,8) (5,7) (5,6) (6,7)\nQ: Is there a path between node 8 and node 1?"
    G_gs = parser.parse(gs_raw_text, "graph_search")
    print(f"GS Nodes: {len(G_gs.nodes)}")
    print(f"GS Edges: {len(G_gs.edges)}")
    assert len(G_gs.edges) > 0, "Should parse edges from raw text"
    
    # Generate Extraction Task
    desc, _ = generator.split_user_content(gs_raw_text)
    s_ext, r_ext = generator.generate_extraction_task(G_gs, desc)
    print("--- Generated GS Extraction Task Sample ---")
    if s_ext: print(json.dumps(s_ext[0], indent=2))
    
    print("\nAll parser tests passed!")

def load_data(path: str) -> List[Dict]:
    data = []
    try:
        with open(path, 'r') as f:
            if path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            elif path.endswith('.json'):
                data = json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sg_path", type=str, help="Path to scene graph json/jsonl")
    parser.add_argument("--eg_path", type=str, help="Path to event graph json/jsonl")
    parser.add_argument("--gs_path", type=str, nargs='+', help="Paths to graph search json/jsonl files")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--test", action="store_true", help="Run parser tests")
    args = parser.parse_args()

    if args.test:
        test_parser()
        exit(0)

    if (args.sg_path or args.eg_path or args.gs_path) and args.output_dir:
        combined_data = []
        
        if args.sg_path:
            print(f"Loading Scene Graph data from {args.sg_path}...")
            sg_data = load_data(args.sg_path)
            print(f"Loaded {len(sg_data)} scene graph samples.")
            combined_data.extend(sg_data)
        
        if args.eg_path:
            print(f"Loading Event Graph data from {args.eg_path}...")
            eg_data = load_data(args.eg_path)
            print(f"Loaded {len(eg_data)} event graph samples.")
            combined_data.extend(eg_data)

        if args.gs_path:
            # Handle potential '+' separator in paths
            expanded_paths = []
            for p in args.gs_path:
                expanded_paths.extend(p.split('+'))

            for path in expanded_paths:
                print(f"Loading Graph Search data from {path}...")
                gs_data = load_data(path)
                print(f"Loaded {len(gs_data)} graph search samples.")
                combined_data.extend(gs_data)
        
        random.shuffle(combined_data)
        print(f"Processing {len(combined_data)} combined samples...")
        
        buckets = process_dataset(combined_data)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Define categories to write: consistency, extraction, algorithm
        categories = ['consistency', 'extraction', 'algorithm']
        
        # Create 'all' bucket
        buckets['all'] = {'simple': [], 'reasoning': []}
        for cat in categories:
            buckets['all']['simple'].extend(buckets[cat]['simple'])
            buckets['all']['reasoning'].extend(buckets[cat]['reasoning'])
        
        # Shuffle 'all' lists
        random.shuffle(buckets['all']['simple'])
        random.shuffle(buckets['all']['reasoning'])
        
        categories.append('all')

        for cat in categories:
            simple_path = os.path.join(args.output_dir, f"derived_tasks_simple_v2_{cat}.json")
            reasoning_path = os.path.join(args.output_dir, f"derived_tasks_reasoning_v2_{cat}.json")
            
            with open(simple_path, 'w') as f:
                json.dump(buckets[cat]['simple'], f, indent=2)
            
            with open(reasoning_path, 'w') as f:
                json.dump(buckets[cat]['reasoning'], f, indent=2)
                    
            print(f"[{cat.upper()}] Saved {len(buckets[cat]['simple'])} simple tasks to {simple_path}")
            print(f"[{cat.upper()}] Saved {len(buckets[cat]['reasoning'])} reasoning tasks to {reasoning_path}")
    else:
        print("Please provide at least one input path (--sg_path, --eg_path, --gs_path) and --output_dir.")
        # Mock data for demonstration if needed, but argparse is preferred now.
        # ...existing mock data code if you want to keep it as fallback...
