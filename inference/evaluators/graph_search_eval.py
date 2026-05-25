# graph_search_eval.py
import json
import string

def eval_graph_search(item):
    """
    Robust evaluation for graph search.
    
    item example:
    {
        "task": "graph_search",
        "subtask": "",
        "prompt": "...",
        "predict": "No.",
        "label": "No.\n"
    }
    """
    pred = item.get("predict", "")
    gt = item.get("label", "")
    
    # Safety: if predict still contains ### (e.g. if extractor wasn't called), extract it here too
    if isinstance(pred, str) and "###" in pred:
        pred = pred.split("###", 1)[1]

    def normalize(text):
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        # 1. Lowercase and strip whitespace
        text = text.strip().lower()
        # 2. Strip standard punctuation from ends (.,?)
        # This handles "Yes." vs "Yes"
        text = text.strip(string.punctuation)
        return text

    pred_norm = normalize(pred)
    gt_norm = normalize(gt)

    if pred_norm == gt_norm:
        return 1.0

    return 0.0

if __name__ == "__main__":
    item = {"task": "graph_search", "subtask": "", "prompt": "user\nIn an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.\nThe nodes are numbered from 0 to 33, and the edges are: (0,6) (0,5) (1,21) (2,26) (2,15) (2,14) (2,20) (2,28) (2,12) (3,29) (3,19) (3,22) (3,21) (4,33) (4,32) (4,9) (4,20) (4,28) (4,7) (4,27) (5,23) (5,17) (5,6) (6,25) (7,32) (7,15) (7,9) (7,11) (7,18) (8,19) (8,22) (8,16) (9,28) (9,27) (10,30) (10,33) (10,32) (10,20) (11,13) (11,12) (12,30) (12,13) (12,33) (12,32) (12,15) (12,28) (12,31) (12,27) (12,18) (13,33) (13,32) (13,20) (14,26) (14,20) (15,26) (15,27) (15,18) (16,29) (16,22) (17,23) (17,24) (18,20) (18,28) (18,31) (19,22) (19,21) (20,33) (20,31) (20,27) (23,24) (26,33) (26,31) (26,27) (27,33) (28,33) (28,31) (30,33) (32,33)\nIs there a path between node 6 and node 15 in this undirected graph?\nassistant\n", "predict": "No.", "label": "No.\n"}
    print(f"Accuracy: {eval_graph_search(item):.4f}")
