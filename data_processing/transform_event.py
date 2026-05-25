import json
import os
import re
from typing import List, Tuple

############################################
# 1. XML Block Extraction
############################################

def extract_event_graph_block(text: str) -> str | None:
    if "<event_graph>" in text and "</event_graph>" in text:
        start = text.index("<event_graph>")
        end = text.index("</event_graph>") + len("</event_graph>")
        return text[start:end]
    return None

############################################
# 2. Parse Nodes and Edges
############################################

NODE_PATTERN = re.compile(r"\((E\d+),\s*\"([^\"]+)\"\)")
EDGE_PATTERN = re.compile(r"\((E\d+),\s*([a-zA-Z\-]+),\s*(E\d+)\)")

def parse_nodes(event_graph: str) -> List[Tuple[str, str]]:
    return NODE_PATTERN.findall(event_graph)

def parse_edges(event_graph: str) -> List[Tuple[str, str, str]]:
    return EDGE_PATTERN.findall(event_graph)

############################################
# 3. Convert to Graph Facts
############################################

def build_graph_facts(nodes, edges) -> str:
    # Map eid to trigger
    node_map = {}
    for eid, label in nodes:
        # Extract trigger word from "trigger|type"
        if '|' in label:
            trigger = label.split('|', 1)[0]
        else:
            trigger = label
        
        # Sanitize: replace spaces with underscores
        trigger = trigger.strip().replace(" ", "_")
        node_map[eid] = trigger

    # Build Edges string with embedded node attributes
    # Format: (id1:trigger1, relation, id2:trigger2), ...
    edge_list = []
    for src, rel, dst in edges:
        src_trigger = node_map.get(src, "Unknown")
        dst_trigger = node_map.get(dst, "Unknown")
        edge_list.append(f"({src}:{src_trigger}, {rel}, {dst}:{dst_trigger})")
    
    return ", ".join(edge_list)

############################################
# 4. Build Graph Module
############################################

def build_graph_module(graph_facts: str) -> str:
    return graph_facts.strip()

############################################
# 5. Process One train_sft.json
############################################

def process_one_file(json_path: str):
    if not os.path.exists(json_path):
        print(f"[Skip] {json_path} not found")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    processed_cnt = 0
    total_cnt = len(data)

    for item in data:
        messages = item.get("messages", [])

        assistant_msg = None
        user_msg = None

        for msg in messages:
            if msg["role"] == "assistant":
                assistant_msg = msg
            elif msg["role"] == "user":
                user_msg = msg

        if assistant_msg is None or user_msg is None:
            continue

        # 1. Check completeness via extraction
        # extract_event_graph_block returns None if <event_graph> or </event_graph> is missing
        event_graph = extract_event_graph_block(assistant_msg["content"])
        if event_graph is None:
            continue

        nodes = parse_nodes(event_graph)
        edges = parse_edges(event_graph)

        if not nodes:
            # If we found tags but no nodes, we skip transformation (or handle as error)
            print("Warning: Found event_graph tags but no nodes parsed.")
            continue

        # 2. Transform to Graph Facts
        graph_facts = build_graph_facts(nodes, edges)
        
        # Replace Assistant content with Graph Facts
        assistant_msg["content"] = graph_facts
        processed_cnt += 1

    # 3. Write to new file with _gskel suffix
    base_name, ext = os.path.splitext(json_path)
    out_path = f"{base_name}_gskel{ext}"

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Processed {processed_cnt}/{total_cnt} items. Written: {out_path}")

############################################
# 6. Main
############################################

def main(paths: List[str]):
    for path in paths:
        if os.path.isfile(path):
            process_one_file(path)
        else:
            print(f"[Skip] {path} not found or is not a file")

def test_sample():
    sample_json = [
        { 
            "messages": [ 
                { 
                    "role": "user", 
                    "content": "\n[event_graph:matres]You are an event graph generation model.\n\nGiven the following document and pre-annotated events, identify all\nevent–event **temporal relations** that are explicitly supported by the text,\nand represent them using Event Graph Facts.\n\nDocument (docid = CNN_20130321_821):\nBarack Obama would make a great stand-up comic , not because he 's the funniest president ever but because he uses jokes the same way many of us comedians do : as a weapon . Barack Obama would make a great stand-up comic , not because he 's the funniest president ever but because he uses jokes the same way many of us comedians do : as a weapon . Traditionally , the ( intentionally ) funny lines by our presidents have had one thing in common : They were self-deprecating . Traditionally , the ( intentionally ) funny lines by our presidents have had one thing in common : They were self-deprecating . Sure , some presidents have used jokes to take jabs at their opponents , but not to the extent of Obama . Sure , some presidents have used jokes to take jabs at their opponents , but not to the extent of Obama . During his tenure , he has increasingly unleashed biting comedic barbs against his critics and political adversaries . During his tenure , he has increasingly unleashed biting comedic barbs against his critics and political adversaries . These jokes are intended to do more than simply entertain you . Obama 's humor is often delivered the way a comedian dealing with a heckler would do it . He tries to undermine his opponents with it and get the crowd -- in this case the public -- on his side . He tries to undermine his opponents with it and get the crowd -- in this case the public -- on his side . I can assure you that having a crowd laugh at your critic / heckler is not only effective in dominating them , it 's also very satisfying .\n\nEvents:\nE1: trigger=\"had\", type=Event, eiid=4\nE2: trigger=\"used\", type=Event, eiid=5\nE3: trigger=\"unleashed\", type=Event, eiid=8\nE4: trigger=\"assure\", type=Event, eiid=20\nE5: trigger=\"intended\", type=Event, eiid=1000\nE6: trigger=\"tries\", type=Event, eiid=1002\n\n============================================================\n### MATRES TEMPORAL REASONING RULES (STRICT)\n============================================================\n\n1. BEFORE / AFTER Direction\n- If the text states “Instead of X, Y happened”, then X happens AFTER Y.\n- For hypotheticals, alternatives, unrealized possibilities, comparisons:\n the realized event occurs BEFORE the unrealized one.\n\n2. Future Mental Events\n- Events containing “expect”, “foresee”, “predict”, “will”, “going to”\n usually refer to the future.\n\n3. VAGUE as Default\n- If no explicit temporal signal exists between two events,\n label VAGUE.\n\n4. No Hallucination\n- Only output relations supported by the text.\n\n5. Allowed Relation Labels ONLY:\n BEFORE, AFTER, EQUAL, VAGUE\n\n============================================================\n### Event Graph Fact Format\n============================================================\n\n1. Event declaration:\n event(event_id, \"trigger|event_type\")\n\n2. Event–event temporal relation:\n relation(src_event_id, relation_type, dst_event_id)\n\n============================================================\n### IMPORTANT OUTPUT RULES\n============================================================\n\n- event_id must be one of the given event ids (E1, E2, ...).\n- Preserve all event ids and relation labels exactly.\n- Do NOT invent new events or new relations.\n- Do NOT explain your reasoning.\n- Do NOT output <event_graph>, <Node>, <Edge>, or any XML format.\n\n============================================================\n### Output\n============================================================\n\nOnly output Event Graph Facts, one per line.\n" 
                }, 
                { 
                    "role": "assistant", 
                    "content": "<event_graph>\n<Node>\n(E1, \"had|Event\")\n(E2, \"used|Event\")\n(E3, \"unleashed|Event\")\n(E4, \"assure|Event\")\n(E5, \"intended|Event\")\n(E6, \"tries|Event\")\n</Node>\n<Edge>\n(E1, VAGUE, E2)\n(E2, BEFORE, E3)\n(E3, BEFORE, E5)\n(E6, BEFORE, E4)\n</Edge>\n</event_graph>" 
                } 
            ], 
            "images": [] 
        },
        {
            "messages": [
              {
                "role": "user",
                "content": "[event_graph:hievents]You are an event graph generation model.\n\nGiven the following document and pre-annotated events, identify all event–event relations that are explicitly supported by the text, and represent them using Event Graph Facts.\n\n============================================================\nEvent Graph Fact Format\n============================================================\n\n1. Event declaration:\n   event(event_id, \"trigger|event_type\")\n\n2. Event–event relation:\n   relation(src_event_id, relation_type, dst_event_id)\n\n============================================================\nAllowed Relation Types (HiEve)\n============================================================\n\n- SuperSub\n- SubSuper\n- Coref\n\n============================================================\nIMPORTANT RULES (STRICT)\n============================================================\n\n1. No Hallucination\n- Only output relations that are clearly and explicitly supported by the document.\n- Do NOT infer implicit temporal, causal, or hierarchical relations.\n\n2. Coreference\n- Use Coref ONLY when two events refer to the same real-world event.\n\n3. Hierarchical Relations\n- Use SuperSub or SubSuper ONLY when the text explicitly expresses a hierarchical relation.\n\n4. Event Identity\n- event_id must be one of the event ids Ex listed below (E1, E2, ...).\n- Do NOT invent new events.\n\n5. Output Constraints\n- Preserve event ids and relation labels exactly.\n- Do NOT explain your reasoning.\n- Do NOT output any format other than Event Graph Facts.\n\n============================================================\nInput\n============================================================\n\nDocument:\nA US drone attack over the weekend, which killed al-Qaida operative Fahd al-Quso, is thought to have been a direct result of information from the agent, according to the Guardian. MI6 and MI5 agents are banned from missions leading to assassinations. Associated Press was made aware of the underwear plot last week but did not publish details until Monday at the request of the White House over operational security. The Foreign Office declined to confirm or deny the involvement of British intelligence.\n\nEvents:\nE1: trigger=\"attack\", type=Occurrence\nE2: trigger=\"killed\", type=Occurrence\nE3: trigger=\"banned\", type=Occurrence\nE4: trigger=\"made\", type=StateChange\nE5: trigger=\"plot\", type=Occurrence\nE6: trigger=\"declined\", type=I_Action\n\n============================================================\nOutput\n============================================================\n\nOnly output Event Graph Facts, one per line.\n"
              },
              {
                "role": "assistant",
                "content": "<event_graph>\n<Node>\n(E1, \"attack|Occurrence\")\n(E2, \"killed|Occurrence\")\n(E3, \"banned|Occurrence\")\n(E4, \"made|StateChange\")\n(E5, \"plot|Occurrence\")\n(E6, \"declined|I_Action\")\n</Node>\n<Edge>\n(E1, SuperSub, E2)\n</Edge>\n</event_graph>"
              }
            ],
            "images": []
        },
        {
            "messages": [
              {
                "role": "user",
                "content": """
[event_graph:maven_ere]You are an event graph generation model.

Given the following document and pre-annotated events, identify **all event–event relations**
that are explicitly supported by the text, and represent them using Event Graph Facts.

Document:
[1] The Cherry Valley massacre was an attack by British and Iroquois forces on a fort and the village of Cherry Valley in eastern New York on November 11, 1778, during the American Revolutionary War.
[2] It has been described as one of the most horrific frontier massacres of the war.
[3] A mixed force of Loyalists, British soldiers, Seneca and Mohawks descended on Cherry Valley, whose defenders, despite warnings, were unprepared for the attack.
[4] During the raid, the Seneca in particular targeted non-combatants, and reports state that 30 such individuals were slain, in addition to a number of armed defenders.
[5] The raiders were under the overall command of Walter Butler, who exercised little authority over the Indians on the expedition.
[6] Historian Barbara Graymont describes Butler's command of the expedition as "criminally incompetent".
[7] The Seneca were angered by accusations that they had committed atrocities at the Battle of Wyoming, and the colonists' recent destruction of their forward bases of operation at Unadilla, Onaquaga, and Tioga.
[8] Butler's authority with the Indians was undermined by his poor treatment of Joseph Brant, the leader of the Mohawks.
[9] Butler repeatedly maintained, against accusations that he permitted the atrocities to take place, that he was powerless to restrain the Seneca.
[10] During the campaigns of 1778, Brant achieved an undeserved reputation for brutality.
[11] He was not present at Wyoming — although many thought he was — and he actively sought to minimize the atrocities that took place at Cherry Valley.
[12] Diaries belonging to British soldiers during the campaign state the regiment as being the "butchers" and given that Butler was the overall commander of the expedition, there is controversy as to who actually ordered or failed to restrain the killings.
[13] The massacre contributed to calls for reprisals, leading to the 1779 Sullivan Expedition which drove the Iroquois out of western New York.

Events:
E1: trigger="restrain", type=Hindering, sentence=9
E2: trigger="minimize", type=Cause_change_of_position_on_a_scale, sentence=11
E3: trigger="described", type=Statement, sentence=2
E4: trigger="achieved", type=Achieve, sentence=10
E5: trigger="contributed", type=Giving, sentence=13
E6: trigger="committed", type=Commitment, sentence=7
E7: trigger="given", type=Giving, sentence=12
E8: trigger="targeted", type=Aiming, sentence=4
E9: trigger="ordered", type=Arranging, sentence=12
E10: trigger="descended on", type=Motion_directional, sentence=3
E11: trigger="drove", type=Motion, sentence=13
E12: trigger="took place", type=Process_start, sentence=11
E13: trigger="permitted", type=Preventing_or_letting, sentence=9
E14: trigger="restrain", type=Hindering, sentence=12
E15: trigger="exercised", type=Practice, sentence=5

============================================================
### IMPORTANT GUIDELINES FOR ERE EVENT-RELATION EXTRACTION
============================================================

Temporal Ordering:
1. If Event A appears earlier in the text than Event B,
   A usually happens BEFORE B unless explicitly contradicted.

Restrictions on CONTAINS and SUBEVENT:
2. Only output CONTAINS or SUBEVENT when explicit hierarchical wording exists.

Restrictions on PRECONDITION:
3. Only use PRECONDITION when explicitly stated.

Directionality & Consistency:
4. BEFORE and AFTER relations must be globally consistent.
5. Do NOT hallucinate relations.

Allowed Relation Types ONLY:
BEFORE, AFTER, OVERLAP, CONTAINS, SIMULTANEOUS,
ENDS-ON, BEGINS-ON, CAUSE, PRECONDITION, SUBEVENT.

============================================================
### Event Graph Fact Format
============================================================

1. Event declaration:
   event(event_id, "trigger|event_type")

2. Event–event relation:
   relation(src_event_id, relation_type, dst_event_id)

============================================================
### IMPORTANT OUTPUT RULES
============================================================

- event_id must be one of the given event ids (E1, E2, ...).
- Preserve all event ids and relation labels exactly.
- Do NOT invent new events or new relations.
- Do NOT explain your reasoning.
- Do NOT output <event_graph>, <Node>, <Edge>, or any XML format.

============================================================
### Output
============================================================

Only output Event Graph Facts, one per line.
"""
              },
              {
                "role": "assistant",
                "content": "<event_graph>\n<Node>\n(E1, \"restrain|Hindering\")\n(E2, \"minimize|Cause_change_of_position_on_a_scale\")\n(E3, \"described|Statement\")\n(E4, \"achieved|Achieve\")\n(E5, \"contributed|Giving\")\n(E6, \"committed|Commitment\")\n(E7, \"given|Giving\")\n(E8, \"targeted|Aiming\")\n(E9, \"ordered|Arranging\")\n(E10, \"descended on|Motion_directional\")\n(E11, \"drove|Motion\")\n(E12, \"took place|Process_start\")\n(E13, \"permitted|Preventing_or_letting\")\n(E14, \"restrain|Hindering\")\n(E15, \"exercised|Practice\")\n</Node>\n<Edge>\n(E1, BEFORE, E11)\n(E10, BEFORE, E1)\n(E10, BEFORE, E11)\n(E10, BEFORE, E2)\n(E12, BEFORE, E1)\n(E12, BEFORE, E10)\n(E12, BEFORE, E11)\n(E12, BEFORE, E2)\n(E12, BEFORE, E3)\n(E12, BEFORE, E5)\n(E12, BEFORE, E7)\n(E13, BEFORE, E1)\n(E13, BEFORE, E10)\n(E13, BEFORE, E11)\n(E13, BEFORE, E12)\n(E13, BEFORE, E2)\n(E13, BEFORE, E3)\n(E13, BEFORE, E5)\n(E13, BEFORE, E7)\n(E14, BEFORE, E1)\n(E14, BEFORE, E10)\n(E14, BEFORE, E11)\n(E14, BEFORE, E12)\n(E14, BEFORE, E2)\n(E14, BEFORE, E3)\n(E14, BEFORE, E5)\n(E14, BEFORE, E7)\n(E2, BEFORE, E11)\n(E6, BEFORE, E1)\n(E6, BEFORE, E10)\n(E6, BEFORE, E11)\n(E6, BEFORE, E12)\n(E6, BEFORE, E13)\n(E6, BEFORE, E2)\n(E6, BEFORE, E3)\n(E6, BEFORE, E4)\n(E6, BEFORE, E5)\n(E6, BEFORE, E7)\n(E9, BEFORE, E1)\n(E9, BEFORE, E10)\n(E9, BEFORE, E11)\n(E9, BEFORE, E12)\n(E9, BEFORE, E14)\n(E9, BEFORE, E2)\n(E9, BEFORE, E3)\n(E9, BEFORE, E5)\n(E9, BEFORE, E7)\n(E15, CONTAINS, E10)\n(E12, PRECONDITION, E2)\n(E1, SIMULTANEOUS, E2)\n</Edge>\n</event_graph>"
              }
            ],
            "images": []
        }
    ]
    
    print("Processing test sample...")
    
    for i, item in enumerate(sample_json):
        messages = item.get("messages", [])
        assistant_msg = None
        user_msg = None

        for msg in messages:
            if msg["role"] == "assistant":
                assistant_msg = msg
            elif msg["role"] == "user":
                user_msg = msg

        if assistant_msg is None or user_msg is None:
            continue

        event_graph = extract_event_graph_block(assistant_msg["content"])
        if event_graph is None:
            continue

        nodes = parse_nodes(event_graph)
        edges = parse_edges(event_graph)

        if not nodes:
            print(f"Error: No nodes parsed in sample {i}")
            continue

        graph_facts = build_graph_facts(nodes, edges)
        
        # Replace Assistant content with Graph Facts
        assistant_msg["content"] = graph_facts

        print(f"\n=== Result Assistant Content (Sample {i}) ===")
        print(assistant_msg["content"])
        print("...\n=== End Result ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test case")
    parser.add_argument("paths", nargs="*", help="JSON files to process")
    args = parser.parse_args()

    if args.test:
        test_sample()
    elif args.paths:
        main(args.paths)
    else:
        print("Please provide input paths as arguments.")
