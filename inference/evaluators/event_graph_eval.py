import re

############################################################
# 1. relation space for each dataset
############################################################

MAVEN_REL = {
    "BEFORE", "AFTER", "OVERLAP", "CONTAINS",
    "SIMULTANEOUS", "ENDS-ON", "BEGINS-ON",
    "CAUSE", "PRECONDITION", "SUBEVENT"
}

MAVEN_CAUSAL_REL = {"CAUSE", "PRECONDITION"}
MAVEN_SUBEVENT_REL = {"SUBEVENT"}
MAVEN_TEMPORAL_REL = {
    "BEFORE", "AFTER", "OVERLAP", "SIMULTANEOUS",
    "ENDS-ON", "BEGINS-ON"
}

MATRES_REL = {"BEFORE", "AFTER", "EQUAL", "VAGUE"}

HIEVENT_REL = {"SuperSub", "SubSuper"} 


############################################################
# 2. parse event_graph text
############################################################

EDGE_PATTERN = re.compile(r"\((E\d+)(?::[^,]+)?,\s*([A-Za-z\-]+),\s*(E\d+)(?::[^,)]+)?\)")

def parse_event_graph_edges(txt):
    edges = []
    for m in EDGE_PATTERN.finditer(txt):
        src, rel, dst = m.group(1), m.group(2), m.group(3)
        edges.append((src, rel, dst))
    return edges


############################################################
# 3. restore pairwise for each dataset
############################################################

def convert_edges_to_maven_ere(edges):
    # Keep only MAVEN-ERE relations
    out = []
    for (s, r, d) in edges:
        if r in MAVEN_REL:
            out.append((s, r, d))
    return out


def convert_edges_to_hievent(edges):
    # Both SuperSub / SubSuper → SUBEVENT
    out = []
    for (s, r, d) in edges:
        if r in {"SuperSub", "SubSuper"}:
            out.append((s, "SUBEVENT", d))
    return out


def convert_edges_to_matres(edges):
    # only temporal 4 relations
    out = []
    for (s, r, d) in edges:
        if r in MATRES_REL:
            out.append((s, r, d))
    return out


def convert_edges_to_maven_causal(edges):
    out = []
    for (s, r, d) in edges:
        if r in MAVEN_CAUSAL_REL:
            out.append((s, r, d))
    return out


def convert_edges_to_maven_subevent(edges):
    out = []
    for (s, r, d) in edges:
        if r in MAVEN_SUBEVENT_REL:
            out.append((s, r, d))
    return out


def convert_edges_to_maven_temporal(edges):
    out = []
    for (s, r, d) in edges:
        if r in MAVEN_TEMPORAL_REL:
            out.append((s, r, d))
    return out


############################################################
# 4. micro P/R/F1 computation
############################################################

def micro_f1(gold, pred):
    gold = set(gold)
    pred = set(pred)

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    P = tp / (tp + fp + 1e-9)
    R = tp / (tp + fn + 1e-9)
    F1 = 2 * P * R / (P + R + 1e-9)

    return P, R, F1


############################################################
# 5. unified evaluator for a SINGLE SAMPLE
############################################################

def evaluate_event_graph_original_metrics(sample):
    """
    sample = {
       "task": "event_graph",
       "subtask": "Hievent" / "ERE" / "MATRES",
       "predict": "<event_graph>...</event_graph>",
       "label":   "<event_graph>...</event_graph>"
    }
    """

    sub = sample["subtask"]
    gold_edges = parse_event_graph_edges(sample["label"])
    pred_edges = parse_event_graph_edges(sample["predict"])

    ###############################
    # HiEvent
    ###############################
    if sub == "Hievent":
        gold = convert_edges_to_hievent(gold_edges)
        pred = convert_edges_to_hievent(pred_edges)

        P, R, F1 = micro_f1(gold, pred)
        return {
            "subtask": "Hievent",
            "precision": P,
            "recall": R,
            "f1": F1
        }

    ###############################
    # MAVEN-ERE
    ###############################
    if sub == "ERE":
        gold = convert_edges_to_maven_ere(gold_edges)
        pred = convert_edges_to_maven_ere(pred_edges)

        P, R, F1 = micro_f1(gold, pred)
        return {
            "subtask": "ERE",
            "precision": P,
            "recall": R,
            "f1": F1
        }

    ###############################
    # MATRES
    ###############################
    if sub == "MATRES":
        gold = convert_edges_to_matres(gold_edges)
        pred = convert_edges_to_matres(pred_edges)

        P, R, F1 = micro_f1(gold, pred)
        return {
            "subtask": "MATRES",
            "precision": P,
            "recall": R,
            "f1": F1
        }
    
    ###############################
    # MAVEN-ERE NEW TASKS
    ###############################
    if sub == "maven_ere_causal_anchor":
        gold = convert_edges_to_maven_causal(gold_edges)
        pred = convert_edges_to_maven_causal(pred_edges)

        P, R, F1 = micro_f1(gold, pred)
        return {
            "subtask": "maven_ere_causal_anchor",
            "precision": P,
            "recall": R,
            "f1": F1
        }

    if sub == "maven_ere_subevent_anchor":
        gold = convert_edges_to_maven_subevent(gold_edges)
        pred = convert_edges_to_maven_subevent(pred_edges)

        P, R, F1 = micro_f1(gold, pred)
        return {
            "subtask": "maven_ere_subevent_anchor",
            "precision": P,
            "recall": R,
            "f1": F1
        }

    if sub == "maven_ere_temporal_anchor":
        gold = convert_edges_to_maven_temporal(gold_edges)
        pred = convert_edges_to_maven_temporal(pred_edges)

        P, R, F1 = micro_f1(gold, pred)
        return {
            "subtask": "maven_ere_temporal_anchor",
            "precision": P,
            "recall": R,
            "f1": F1
        }

    raise ValueError("Unknown subtask:", sub)

def test_evaluator():
    print("=== Testing New Format Parsing ===")
    
    # 1. MAVEN-ERE Example
    maven_ere_text = "(E1:shortened, BEFORE, E11:abandoned), (E1:shortened, BEFORE, E12:qualifying), (E1:shortened, BEFORE, E3:winning), (E1:shortened, BEFORE, E7:defeated), (E1:shortened, BEFORE, E9:ended), (E11:abandoned, BEFORE, E12:qualifying), (E11:abandoned, BEFORE, E3:winning), (E11:abandoned, BEFORE, E7:defeated), (E3:winning, BEFORE, E12:qualifying), (E5:moved, BEFORE, E1:shortened), (E5:moved, BEFORE, E10:held), (E5:moved, BEFORE, E11:abandoned), (E5:moved, BEFORE, E12:qualifying), (E5:moved, BEFORE, E2:led), (E5:moved, BEFORE, E3:winning), (E5:moved, BEFORE, E4:impacted), (E5:moved, BEFORE, E6:Played), (E5:moved, BEFORE, E7:defeated), (E5:moved, BEFORE, E8:tournament), (E5:moved, BEFORE, E9:ended), (E6:Played, BEFORE, E12:qualifying), (E6:Played, BEFORE, E3:winning), (E7:defeated, BEFORE, E12:qualifying), (E7:defeated, BEFORE, E3:winning), (E9:ended, BEFORE, E11:abandoned), (E9:ended, BEFORE, E12:qualifying), (E9:ended, BEFORE, E3:winning), (E9:ended, BEFORE, E7:defeated), (E10:held, CONTAINS, E1:shortened), (E10:held, CONTAINS, E11:abandoned), (E10:held, CONTAINS, E12:qualifying), (E10:held, CONTAINS, E2:led), (E10:held, CONTAINS, E3:winning), (E10:held, CONTAINS, E4:impacted), (E10:held, CONTAINS, E6:Played), (E10:held, CONTAINS, E7:defeated), (E10:held, CONTAINS, E9:ended), (E2:led, CONTAINS, E1:shortened), (E2:led, CONTAINS, E11:abandoned), (E2:led, CONTAINS, E12:qualifying), (E2:led, CONTAINS, E3:winning), (E2:led, CONTAINS, E4:impacted), (E2:led, CONTAINS, E6:Played), (E2:led, CONTAINS, E7:defeated), (E2:led, CONTAINS, E9:ended), (E4:impacted, CONTAINS, E1:shortened), (E4:impacted, CONTAINS, E11:abandoned), (E4:impacted, CONTAINS, E12:qualifying), (E4:impacted, CONTAINS, E3:winning), (E4:impacted, CONTAINS, E7:defeated), (E4:impacted, CONTAINS, E9:ended), (E6:Played, CONTAINS, E1:shortened), (E6:Played, CONTAINS, E11:abandoned), (E6:Played, CONTAINS, E7:defeated), (E6:Played, CONTAINS, E9:ended), (E8:tournament, CONTAINS, E1:shortened), (E8:tournament, CONTAINS, E11:abandoned), (E8:tournament, CONTAINS, E12:qualifying), (E8:tournament, CONTAINS, E2:led), (E8:tournament, CONTAINS, E3:winning), (E8:tournament, CONTAINS, E4:impacted), (E8:tournament, CONTAINS, E6:Played), (E8:tournament, CONTAINS, E7:defeated), (E8:tournament, CONTAINS, E9:ended), (E6:Played, OVERLAP, E4:impacted), (E8:tournament, OVERLAP, E10:held), (E5:moved, PRECONDITION, E10:held), (E7:defeated, PRECONDITION, E12:qualifying), (E7:defeated, PRECONDITION, E3:winning), (E4:impacted, SUBEVENT, E1:shortened), (E4:impacted, SUBEVENT, E9:ended), (E8:tournament, SUBEVENT, E1:shortened), (E8:tournament, SUBEVENT, E12:qualifying), (E8:tournament, SUBEVENT, E3:winning), (E8:tournament, SUBEVENT, E4:impacted), (E8:tournament, SUBEVENT, E6:Played), (E8:tournament, SUBEVENT, E7:defeated), (E8:tournament, SUBEVENT, E9:ended)"
    
    sample_ere = {
        "task": "event_graph",
        "subtask": "ERE",
        "predict": maven_ere_text,
        "label": maven_ere_text  # Use same as label to check perfect score
    }
    print("\n[MAVEN-ERE] Result (expect 1.0):")
    print(evaluate_event_graph_original_metrics(sample_ere))

    # 2. MATRES Example
    matres_text = "(E1:say, AFTER, E2:gathered), (E1:say, AFTER, E3:addressing), (E1:say, AFTER, E4:prohibit), (E10:know, AFTER, E14:traveling), (E10:know, BEFORE, E12:said), (E10:know, VAGUE, E11:want), (E10:know, VAGUE, E13:fallen), (E11:want, AFTER, E14:traveling), (E11:want, BEFORE, E12:said), (E11:want, BEFORE, E13:fallen), (E12:said, AFTER, E13:fallen), (E12:said, AFTER, E14:traveling), (E13:fallen, AFTER, E14:traveling), (E13:fallen, BEFORE, E15:sought), (E14:traveling, BEFORE, E15:sought), (E15:sought, BEFORE, E16:said), (E15:sought, BEFORE, E17:wanted), (E16:said, AFTER, E17:wanted), (E16:said, AFTER, E19:is), (E16:said, VAGUE, E18:admitted), (E17:wanted, BEFORE, E18:admitted), (E17:wanted, EQUAL, E19:is), (E18:admitted, AFTER, E19:is), (E18:admitted, VAGUE, E20:said), (E19:is, BEFORE, E20:said), (E2:gathered, BEFORE, E3:addressing), (E2:gathered, VAGUE, E4:prohibit), (E20:said, AFTER, E21:discovered), (E20:said, AFTER, E22:buried), (E21:discovered, AFTER, E22:buried), (E21:discovered, AFTER, E26:used), (E21:discovered, BEFORE, E23:undergoing), (E21:discovered, BEFORE, E24:said), (E21:discovered, BEFORE, E25:resembles), (E22:buried, AFTER, E26:used), (E22:buried, BEFORE, E23:undergoing), (E22:buried, BEFORE, E24:said), (E22:buried, BEFORE, E25:resembles), (E23:undergoing, AFTER, E25:resembles), (E23:undergoing, AFTER, E26:used), (E23:undergoing, AFTER, E28:found), (E23:undergoing, AFTER, E29:spotted), (E23:undergoing, BEFORE, E24:said), (E23:undergoing, BEFORE, E27:said), (E24:said, AFTER, E25:resembles), (E24:said, AFTER, E26:used), (E24:said, AFTER, E28:found), (E24:said, AFTER, E29:spotted), (E24:said, BEFORE, E27:said), (E25:resembles, AFTER, E26:used), (E25:resembles, AFTER, E28:found), (E25:resembles, AFTER, E29:spotted), (E25:resembles, BEFORE, E27:said), (E26:used, AFTER, E29:spotted), (E26:used, BEFORE, E27:said), (E26:used, BEFORE, E28:found), (E27:said, AFTER, E28:found), (E27:said, AFTER, E29:spotted), (E27:said, AFTER, E30:crossing), (E27:said, AFTER, E31:perform), (E27:said, AFTER, E32:shot), (E28:found, AFTER, E29:spotted), (E28:found, AFTER, E30:crossing), (E28:found, AFTER, E31:perform), (E28:found, AFTER, E32:shot), (E29:spotted, AFTER, E30:crossing), (E29:spotted, AFTER, E31:perform), (E29:spotted, VAGUE, E32:shot), (E3:addressing, AFTER, E4:prohibit), (E3:addressing, AFTER, E5:was), (E3:addressing, AFTER, E6:shot), (E30:crossing, AFTER, E31:perform), (E30:crossing, BEFORE, E33:killed), (E30:crossing, VAGUE, E32:shot), (E31:perform, BEFORE, E32:shot), (E31:perform, BEFORE, E33:killed), (E32:shot, BEFORE, E33:killed), (E33:killed, BEFORE, E34:giving), (E34:giving, AFTER, E35:found), (E35:found, AFTER, E39:prohibit), (E35:found, BEFORE, E36:declined), (E35:found, BEFORE, E37:pursued), (E35:found, BEFORE, E38:said), (E36:declined, AFTER, E37:pursued), (E36:declined, AFTER, E39:prohibit), (E36:declined, AFTER, E40:shot), (E36:declined, AFTER, E41:stood), (E36:declined, AFTER, E42:chatting), (E36:declined, AFTER, E43:is), (E36:declined, BEFORE, E38:said), (E37:pursued, AFTER, E39:prohibit), (E37:pursued, AFTER, E40:shot), (E37:pursued, AFTER, E41:stood), (E37:pursued, AFTER, E42:chatting), (E37:pursued, AFTER, E43:is), (E37:pursued, BEFORE, E38:said), (E38:said, AFTER, E39:prohibit), (E38:said, AFTER, E40:shot), (E38:said, AFTER, E41:stood), (E38:said, AFTER, E42:chatting), (E38:said, AFTER, E43:is), (E39:prohibit, AFTER, E41:stood), (E39:prohibit, AFTER, E43:is), (E39:prohibit, BEFORE, E40:shot), (E39:prohibit, VAGUE, E42:chatting), (E4:prohibit, AFTER, E5:was), (E4:prohibit, EQUAL, E6:shot), (E40:shot, AFTER, E41:stood), (E40:shot, AFTER, E42:chatting), (E40:shot, AFTER, E43:is), (E41:stood, BEFORE, E42:chatting), (E41:stood, VAGUE, E43:is), (E42:chatting, AFTER, E43:is), (E5:was, BEFORE, E6:shot), (E5:was, BEFORE, E7:suggests), (E5:was, BEFORE, E8:amassed), (E6:shot, BEFORE, E7:suggests), (E6:shot, BEFORE, E8:amassed), (E7:suggests, AFTER, E10:know), (E7:suggests, AFTER, E8:amassed), (E7:suggests, AFTER, E9:gotten), (E7:suggests, BEFORE, E12:said), (E7:suggests, VAGUE, E11:want), (E8:amassed, BEFORE, E10:know), (E8:amassed, BEFORE, E11:want), (E8:amassed, BEFORE, E12:said), (E8:amassed, BEFORE, E9:gotten), (E9:gotten, AFTER, E14:traveling), (E9:gotten, BEFORE, E12:said), (E9:gotten, VAGUE, E10:know), (E9:gotten, VAGUE, E11:want), (E9:gotten, VAGUE, E13:fallen)"
    
    sample_matres = {
        "task": "event_graph",
        "subtask": "MATRES",
        "predict": matres_text,
        "label": matres_text
    }
    print("\n[MATRES] Result (expect 1.0):")
    print(evaluate_event_graph_original_metrics(sample_matres))

    # 3. HiEve Example
    hieve_text = "(E12:deaths, Coref, E16:deaths), (E12:deaths, Coref, E22:kill), (E12:deaths, Coref, E26:slayings), (E15:charged, SuperSub, E24:filed), (E15:charged, SuperSub, E30:jailed), (E16:deaths, Coref, E22:kill), (E16:deaths, Coref, E26:slayings), (E19:abduct, Coref, E27:abductions), (E19:abduct, SuperSub, E23:drove), (E2:died, Coref, E10:killed), (E2:died, Coref, E6:shot), (E22:kill, Coref, E26:slayings), (E23:drove, SubSuper, E27:abductions), (E6:shot, Coref, E10:killed)"
    
    sample_hieve = {
        "task": "event_graph",
        "subtask": "Hievent",
        "predict": hieve_text,
        "label": hieve_text
    }
    print("\n[HiEve] Result (expect 1.0):")
    print(evaluate_event_graph_original_metrics(sample_hieve))

    # 4. MAVEN-ERE CAUSAL ANCHOR
    causal_text_pred = "(E1, CAUSE, E2), (E1, BEFORE, E2)"
    causal_text_gold = "(E1, CAUSE, E2)"
    sample_causal = {
        "task": "event_graph",
        "subtask": "maven_ere_causal_anchor",
        "predict": causal_text_pred,
        "label": causal_text_gold
    }
    print("\n[MAVEN-ERE CAUSAL] Result (expect 1.0):")
    print(evaluate_event_graph_original_metrics(sample_causal))

    # 5. MAVEN-ERE SUBEVENT ANCHOR
    subevent_text_pred = "(E1, SUBEVENT, E2), (E1, CAUSE, E2)"
    subevent_text_gold = "(E1, SUBEVENT, E2)"
    sample_subevent = {
        "task": "event_graph",
        "subtask": "maven_ere_subevent_anchor",
        "predict": subevent_text_pred,
        "label": subevent_text_gold
    }
    print("\n[MAVEN-ERE SUBEVENT] Result (expect 1.0):")
    print(evaluate_event_graph_original_metrics(sample_subevent))

    # 6. MAVEN-ERE TEMPORAL ANCHOR
    temporal_text_pred = "(E1, BEFORE, E2), (E1, CAUSE, E2)"
    temporal_text_gold = "(E1, BEFORE, E2)"
    sample_temporal = {
        "task": "event_graph",
        "subtask": "maven_ere_temporal_anchor",
        "predict": temporal_text_pred,
        "label": temporal_text_gold
    }
    print("\n[MAVEN-ERE TEMPORAL] Result (expect 1.0):")
    print(evaluate_event_graph_original_metrics(sample_temporal))


if __name__ == "__main__":
    test_evaluator()