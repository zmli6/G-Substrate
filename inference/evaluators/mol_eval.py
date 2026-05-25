# mol_eval.py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

rouge = Rouge()
smooth = SmoothingFunction().method1

def eval_molecular(item, gt_root=None):
    """
    item example:
    {
        "task": "mol",
        "subtask": "",
        "prompt": "....",
        "predict": "The molecule is ...",
        "label": "The molecule is ...\n"
    }
    """
    pred = item["predict"].strip()
    gt   = item["label"].strip()

    # BLEU
    bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth)

    # ROUGE-L
    rouge_l = rouge.get_scores(pred, gt)[0]["rouge-l"]["f"]

    return {"bleu": bleu, "rouge_l": rouge_l}

if __name__ == "__main__":
    item = {"task": "mol", "subtask": "", "prompt": "user\nSMILES:\nCCC/C=C/C=C/C1Cc2cc(O)cc(O)c2C(=O)O1\n\nWhat can you tell me about this molecule?\nassistant\n", "predict": "The molecule is a natural product found in Aspergillus fischeri with data available.", "label": "The molecule is a natural product found in Aspergillus with data available.\n"}
    print(eval_molecular(item))