import sys
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Add R2Gen to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# src/utils -> src -> HyperLogicRAG -> MICCAI2026 -> R2Gen
r2gen_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'R2Gen'))
if os.path.exists(r2gen_path) and r2gen_path not in sys.path:
    sys.path.append(r2gen_path)

def compute_scores(gts, res):
    """
    Compute BLEU-1, BLEU-4, METEOR, ROUGE-L
    Wrapper around R2Gen metrics or fallback to NLTK/Rouge
    
    Args:
        gts (dict): Ground truths {id: [text]}
        res (dict): Results {id: [text]}
    """
    # Try importing R2Gen metrics if available in path
    # (Assuming R2Gen is in python path)
    try:
        from modules.metrics import compute_scores as r2gen_compute
        return r2gen_compute(gts, res)
    except ImportError:
        pass

    # Fallback to NLTK
    bleu1s = []
    bleu4s = []
    chencherry = SmoothingFunction()
    
    for key in res:
        candidate = res[key][0].split()
        references = [gts[key][0].split()]
        
        # BLEU-1
        b1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
        # BLEU-4
        b4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        
        bleu1s.append(b1)
        bleu4s.append(b4)
        
    return {
        'BLEU_1': sum(bleu1s)/len(bleu1s) if bleu1s else 0.0,
        'BLEU_4': sum(bleu4s)/len(bleu4s) if bleu4s else 0.0,
        'METEOR': 0.0, # NLTK METEOR requires extra setup, skipping for fallback
        'ROUGE_L': 0.0 # Skipping for fallback
    }
