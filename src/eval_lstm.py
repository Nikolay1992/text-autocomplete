import evaluate

def print_rouge_scores(val_loss, val_loader, preds_texts, targets_texts):

    rouge = evaluate.load("rouge")

    
    if preds_texts and targets_texts:
        rouge_score = rouge.compute(predictions=preds_texts, references=targets_texts)
        print(f"Val loss: {val_loss/len(val_loader):.4f} | "
              f"ROUGE-1: {rouge_score['rouge1']:.4f} | ROUGE-2: {rouge_score['rouge2']:.4f}")
    else:
        print("Val loss:", val_loss/len(val_loader), "| No valid samples for ROUGE")
        
def print_rouge_scores_test(preds_texts, targets_texts):

    rouge = evaluate.load("rouge")

    
    if preds_texts and targets_texts:
        rouge_score = rouge.compute(predictions=preds_texts, references=targets_texts)
        print(f"ROUGE-1: {rouge_score['rouge1']:.4f} | ROUGE-2: {rouge_score['rouge2']:.4f}")
