import torch
import numpy as np
from rich import print
from rich.traceback import install
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from dataset import SCANDataset

install()

def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs


def main_t5_lora(
    train_path: str,
    test_path: str,
    model_suffix: str,
    hyperparams: dict,
    random_seed: int = 42,
):
    """
    Fine-tunes a pre-trained T5 using LoRA (PEFT) on the SCAN dataset.
    """
    # Reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparams
    device = hyperparams["device"]
    learning_rate = hyperparams["learning_rate"]
    batch_size = hyperparams["batch_size"]
    epochs = hyperparams["epochs"]

    # Load the raw SCAN data
    train_dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)

    # Load a small T5 model (e.g., "t5-small")
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # ---- LoRA Configuration ----
    # We specify which modules to apply LoRA to. In T5, the relevant attention layers
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "k", "v", "o"],  
    )

    # Wrap base model with LoRA
    model = get_peft_model(base_model, lora_config)
    print(f"[{model_suffix}] LoRA parameters added to T5. Trainable params:")
    model.print_trainable_parameters()  # see how many params are actually trainable

    # Create DataLoaders with T5 tokenization in a custom collate_fn
    def collate_fn(batch):
        src_texts = []
        tgt_texts = []
        for item in batch:
            src_str = train_dataset.decode(item["src"]).strip()
            tgt_str = train_dataset.decode(item["tgt"]).strip()
            src_texts.append(src_str)
            tgt_texts.append(tgt_str)

        src_encodings = tokenizer(
            ["translate command to actions: " + s for s in src_texts],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tgt_encodings = tokenizer(
            tgt_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = src_encodings["input_ids"]
        attention_mask = src_encodings["attention_mask"]
        labels = tgt_encodings["input_ids"]
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Optimizer on top of LoRA parameters only
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                eval_loss += outputs.loss.item()
        eval_loss /= len(test_loader)

        print(
            f"[{model_suffix}] Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Test Loss: {eval_loss:.4f}"
        )

    # Inference / Accuracy
    token_acc_list = []
    seq_acc_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=50,
                num_beams=1,  
            )
            pred_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs
            ]
            
            gold_texts = []
            for labels_ids in batch["labels"]:
                labels_ids = labels_ids[labels_ids >= 0]  
                gold_texts.append(tokenizer.decode(labels_ids, skip_special_tokens=True))

            for pred, gold in zip(pred_texts, gold_texts):
                pred_tokens = pred.split()
                gold_tokens = gold.split()
                matches = sum(p == g for p, g in zip(pred_tokens, gold_tokens))
                total_tokens = max(len(gold_tokens), 1)
                token_acc = matches / total_tokens
                token_acc_list.append(token_acc)
                seq_acc = 1.0 if pred_tokens == gold_tokens else 0.0
                seq_acc_list.append(seq_acc)

    final_token_acc = float(np.mean(token_acc_list))
    final_seq_acc = float(np.mean(seq_acc_list))
    print(
        f"[{model_suffix}] Final Token Acc: {final_token_acc:.4f} | "
        f"Final Seq Acc: {final_seq_acc:.4f}"
    )

    return model, final_token_acc, final_seq_acc


def run_all_variations(n_runs=3):
    """Run LoRA training multiple times for all dataset size variations."""
    results = {}
    hyperparams = {
        "learning_rate": 5e-4,
        "batch_size": 128,
        "epochs": 10,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Prepare data pairs
    for train_path, test_path, size in get_dataset_pairs():
        results[f"p{size}"] = []

    for run in range(n_runs):
        seed = 42 + run
        print(f"\nStarting run {run+1}/{n_runs} with seed {seed}")
        print("=" * 70)

        for train_path, test_path, size in get_dataset_pairs():
            print(f"\nTraining dataset size p{size}")
            model, accuracy, seq_acc = main_t5_lora(
                train_path, test_path, f"T5_LoRA_p{size}", hyperparams, random_seed=seed
            )
            results[f"p{size}"].append((accuracy, seq_acc))

    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Accuracy ± Std Dev")
    print("-" * 50)

    for size, accuracies in results.items():
        accuracies = [(float(a), float(s)) for a, s in accuracies]
        mean = np.mean(accuracies, axis=0)
        std = np.std(accuracies, axis=0)

        print(f"{size:11} | Mean Token Acc: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[0]:.4f}' for acc in accuracies)}")
        print(f"Mean Seq  Acc: {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[1]:.4f}' for acc in accuracies)}\n")


if __name__ == "__main__":
    run_all_variations()
