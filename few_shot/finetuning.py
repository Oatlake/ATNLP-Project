from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from dataset import SCANDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
#####pip install SentencePiece
##pip install transformers==4.30.2
##pip install peft


class SCANT5Dataset(Dataset):
    def __init__(self, scan_dataset, tokenizer, prefix="translate command to actions: ", max_len=128):
        """
        scan_dataset: your existing SCANDataset instance
        tokenizer: T5Tokenizer
        prefix: optional text prefix to give T5 a "task hint"
        """
        self.scan_dataset = scan_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prefix = prefix

    def __len__(self):
        return len(self.scan_dataset)

    def __getitem__(self, idx):
        # Original SCANDataset returns e.g. {src: tensor, tgt: tensor}
        # We can decode them into strings
        item = self.scan_dataset[idx]
        src_str = self.scan_dataset.decode(item["src"])  # e.g., "jump around right twice"
        tgt_str = self.scan_dataset.decode(item["tgt"])  # e.g., "I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT"

        # Optionally remove extra spaces or do cleanup
        src_str = src_str.strip()
        tgt_str = tgt_str.strip()

        # Create combined prompt for T5
        input_text = f"{self.prefix} {src_str}"
        # Tokenize
        source_encodings = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        target_encodings = self.tokenizer(
            tgt_str,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": source_encodings["input_ids"].squeeze(),
            "attention_mask": source_encodings["attention_mask"].squeeze(),
            "labels": target_encodings["input_ids"].squeeze(),
        }

# Initialize T5
device = torch.device("cuda" if torch.cuda.is_available() else 'mps')
model_name = "t5-small"  # or "t5-base", "flan-t5-base", etc.
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Build dataset
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

scan_dataset_train = SCANDataset(train_path)  # your existing code
scan_dataset_test = SCANDataset(test_path)
train_dataset = SCANT5Dataset(scan_dataset_train, tokenizer, max_len=128)
test_dataset = SCANT5Dataset(scan_dataset_test, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Training loss: {avg_loss:.4f}")

    # Optional: Evaluate on test_loader
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            eval_loss += loss.item()
    eval_loss /= len(test_loader)
    print(f"Epoch {epoch+1} - Eval loss: {eval_loss:.4f}")


def generate_actions_t5(model, tokenizer, command_str, device, max_len=50):
    model.eval()
    input_text = f"translate command to actions: {command_str}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_len,
        num_beams=5,   # or whatever decoding strategy you like
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage:
command = "look around right twice"
predicted_actions = generate_actions_t5(model, tokenizer, command, DEVICE)
print("Predicted: ", predicted_actions)
