import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2"  # or "gpt2-medium", "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()  # We won't be fine-tuning in few-shot



def build_fewshot_prompt(examples, new_command):
    """
    examples: list of (command, action) tuples from your training set
    new_command: the command we want GPT-2 to predict an action sequence for
    """
    prompt = ""
    for i, (cmd, act) in enumerate(examples, start=1):
        prompt += (
            f"Example {i}:\n"
            f"Command: {cmd}\n"
            f"Action: {act}\n\n"
        )
    # Now provide the new command
    prompt += f"Now, Command: {new_command}\nAction:"
    return prompt

# Suppose you pick 2 examples
fewshot_examples = [
    ("jump around right twice", "I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT"),
    ("walk left", "I_WALK I_TURN_LEFT"),
]
new_command = "look around right twice"
prompt_text = build_fewshot_prompt(fewshot_examples, new_command)

print(prompt_text)
# e.g.,
# Example 1:
# Command: jump around right twice
# Action: I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT
#
# Example 2:
# Command: walk left
# Action: I_WALK I_TURN_LEFT
#
# Now, Command: look around right twice
# Action:


def gpt2_fewshot_inference(model, tokenizer, prompt, max_length=50):
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generate
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=len(input_ids[0]) + max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,  # needed for GPT-2
    )
    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

generated_text = gpt2_fewshot_inference(model, tokenizer, prompt_text, max_length=50)
print("=== GENERATED TEXT ===")
print(generated_text)


# Find the text after "Action:" in the generated output
split_output = generated_text.split("Action:")
if len(split_output) >= 2:
    predicted_action_seq = split_output[-1].strip()
else:
    predicted_action_seq = "No Action Found"
print("Predicted Action Sequence:", predicted_action_seq)
