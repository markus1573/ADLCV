import argparse

import torch
from gpt import AndersenGPT
from train import (
    EMBED_DIM,
    MAX_SEQ_LEN,
    MODEL_SAVE_PATH,
    NUM_HEADS,
    NUM_LAYERS,
    POS_ENC,
    PRETRAINED_TOKENIZER,
)
from transformers import AutoTokenizer


def generate_text(model, tokenizer, prompt, max_gen_len=500, device="cpu", strategy="greedy", temp = 0.8):
    """
    Given a prompt string, generate a continuation using greedy decoding.
    The prompt is encoded using the pretrained tokenizer.
    """
    # Encode the prompt (returns a list of token ids)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_gen_len):
        # Ensure we work with the last MAX_SEQ_LEN tokens if the sequence gets too long.
        if input_ids.shape[1] > MAX_SEQ_LEN:
            input_ids = input_ids[:, -MAX_SEQ_LEN:]

        # Forward pass: get logits for all tokens in the sequence.
        logits = model(input_ids)
        
        # Get the logits for the last token only
        next_token_logits = logits[:, -1, :]

        # Two strategies for generating the next token
        if strategy == "greedy":
            # Greedy: choose the token with highest probability.
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        elif strategy == "sampling":
            # Multinomial Sampling: Sample from the probability distribution.
            probabilities = torch.softmax(next_token_logits / temp, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)

        # Append predicted token to input_ids
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # Stop early if the model generates the EOS token
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Decode the full sequence to text.
    output_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return output_text


# Standard prompts for evaluation
PROMPTS = [
    "Tell a story.",
    "Can you tell me a story about a great, big tree?",
]
STRATEGY = "sampling"  # or "sampling"
TEMP = 1.2
if STRATEGY=="greedy":
    OUTPUT_FILE = "generation_results_greedy.txt"
else:
    OUTPUT_FILE = f"generation_results_sampling_{TEMP}.txt"

@torch.no_grad()
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Loading model on {device} ...")

    # Load the same pretrained tokenizer used during training.
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)

    # GPT2 does not have a PAD token by default; set it to the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Instantiate the GPT-style model with the same hyperparameters as during training.
    model = AndersenGPT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        pos_enc=POS_ENC,
        dropout=0.0,
        fc_dim=None,
        num_tokens=tokenizer.vocab_size,
    ).to(device)

    # Load the model checkpoint.
    state_dict = torch.load(MODEL_SAVE_PATH + "/best.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate text with AndersenGPT")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Enable interactive mode for manual prompts")
    args = parser.parse_args()

    if args.interactive:
        # Interactive mode
        print("Enter a prompt and the model will generate a continuation.")
        print(f"Type 'quit' or 'exit' to stop. Results will be saved to {OUTPUT_FILE}\n")
        results = []
        prompt_num = 0
        while True:
            prompt = input("Prompt: ").strip()
            if prompt.lower() in ["quit", "exit"]:
                break
            prompt_num += 1
            generated_text = generate_text(
                model, tokenizer, prompt, max_gen_len=500, device=device, strategy = STRATEGY, temp=TEMP
            )
            results.append(f"PROMPT {prompt_num}:\n{prompt}\n\nGENERATED:\n{generated_text}\n")
            print("\n--- Generated Text ---")
            print(generated_text)
            print("----------------------\n")
        
        # Save results to file
        if results:
            with open(OUTPUT_FILE, "w") as f:
                f.write("\n" + "=" * 60 + "\n\n".join(results))
                f.write("\n" + "=" * 60 + "\nFINISHED\n")
            print(f"\nResults saved to {OUTPUT_FILE}")
    else:
        # Batch mode with standard prompts
        results = []
        for i, prompt in enumerate(PROMPTS, 1):
            print(f"Generating response for prompt {i}/{len(PROMPTS)}...")
            generated_text = generate_text(
                model, tokenizer, prompt, max_gen_len=500, device=device, strategy = STRATEGY, temp=TEMP
            )
            results.append(f"PROMPT {i}:\n{prompt}\n\nGENERATED:\n{generated_text}\n")
            print(f"  Done.")

        # Save results to file
        with open(OUTPUT_FILE, "w") as f:
            f.write("\n" + "=" * 60 + "\n\n".join(results))
            f.write("\n" + "=" * 60 + "\nFINISHED\n")

        print(f"\nResults saved to {OUTPUT_FILE}")

        # Also print to console
        print("\n" + "=" * 60)
        for result in results:
            print(result)
            print("=" * 60)
        print("FINISHED")


if __name__ == "__main__":
    main()
