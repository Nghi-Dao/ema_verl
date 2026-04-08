import os
from datasets import load_dataset

def prepare_gsm8k_data():
    # The exact system prompt from the MaxRL paper for SmolLM
    system_prompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
    
    # Define splits to process
    splits = {
        "train": "train",
        "test": "test"
    }

    out_dir = os.path.expanduser("~/data/gsm8k")
    os.makedirs(out_dir, exist_ok=True)

    for split_name, hf_split in splits.items():
        print(f"🔄 Processing GSM8K {split_name} set...")
        
        # Load the specific split
        ds = load_dataset("openai/gsm8k", "main", split=hf_split)

        def process_example(example):
            # Format the user prompt
            question = example["question"] + " Let’s think step by step and output the final answer within \\boxed{}."

            # Extract the ground truth (everything after ####)
            answer = example["answer"].split("####")[1].strip()

            # Return the exact schema verl requires
            return {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "reward_model": {"ground_truth": answer},
                "data_source": "gsm8k"
            }

        # Apply formatting and drop old columns
        processed_ds = ds.map(process_example, remove_columns=ds.column_names)

        # Save to parquet
        out_path = os.path.join(out_dir, f"{split_name}.parquet")
        processed_ds.to_parquet(out_path)
        
        print(f"✅ Successfully saved {len(processed_ds)} examples to {out_path}")

if __name__ == "__main__":
    prepare_gsm8k_data()
