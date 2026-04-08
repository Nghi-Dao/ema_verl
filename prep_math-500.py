import os
from datasets import load_dataset

def prepare_math500_data():
    print("Downloading MATH-500 Test Set...")
    # Load the standard MATH-500 evaluation dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # The exact system prompt from the MaxRL paper for SmolLM
    system_prompt = "Please reason step by step and put the final answer in \\boxed{}"

    def process_example(example):
        # Format the user prompt
        question = example["problem"] + " Let’s think step by step and output the final answer within \\boxed{}."

        # Extract the ground truth (MATH-500 already provides the clean extracted answer)
        answer = str(example["answer"]).strip()

        # Return the exact schema verl requires
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "reward_model": {"ground_truth": answer},
            "data_source": "math-500"  # Fixed key
        }

    print("Formatting dataset for verl...")
    # Apply formatting and drop the old columns
    processed_ds = ds.map(process_example, remove_columns=ds.column_names)

    # Save to the directory your bash script points to
    out_dir = os.path.expanduser("~/data/math-500")
    os.makedirs(out_dir, exist_ok=True)
    
    test_path = os.path.join(out_dir, "test.parquet")

    processed_ds.to_parquet(test_path)
    print(f"✅ Successfully saved MATH-500 Test to {test_path} ({len(processed_ds)} rows)")

if __name__ == "__main__":
    prepare_math500_data()
