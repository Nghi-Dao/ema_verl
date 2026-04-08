import os
from datasets import load_dataset

def prepare_polaris_training_data():
    print("Downloading POLARIS Train Set...")
    # Load the standard POLARIS dataset (it only has a 'train' split)
    ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")

    print("Splitting into train and test sets...")
    # Reserve 1,000 examples for validation/testing
    ds = ds.train_test_split(test_size=1000, seed=42)
    train_ds = ds["train"]
    test_ds = ds["test"]

    # The exact system prompt from the MaxRL paper for SmolLM
    system_prompt = "Please reason step by step and put the final answer in \\boxed{}"

    def process_example(example):
        # Format the user prompt
        question = example["problem"] + " Let’s think step by step and output the final answer within \\boxed{}."

        # Extract the ground truth
        answer = example["answer"].strip()

        # Return the exact schema verl requires
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "reward_model": {"ground_truth": answer},
            "data_source": "polaris"  # Fixed key
        }

    print("Formatting dataset for verl...")
    # Apply formatting and drop the old columns
    processed_train = train_ds.map(process_example, remove_columns=train_ds.column_names)
    processed_test = test_ds.map(process_example, remove_columns=test_ds.column_names)

    # Save to the directory your bash script points to
    out_dir = os.path.expanduser("~/data/polaris")
    os.makedirs(out_dir, exist_ok=True)
    
    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")

    processed_train.to_parquet(train_path)
    processed_test.to_parquet(test_path)
    
    print(f"✅ Successfully saved POLARIS Train to {train_path}")
    print(f"✅ Successfully saved POLARIS Test to {test_path}")

if __name__ == "__main__":
    prepare_polaris_training_data()
