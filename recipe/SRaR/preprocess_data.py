"""
Preprocess DAPO math rubric data for RaR/SRaR training.
Converts raw parquet with 'problem', 'rubric', 'ground_truth' columns
into verl-compatible format with:
  - prompt: list of chat message dicts (with format prefix)
  - extra_info: dict containing rubric text
  - reward_model: dict with ground_truth
  - data_source: string identifier

The format prompt follows Section B.3 of "Step-wise Rubric Rewards for LLM Reasoning":
instructs the model to use ### Step N: format and \\boxed{} for final answer.
"""

import argparse
import json

import pandas as pd

FORMAT_PROMPT_PREFIX = """Solve the following math problem step by step. Follow these formatting rules:
1. Steps: Break your solution into multiple clear steps. Begin each step with "### Step N:" (e.g., ### Step 1:, ### Step 2:, ### Step 3:).
2. Final Answer: End your response with the answer inside \\boxed{}.

Problem: """


def process_train(input_path: str, output_path: str):
    """Process training data with rubrics."""
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Columns: {df.columns.tolist()}")

    records = []
    for _, row in df.iterrows():
        problem = str(row["problem"]).strip()
        rubric = str(row["rubric"]).strip() if pd.notna(row.get("rubric")) else ""
        ground_truth = row["ground_truth"]

        # Convert ground_truth to string
        if isinstance(ground_truth, float) and ground_truth == int(ground_truth):
            ground_truth = str(int(ground_truth))
        else:
            ground_truth = str(ground_truth)

        # Build the prompt with format prefix
        user_content = FORMAT_PROMPT_PREFIX + problem

        record = {
            "prompt": [{"role": "user", "content": user_content}],
            "extra_info": {"rubric": rubric},
            "reward_model": {"ground_truth": ground_truth},
            "data_source": "math_dapo",
        }
        records.append(record)

    out_df = pd.DataFrame(records)
    out_df.to_parquet(output_path, index=False)
    print(f"Saved {len(out_df)} rows to {output_path}")


def process_val(input_path: str, output_path: str):
    """Process validation data (no rubrics needed)."""
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Columns: {df.columns.tolist()}")

    records = []
    for _, row in df.iterrows():
        problem = str(row["problem"]).strip()
        ground_truth = str(row["ground_truth"]).strip()
        data_source = str(row.get("data_source", "math")) if "data_source" in df.columns else "math"
        data_source = data_source.lower()

        # Build the prompt with format prefix
        user_content = FORMAT_PROMPT_PREFIX + problem

        record = {
            "prompt": [{"role": "user", "content": user_content}],
            "extra_info": {"rubric": ""},
            "reward_model": {"ground_truth": ground_truth},
            "data_source": data_source,
        }
        records.append(record)

    out_df = pd.DataFrame(records)
    out_df.to_parquet(output_path, index=False)
    print(f"Saved {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for RaR/SRaR training")
    parser.add_argument("--mode", choices=["train", "val"], required=True)
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    if args.mode == "train":
        process_train(args.input, args.output)
    else:
        process_val(args.input, args.output)
