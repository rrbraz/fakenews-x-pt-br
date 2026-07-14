#!/usr/bin/env python
"""
Script to enrich the final_dataset with LLM analysis.

This script loads the final_dataset.csv, calls an LLM model via OpenRouter API
with a custom prompt for each tweet, and adds the LLM analysis to the dataset.

Usage:
    python enrich_dataset.py [--input INPUT_PATH] [--output OUTPUT_PATH]
                             [--model MODEL_NAME] [--api-key API_KEY]
                             [--sample SAMPLE_SIZE]

Example:
    # Process the entire dataset with default model
    python enrich_dataset.py

    # Process 10 random samples with Claude 3 Haiku
    python enrich_dataset.py --sample 10 --model "anthropic/claude-3-haiku:beta"

    # Specify custom input and output paths
    python enrich_dataset.py --input "path/to/input.parquet" --output "path/to/output.parquet"

Requirements:
    - OpenRouter API key (set as OPENROUTER_API_KEY environment variable or pass with --api-key)
    - Python packages: pandas, requests, tqdm
"""

import os
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.utils.llm_enrichment import enrich_dataset, load_dataset

# Load .env (OPENROUTER_API_KEY) before reading os.environ.
load_dotenv()


def main():
    """Run the dataset enrichment process."""

    parser = argparse.ArgumentParser(description="Enrich dataset with LLM analysis")
    parser.add_argument(
        "--input",
        type=str,
        default="data/dataset.parquet",
        help="Path to the input file (CSV or Parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated/dataset_enriched.parquet",
        help="Path to save the enriched Parquet file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-haiku-4.5",
        help="LLM model to use (OpenRouter id). "
             "Suggested fast options: anthropic/claude-haiku-4.5, "
             "google/gemini-3.1-flash-lite, openai/gpt-5.4-mini.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (if not set in environment)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of rows to process (for testing or partial processing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent API calls (default 1 = serial). 8--16 typically "
             "saturates OpenRouter's per-key rate limits for light models.",
    )

    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenRouter API key: ")
        # Set it for this session
        os.environ["OPENROUTER_API_KEY"] = api_key

    # Print configuration
    print("\n=== Dataset Enrichment Configuration ===")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Sample size: {'All' if args.sample is None else args.sample}")
    print("=======================================\n")

    # Load dataset
    start_time = time.time()
    df = load_dataset(args.input)

    # Enrich dataset
    print(f"\nStarting enrichment process with model: {args.model}")
    if args.sample:
        print(f"Processing {args.sample} samples (random selection)")
    else:
        print(f"Processing all {len(df)} rows")

    df_enriched = enrich_dataset(
        df,
        model=args.model,
        api_key=api_key,
        sample_size=args.sample,
        output_path=args.output,
        workers=args.workers,
    )

    # Print summary
    end_time = time.time()
    duration = end_time - start_time

    print("\n=== Enrichment Process Complete ===")
    print(f"Processed {len(df_enriched)} rows")
    print(f"Time taken: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Enriched dataset saved to: {args.output}")
    print("===================================\n")

    # Print a sample of the enriched data
    print("Sample of enriched data:")
    sample_df = df_enriched.sample(min(3, len(df_enriched)))
    for i, row in sample_df.iterrows():
        print(f"\n--- Tweet {i} ---")
        print(f"Text: {row['full_text'][:100]}...")
        print(f"Label: {row['label']}")
        print(f"LLM Analysis: {row['llm_analysis'][:200]}...")


if __name__ == "__main__":
    main()
