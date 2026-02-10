"""
Download and save DevRev Search datasets from Hugging Face.

Datasets:
- annotated_queries: Queries paired with annotated (golden) article chunks
- knowledge_base: Article chunks from DevRev's customer-facing support documentation
- test_queries: Held-out queries used for evaluation
"""

from datasets import load_dataset
import pandas as pd
import os

# Create data directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("DevRev Search Dataset Downloader")
print("=" * 60)

# 1. Load annotated queries
print("\n[1/3] Loading annotated_queries...")
annotated_queries = load_dataset("devrev/search", "annotated_queries", split="train")
print(f"  - Loaded {len(annotated_queries):,} annotated queries")
print(f"  - Features: {list(annotated_queries.features.keys())}")

# 2. Load knowledge base
print("\n[2/3] Loading knowledge_base...")
knowledge_base = load_dataset("devrev/search", "knowledge_base", split="corpus")
print(f"  - Loaded {len(knowledge_base):,} knowledge base chunks")
print(f"  - Features: {list(knowledge_base.features.keys())}")

# 3. Load test queries
print("\n[3/3] Loading test_queries...")
test_queries = load_dataset("devrev/search", "test_queries", split="test")
print(f"  - Loaded {len(test_queries):,} test queries")
print(f"  - Features: {list(test_queries.features.keys())}")

# Save as parquet files
print("\n" + "=" * 60)
print("Saving datasets to parquet files...")
print("=" * 60)

# Convert to pandas and save
annotated_df = annotated_queries.to_pandas()
annotated_df.to_parquet(os.path.join(DATA_DIR, "annotated_queries.parquet"), index=False)
print(f"  ✓ Saved annotated_queries.parquet")

knowledge_df = knowledge_base.to_pandas()
knowledge_df.to_parquet(os.path.join(DATA_DIR, "knowledge_base.parquet"), index=False)
print(f"  ✓ Saved knowledge_base.parquet")

test_df = test_queries.to_pandas()
test_df.to_parquet(os.path.join(DATA_DIR, "test_queries.parquet"), index=False)
print(f"  ✓ Saved test_queries.parquet")

# Print summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"\nAnnotated Queries: {len(annotated_df):,} rows")
print(annotated_df.head(3).to_string())

print(f"\nKnowledge Base: {len(knowledge_df):,} rows")
print(knowledge_df.head(3).to_string())

print(f"\nTest Queries: {len(test_df):,} rows")
print(test_df.head(3).to_string())

print("\n" + "=" * 60)
print(f"All datasets saved to {DATA_DIR}/ directory")
print("=" * 60)
