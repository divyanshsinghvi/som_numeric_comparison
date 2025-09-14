# Debugging Numeric Comparisons in LLMs

This repository contains the code and datasets accompanying the blog post:  
[Debugging Numeric Comparisons in LLMs](https://divyanshsinghvi.github.io/blog/2025/debugging-numeric-comparisons-llms)

## Key Items
- `experiment.ipynb` — main notebook with experiments, visualizations, probing & ablation code  
- `dataset_gen1.py` — script to generate numeric comparison datasets  
- `gemma_numeric_ab_dataset.jsonl`, `gemma_string_ab_dataset.jsonl` — datasets for numeric vs. string comparisons  
- Supporting helper scripts and requirements files for reproducibility  

## Getting Started
```bash
# Clone the repo
git clone https://github.com/divyanshsinghvi/som_numeric_comparison.git
cd som_numeric_comparison

# Install dependencies
pip install -r requirements.txt

# Run dataset generation
python dataset_gen1.py
