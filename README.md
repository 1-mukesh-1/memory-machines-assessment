# Lincoln Historiographical Divergence Analysis

Compares Lincoln's own writings (Library of Congress) against biographer accounts (Project Gutenberg) using an LLM judge to detect inconsistencies.

**Live Demo:** https://memory-machines-assessment-mukesh.streamlit.app/

## Setup

```bash
# Install dependencies
uv sync

# Set API key
export OPENAI_API_KEY="your-key"
```

## Run Pipeline

```bash
python -m src.cli part1              # Data acquisition
python -m src.cli part2              # Event extraction
python -m src.cli part3              # LLM judge
python -m src.cli part3 --experiments  # Run all experiments
```

## Run Streamlit App

```bash
streamlit run app/Home.py
```

## Project Structure

```
├── app/                    # Streamlit app
├── src/
│   ├── part1_acquisition/  # Scraping
│   ├── part2_extraction/   # LLM extraction
│   ├── part3_judge/        # LLM judge + experiments
│   └── contracts/          # Data schemas
├── data/
│   ├── normalized/         # Part 1 output
│   ├── extractions/        # Part 2 output
│   └── judgments/          # Part 3 output
└── report.html             # Final report
```

## Data Sources

- **Gutenberg:** 5 biographies (Nicolay & Hay, Ketcham, Morse, Browne, Charnwood)
- **Library of Congress:** 5 Lincoln documents