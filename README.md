# 🎩 Lincoln Historiographical Divergence Analysis

**ML Evaluation Engineer Technical Assessment for Memory Machines**

An automated system to analyze how different authors describe the same historical events about Abraham Lincoln. Compares first-person accounts (Lincoln's writings) against third-party biographer accounts using an LLM-based judge.

## 🚀 Live Demo

**[View the Streamlit App →](https://your-app-name.streamlit.app)** *(Update with your deployed URL)*

---

## 📋 Project Overview

### Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| **1. Data Acquisition** | Scrape documents from Project Gutenberg & Library of Congress | Normalized JSON |
| **2. Event Extraction** | Extract claims about 5 key events using LLM | Claims + Metadata |
| **3. LLM Judge** | Compare Lincoln vs Biographers for consistency | Scores + Contradictions |
| **4. Analysis** | Statistical validation (Ablation, Self-Consistency, Kappa) | Charts + Insights |

### Key Events Analyzed

1. 🗳️ Election Night 1860
2. 🏰 Fort Sumter Decision
3. 📜 Gettysburg Address
4. 🎤 Second Inaugural Address
5. 🎭 Ford's Theatre Assassination

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/memory-machines-assessment.git
cd memory-machines-assessment

# Install dependencies with uv
uv sync

# Or run individual stages
python -m src.cli part1    # Data acquisition
python -m src.cli part2    # Event extraction
python -m src.cli part3    # LLM judge
python -m src.cli part3 --experiments  # Run all experiments
```

### Streamlit App (Visualization)

```bash
streamlit run app/Home.py
```

---

## 📊 Key Results

### Ablation Study

| Strategy | Mean Score | Std Dev |
|----------|------------|---------|
| Zero-Shot | 74.0 | 22.9 |
| **Chain-of-Thought** | **83.4** | 17.4 |
| Few-Shot | 74.6 | 13.3 |

**Finding:** Chain-of-Thought prompting achieves the best results.

### Self-Consistency

- 5 runs with temperature=0.7
- Average std dev: ~8 points
- Most stable: Gettysburg comparisons
- Least stable: Fort Sumter/Morse

### Cohen's Kappa

- Agreement: 91.7%
- Kappa: 0.0 (due to class imbalance - see report for details)

---

## 🚀 Deployment to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `app/Home.py`
5. Add secrets if needed (Settings → Secrets):
   ```toml
   OPENAI_API_KEY = "your-key"
   ```
6. Deploy!

---

## 📚 Data Sources

- **Project Gutenberg:** 5 Lincoln biographies
  - Nicolay & Hay, Ketcham, Morse, Browne, Charnwood
- **Library of Congress:** 5 Lincoln documents
  - Election letter, Fort Sumter correspondence, Gettysburg Address, Second Inaugural, Last Public Address

---

## 🧪 Technologies Used

- **Pipeline:** Python, httpx, BeautifulSoup, Pydantic
- **LLM:** OpenAI GPT-4o-mini
- **Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **UI:** Streamlit
- **Deployment:** Streamlit Cloud

---

## 📝 License

MIT License - Built for Memory Machines Technical Assessment with pip
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/Home.py
```

### Environment Variables (for live execution)

```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

---

## 📁 Project Structure

```
├── app/                          # Streamlit application
│   ├── Home.py                   # Landing page
│   ├── components/               # Shared components
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Data loading with caching
│   │   └── charts.py             # Visualization functions
│   └── pages/                    # App pages
│       ├── 1_📚_Data_Acquisition.py
│       ├── 2_🔍_Event_Extraction.py
│       ├── 3_⚖️_LLM_Judge.py
│       └── 4_📊_Analysis.py
├── src/                          # Core pipeline code
│   ├── part1_acquisition/        # Data scraping
│   ├── part2_extraction/         # LLM extraction
│   ├── part3_judge/              # LLM judge
│   └── contracts/                # Data schemas
├── data/                         # Generated data
│   ├── normalized/               # Stage 1 output
│   ├── extractions/              # Stage 2 output
│   └── judgments/                # Stage 3 output
├── .streamlit/
│   └── config.toml               # Streamlit theming
├── pyproject.toml                # Project configuration
├── requirements.txt              # Dependencies
└── README.md
```

---

## 🔧 Running the Pipeline

### Full Pipeline (CLI)

```bash
# Run all stages
python -m src.cli all

# Or