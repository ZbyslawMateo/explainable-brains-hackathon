# Explainable Brains Hackathon

**Copenhagen · Tuesday May 26th, 2026 · 16:00–20:00**  
*From signals to understanding — a 4-hour sprint to make complex brain imaging data accessible, interpretable, and actionable.*

---

The brain is under pressure. Neurological and mental health conditions are among the most widespread and least solved problems in medicine. As populations age and daily life grows more cognitively demanding, the burden on patients, families, healthcare systems, and economies continues to grow.

Vibraint ApS ([vibraint.dk](https://vibraint.dk)) builds tools to accelerate treatment discovery for brain diseases. Their platform processes complex 3D microscopy scans of rodent brains into interactive, interpretable brain maps making it possible to track how drugs distribute across brain regions, map neural activity patterns in response to treatment, and localise specific receptors and cell types at the scale and resolution that drug development demands.

Drug development for brain diseases has one of the highest failure rates in medicine. The bottleneck is not the amount of data. It is the complexity that makes exploration and interpretation far from straightforward.

> **Overarching question: How can we extract meaningful information from complex brain imaging data?**

---

## Challenge A — Smart image data selection for generalizable AI models

Training AI models for brain imaging is not bottlenecked by compute — it is bottlenecked
by data selection. A smaller, well-curated dataset often outperforms a large, noisy one.
Reliable ground truth labels are generated through time-intensive semi-manual processes,
so choosing *which* patches to label matters enormously.

The challenge is to automatically identify the most informative signal patterns that
represent the diversity of the dataset, enabling models to generalize well while
minimizing the need for manual labeling.

*Standard laptop is sufficient — embeddings are precomputed. GPU useful only if you want to run custom models — see [LightningAI](#lightningai-studio) below.*

**→ [Challenge A — solution, data, and quick start](CHALLENGE_A.md)**


## Challenge B — Guided brain data exploration for biological insight

Brain scans go through signal extraction and quantification in Vibraint's analysis
pipeline. The final output is rich but complex: spreadsheets summarizing quantified
signal per brain region and sample, statistical comparisons between groups, and spatial
brain maps. This data is difficult to visualize intuitively, hard to navigate, and
challenging to interpret without specialist tools.

*Works on any laptop. No GPU needed.*

**→ [Challenge B — solution, data, and quick start](CHALLENGE_B.md)**

---

## Setup

### 1. Fork and clone the repo

First, click **Fork** at the top right of this page to copy the repo to your GitHub account.  
Then, clone your fork:
```bash
git clone https://github.com/explainable-brains/explainable-brains-hackathon.git
cd explainable-brains-hackathon
```

### 2. Create the environment

With conda:
```bash
conda env create -f environment.yml
conda activate explainable-brains
```

With pip:
```bash
pip install -r requirements.txt
```

### 3. Bucket access

Data lives in a cloud bucket. Credentials are in `bucket_access/config.py` — already
in the repo for the duration of the hackathon.

```python
from bucket_access.bucket_utils import list_files, download_file, read_h5_patches

# see what's in the bucket
list_files('challengeA/')
list_files('challengeB/')
```

See [bucket_access/bucket_utils.py](bucket_access/bucket_utils.py) for all available functions.

### 4. Set up Claude Code. See instructions below.

### 5. Work on the challenge. Submit your work

Push your code to your fork before demos start at 18:55:

```bash
git add .
git commit -m "hackathon submission, team X"
git push
```

Share your fork URL when you demo so the judges and other teams can see what you built.

---

## Claude Code setup

Claude Code is an AI coding assistant that runs in your terminal and reads, writes,
and executes code across your whole project.

### Option A — Anthropic API credits (no subscription needed)

Anthropic is providing **$20 in API credits** per participant.

1. Claim your credits at **[appliedfutures.io/hackathons/explainable-brains](https://appliedfutures.io/hackathons/explainable-brains)**
   *(link goes live at 16:00 — use your Organization ID from console.anthropic.com, not your claude.ai user ID)*

2. Install Claude Code:

   **Mac / Linux:**
   <pre>curl -fsSL https://claude.ai/install.sh | bash</pre>

    **Windows:**
   Download and run the installer from [claude.ai/download](https://claude.ai/download)

3. Set your API key:

   **Mac / Linux:**
   <pre>echo "export ANTHROPIC_API_KEY=sk-ant-..." >> ~/.bashrc
   source ~/.bashrc</pre>

   **Windows — PowerShell:**
   <pre>[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")</pre>

   **Windows — UI:**
   Search "environment variables" in the Start menu → Edit environment variables for your account → New → add `ANTHROPIC_API_KEY` and your key value

4. Run:
```bash
   claude
```

### Option B — Already have Claude Code running with a subscription

You can use the hackathon API credits instead of your subscription.

1. Claim your credits at **[appliedfutures.io/hackathons/explainable-brains](https://appliedfutures.io/hackathons/explainable-brains)**
   *(use your Organization ID from console.anthropic.com)*

2. Set the API key — this overrides your subscription automatically, no reinstall needed:

   **Mac / Linux:**
   <pre>echo "export ANTHROPIC_API_KEY=sk-ant-..." >> ~/.bashrc
   source ~/.bashrc</pre>

   **Windows — PowerShell:**
   <pre>[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")</pre>

   **Windows — UI:**
   Search "environment variables" in the Start menu → Edit environment variables for your account → New → add `ANTHROPIC_API_KEY` and your key value

---

## LightningAI Studio

If you need more compute for Challenge A — more CPU, RAM, or GPU for custom model work.

1. Go to [lightning.ai](https://lightning.ai) and create a free account
2. Create a new Studio — Python template
3. Clone the repo and install any missing packages into the existing environment:
```bash
git clone https://github.com/explainable-brains/explainable-brains-hackathon.git
cd explainable-brains-hackathon
```
4. Set up Claude Code:
```bash
curl -fsSL https://claude.ai/install.sh | bash
echo "export ANTHROPIC_API_KEY=sk-ant-..." >> ~/.bashrc
source ~/.bashrc
claude
```
