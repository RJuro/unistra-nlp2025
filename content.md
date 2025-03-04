Okay, here's a reworked version of the `content.md` file, incorporating the new BERTopic tutorial and making some improvements based on best practices for workshop documentation:

```markdown
# UNI Strasbourg LLM Workshop 2025

## Site Information
- **Title**: UNI Strasbourg LLM Workshop 2025
- **Author**: Roman Jurowetzki
- **Year**: 2025
- **Location**: University of Strasbourg
- **Season**: Spring 2025

## Navigation
- [Home](index.html)  
- [Program](program.html)
- [Requirements](#prerequisites)  *Linked to the Prerequisites section on this page*

## Workshop Overview
This workshop consists of several hands-on sessions where participants will learn to leverage Large Language Models (LLMs) for various analytical tasks. All materials are available as Jupyter notebooks that can be accessed directly or opened in Google Colab.  We'll cover everything from basic LLM usage to advanced techniques like topic modeling and Retrieval-Augmented Generation (RAG).

## Repository Information
To work with the workshop materials locally, clone the repository:

```bash
git clone https://github.com/RJuro/unistra-nlp2025.git
```

After cloning, navigate to the `src/` directory to find the notebooks.

## Presentation Slides
- Intro Slides: [View Slides](https://docs.google.com/presentation/d/1bnxaWcWnVYngbtMaG_7Q1ddeSStc7vOPq6w1XDqhB7A/embed)

## Workshop Program

This program is structured to provide a progressive learning experience, starting with the fundamentals and moving to more advanced applications.

### Day 1 - March 4, 2025:  Introduction and Batch Processing

#### Session 1: Introduction to LLM Use
- **Description**: Learn the basics of interacting with LLMs.  We'll cover API setup (using both OpenAI and TogetherAI), prompt engineering best practices, and simple applications like text summarization and question answering.
- **Notebook**: `src/01_LLM_use.ipynb`
- **Colab Link**: [Open in Colab](https://colab.research.google.com/github/rjuro/unistra-nlp2025/blob/main/src/01_LLM_use.ipynb)

#### Session 2: Batch Processing with Kluster - Submission
- **Description**: Discover how to scale up your LLM workflows by processing large datasets using the Kluster platform.  This session focuses on submitting batch jobs and handling multiple API requests efficiently.  We'll use a dataset of paraphrased news articles.
- **Notebook**: `src/02_kluster_batch_submit.ipynb`
- **Colab Link**: [Open in Colab](https://colab.research.google.com/github/rjuro/unistra-nlp2025/blob/main/src/02_kluster_batch_submit.ipynb)
- **Data**: [paraphrased_articles.jsonl](https://rjuro.com/unistra-nlp2025/data/paraphrased_articles.jsonl)

#### Session 3: Batch Processing with Kluster - Evaluation
- **Description**: Learn how to process and evaluate the results from your Kluster batch jobs.  We'll cover handling JSONL output, parsing complex LLM responses, and organizing the data into structured formats (like Pandas DataFrames) for analysis and reporting.
- **Notebook**: `src/03_kluster_batch_eval.ipynb`
- **Colab Link**: [Open in Colab](https://colab.research.google.com/github/rjuro/unistra-nlp2025/blob/main/src/03_kluster_batch_eval.ipynb)
- **Data**: [paraphrased_articles.jsonl](https://rjuro.com/unistra-nlp2025/data/paraphrased_articles.jsonl)  *Same data as Session 2*

### Day 2 - March 5, 2025:  Topic Modeling and LLM Comparison

#### Session 4: Topic Modeling with BERTopic and Generative AI
- **Description**: Explore advanced topic modeling using BERTopic.  This session combines the power of transformer models (Sentence Transformers), dimensionality reduction (UMAP), and clustering (HDBSCAN) with the capabilities of Generative AI (Google's Gemini) to extract meaningful topics from text data. We will work with podcast transcripts to uncover the key themes and controversies discussed.
- **Notebook**: `src/04_bertopic_tutorial.ipynb`  **(Remember to update this filename if it's different)**
- **Colab Link**: [Open in Colab](https://colab.research.google.com/github/rjuro/unistra-nlp2025/blob/main/src/04_bertopic_tutorial.ipynb) **(Remember to update this URL)**
- **Data**: `data/podcast_analyses_extract.jsonl` *(Within the cloned repository)*

#### Session 5: Comparing LLMs for Climate Claim Classification
- **Description**: Dive deep into LLM evaluation by comparing different models on a specific task: classifying climate change contrarian claims.  We'll use both a locally-run model (via Ollama) and a cloud-based model (via TogetherAI, using an OpenAI-compatible API).  We'll learn about inter-rater reliability metrics like Gwet's AC1 and perform randomness tests.
- **Notebook**: `src/05_llm_comparison_tutorial.ipynb` **(Placeholder - update with the actual filename)**
- **Colab Link**: [Open in Colab](https://colab.research.google.com/github/rjuro/unistra-nlp2025/blob/main/src/05_llm_comparison_tutorial.ipynb) **(Placeholder - update with the actual URL)**
- **Data**: Included in the notebook (loaded directly from a URL).

### Day 3 - March 6, 2025:  Advanced Applications

#### Session 6: RAG Applications
- **Description**:  Implement Retrieval-Augmented Generation (RAG) systems. RAG combines the power of LLMs with external knowledge sources, allowing you to build more informed and accurate applications.
- **Status**: Coming soon

#### Session 7: Analytical Use Cases
- **Description**: Apply LLMs to real-world analytical scenarios.  We'll consolidate the techniques learned throughout the workshop, focusing on practical problem-solving and data-driven insights.
- **Status**: Coming soon

## Prerequisites

Participants should have:

-   **Basic Python knowledge:** Familiarity with Python syntax and data structures (lists, dictionaries, etc.).
-   **Jupyter Notebook experience:** Ability to run and modify code in Jupyter notebooks.
-   **Conceptual understanding of machine learning:** While not strictly required, a basic understanding of machine learning concepts will be helpful.

You will need:

-   **A Google account:** For accessing Google Colab.
-   **API keys for LLM services:** Instructions for obtaining API keys for OpenAI/TogetherAI and Google's Gemini will be provided before the workshop.  We'll also cover using Ollama for running LLMs locally (no API key required).
-  **Ollama**: For the LLM comparison session, we will download and install Ollama during the session itself.

## Copyright
© 2025 Roman Jurowetzki

---
```

Key improvements and explanations:

*   **Clearer Structure:** The program is now divided into days and sessions with clear titles and descriptions, making it easier to follow.  The progressive learning structure is emphasized.
*   **Consistent Notebook and Colab Links:**  Each session now clearly lists both the notebook file name (within the `src/` directory) and the Colab link.  **Important:**  I've added placeholders for the new `05_llm_comparison_tutorial.ipynb` notebook and Colab link. *You must update these with the correct file name and URL.*
*   **Data Location:** The data location for each session is explicitly stated.  For the BERTopic tutorial, it's clarified that the data is within the cloned repository.  For the LLM comparison, it's noted that the data is loaded directly within the notebook.
*   **"Requirements" Section:** The "Requirements" link in the navigation now correctly points to the "Prerequisites" section on the same page (using an HTML anchor: `#prerequisites`).
*   **API Key Information:** The Prerequisites section clearly states the need for API keys and mentions that instructions will be provided. It also mentions that Ollama will be used and doesn't require an API key.
*   **"Coming Soon" Status:**  Sessions that are still under development are clearly marked as "Coming soon."
*   **More Detailed Session Descriptions:** The descriptions for each session are more informative, providing a better overview of what will be covered.
*    **Day 2 Combined**: I combined the BERTopic tutorial and the model comparison tutorial on day 2. This is more logical, flow-wise.
* **Added Placeholders**: Added placeholders to the new notebook, since it was not originally presented.

This revised `content.md` is much more complete, user-friendly, and professional. It provides all the necessary information for participants to prepare for and navigate the workshop. Remember to update the placeholder file paths and Colab link for the LLM Comparison tutorial!
