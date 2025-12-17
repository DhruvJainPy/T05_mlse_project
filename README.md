# Multimodal Comic Panel Retrieval Using CLIP and Vector Databases

### Team Members
* **Dhruv Jain** — 202418020
* **Aayan Kumar** — 202418001
* **Mihir Pandya** — 202418032
* **Naman Gandhi** — 202418038

---

## 📖 Project Overview

Comics are a deeply multimodal storytelling medium where meaning emerges from the interaction between images, dialogue, narration, and narrative flow. Traditional retrieval systems fail to capture this complexity because they operate on either text or images in isolation.

This project implements a **multimodal semantic retrieval system** for comic panels using **CLIP (Contrastive Language–Image Pretraining)** and vector similarity search. The system is demonstrated using Spider-Man comic panels enriched with structured metadata and narrative context.

**The system supports:**
* **Text → Image retrieval:** Retrieve relevant comic panels using natural language queries.
* **Image → Text retrieval:** Retrieve dialogue, narration, and narrative context from a given panel image.

## ⚠️ Key Challenges Addressed
* **Multiple text modalities per panel:** Handling dialogue, narration, summaries, and context simultaneously.
* **Weak alignment:** Addressing the gap between visuals and specific dialogue lines.
* **Failure of keyword-based search:** Overcoming issues with paraphrasing and abstraction.
* **Cross-modal retrieval:** The need for true semantic understanding between text and image.

## 📂 Dataset and Metadata

Each comic panel is manually segmented and stored as an individual image. Every panel is associated with rich metadata, including:
* Dialogue and narration
* Scene summary
* Story context (before and after the panel)
* Characters present

**Note:** All textual fields are concatenated into a single enriched text representation before embedding to maximize semantic coverage.

## ⚙️ Methodology

### 1. Text Enrichment
All panel-related textual information is merged into a single semantically dense string. This improves CLIP’s alignment between the visual and textual modalities.

### 2. Embedding Generation
* **Image Embeddings:** Generated using the CLIP image encoder.
* **Text Embeddings:** Generated using the CLIP text encoder.
* *Normalization:* All embeddings are L2-normalized (512 dimensions).

### 3. Vector Database
* **FAISS** is used for fast similarity search.
* Separate indexes are maintained for image and text embeddings.
* Inner product similarity is used (equivalent to cosine similarity for normalized vectors).

### 4. Retrieval Pipelines
* **Text → Image:** Text query → Search Image Index.
* **Image → Text:** Image query → Search Text Index.
* *Re-ranking:* Optional Llama-4 Instruct re-ranking improves semantic precision.

## 🛠 Technologies Used
* Python
* PyTorch
* OpenCLIP
* FAISS
* Streamlit
* NVIDIA Llama-4 Instruct API

## 🚀 Streamlit Application

**🔗 Live Demo:**
👉 ([https://<your-streamlit-app-name>.streamlit.app](https://spiderversecomic.streamlit.app/Chatbot))
*(Replace with your actual Streamlit Cloud URL)*

## 💻 How to Run Locally

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/T05_mlse_project.git

# Navigate to the project directory
cd T05_mlse_project

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
