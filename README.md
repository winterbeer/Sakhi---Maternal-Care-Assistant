# 🤰 Sakhi - AI-Based Maternal Care Assistant

**Sakhi** is a personalized maternal healthcare assistant built using **LangChain**, **LLMs**, and **FastAPI**, designed to provide critical support to **pregnant women**, **newborns**, and **malnourished children**, especially in **rural areas**. 

This project was developed with a vision to make maternal care **accessible**, **AI-powered**, and **culturally relevant**, aligned with the idea of **Gram Chikitsalay** (village-level clinics).

---

## 🌟 Features

### 🩺 Module 1: Symptom Checker
- Uses LLM with structured output to detect potential health concerns based on user-described symptoms.
- Provides guidance in simple language understandable by non-technical users.

### 🥗 Module 2: Nutrition & Diet Planner
- Uses Retrieval-Augmented Generation (RAG) to answer user queries based on trusted pregnancy nutrition documents.
- Personalized diet suggestions depending on trimester, symptoms, and local food habits.

### 🌐 Language Support
- Multilingual interface (currently supports **English** and **Hindi**) to cater to rural populations.
- Users can switch between languages seamlessly.

### 🧠 Powered By
- 🔗 LangChain for modular and scalable GenAI pipelines.
- 🤖 LLMs for natural language understanding.
- ⚡ FastAPI for building a fast and interactive backend.
- 🧠 Memory & RAG for contextual, document-grounded responses.

---

## 🛠 Tech Stack

| Frontend | Backend | GenAI | Misc |
|---------|--------|--------|------|
| Streamlit | FastAPI | LangChain + LLMs | Python, OpenCV, FER, Pandas |
| HTML/CSS | REST APIs | RAG, Agents (upcoming) | English/Hindi Support |

---

## 💡 Architecture Overview

```mermaid
graph TD;
    UserInput --> StreamlitUI
    StreamlitUI --> FastAPI
    FastAPI --> LangChainPipeline
    LangChainPipeline -->|Structured Output| LLM
    LangChainPipeline -->|RAG| VectorStore
    VectorStore -->|Docs| PDF/CSV/Text
    LLM --> FastAPI --> StreamlitUI --> UserOutput
