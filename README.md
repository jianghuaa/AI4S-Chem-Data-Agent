# AI4S Chem-Data Agent 🤖🧪

An intelligent, autonomous LLM-based Agent for chemistry literature retrieval and data analysis, designed for AI for Science (AI4S) research workflows.

## 🌟 Overview
This project implements a conversational agent that bridges **local structured knowledge bases** with **real-time web search**. Instead of hallucinating, the agent acts as a rigorous "Socratic Mentor", guiding the researcher through logical deductions regarding photophysics (e.g., RTP, ISC, TADF mechanisms).

## 🚀 Key Features
- **Socratic Prompting**: Acts as a strict peer-reviewer, questioning experimental controls, mechanism origins, and data validity.
- **Dual-Database RAG**: Autonomously queries local CSV databases (`master_literature.csv` for conclusions, `rtp_records.csv` for detailed photophysical parameters).
- **Cascading Web Search**: If local data is missing, the agent triggers a Tavily web search to fetch external chemical properties (e.g., molecular weights, CAS numbers).
- **Markdown Reader**: Capable of reading and critiquing local `.md` files (e.g., literature reviews, manuscript drafts).

## 🛠️ Tech Stack
- **Language**: Python 3.11
- **LLM Engine**: Zhipu GLM-5.3-Flash (via OpenAI-compatible API)
- **Framework**: LangChain / LangChain-Classic
- **Data Handling**: Pandas
- **Search Engine**: Tavily API

## 📂 Repository Structure
- `chem_agent.py`: Main entry point for the agent loop.
- `master_literature.csv`: Structured summaries of reviewed RTP literature.
- `rtp_records.csv`: Extracted photophysical parameters (Wavelength, Lifetime, PLQY).
- `hosts.csv`: Host material properties for host-guest systems.

## 🚀 Quick Start
1. Install dependencies:
   ```bash
   pip install langchain langchain-openai langchain-classic pandas tavily-python python-dotenv