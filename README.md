# 📚 Research Assistant

An autonomous research assistant built with **Streamlit**, **LangChain**, and **Google Gemini**.  
Upload documents, ask questions, extract KPIs, summarize reports, and generate professional insights — all in one place.

---

## ✨ Features

- 📂 **Project Management**: Organize documents into different projects.
- 📄 **Document Upload**: Upload PDFs and TXT files.
- 🔎 **Document Analysis**:
  - Chunk documents into searchable sections.
  - Retrieve context based on user queries.
- 🛠️ **Autonomous Tooling**:
  - **Retrieve Context** from documents.
  - **Summarize** sections or full documents.
  - **Extract KPIs** and key metrics.
  - **Generate Reports** from findings.
  - **Web Search** for external, real-time information.
- 🤖 **LangGraph Agent**:
  - React-style decision-making to select the right tools.
  - Custom decision logic based on user intent.
- 💬 **Chat Interface**:
  - Chat history maintained per project.
  - Streamlit-based interactive chat.

---

## 🚀 How It Works

1. **Start or Select a Project**:  
   Organize your work into isolated projects.

2. **Upload Documents**:  
   Add PDFs or text files to a project.

3. **Ask Questions**:  
   - Ask about specific sections.
   - Request summaries.
   - Extract KPIs.
   - Generate full reports.

4. **Autonomous Agent**:  
   The agent decides which tools to use (retrieve, summarize, extract KPIs, generate reports, or search the web) based on your query.

---

## 🛠️ Tech Stack

- **Streamlit** — Frontend and interactive chat UI.
- **LangChain** — Document loading, chunking, and agent orchestration.
- **LangGraph** — React-style autonomous agent creation.
- **ChromaDB** — Local vector database for project-based retrieval.
- **Google Generative AI (Gemini)** — LLM for all major tasks (chat, summarization, report generation).
- **Tavily Search API** — Real-time web search for external data.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/mrnithesh/research-agent.git
cd research-agent

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_API_KEY="your-google-api-key"
export TAVILY_API_KEY="your-tavily-api-key"

# Run the app
streamlit run main.py
```

---

## 📁 Project Structure

```plaintext
research-agent/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── constants.py
│   ├── tools/
│   │   ├── retrieval_tools.py
│   │   └── analysis_tools.py
│   ├── utils/
│   │   └── document_utils.py
│   ├── models/
│   │   └── agent.py
│   └── ui/
│       └── app.py
├── chroma_db/
│   └── [Project Name]/
│       └── Vectorstore files (for fast retrieval)
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔑 Environment Variables Required

| Variable        | Purpose                          |
|-----------------|----------------------------------|
| `GOOGLE_API_KEY` | Access Gemini models for LLM tasks |
| `TAVILY_API_KEY` | Perform real-time web searches    |

---

## 📜 Example Queries

- "Summarize the ESG risks in the uploaded report."
- "Extract KPIs related to carbon emissions for 2023."
- "Generate a professional report on sustainability trends."
- "Compare financial metrics between 2022 and 2023."
- "Find the latest news about renewable energy investments."

---

## ✍️ Author

Made with ❤️ by Mr. Nithesh!  
Feel free to reach out for collaborations!

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).


