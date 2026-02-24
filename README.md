# Generative AI Assignment — Task 1 & Task 2

---

## Task 1: Streamlit Interface for Local LLM (`task1_ollama_streamlit_app.py`)

### Setup
```bash
# 1. Install Ollama (one-time)
#    Visit: https://ollama.com/download

# 2. Pull a model
ollama pull llama3           # Recommended
# or: ollama pull deepseek-r1
# or: ollama pull mistral

# 3. Start Ollama server (runs in background)
ollama serve

# 4. Install Python dependencies
pip install streamlit requests

# 5. Run the app
streamlit run task1_ollama_streamlit_app.py
```

### Features Implemented
- Text input box for user queries
- Response display area with streaming (blinking cursor)
- Conversation history panel (persists across messages)
- Reset button to clear conversation
- Model selector (auto-detects all locally pulled models)
- System prompt customisation
- Temperature slider
- Session statistics (message count)

---

## Task 2: Medical Fine-Tuning with QLoRA (`task2_medical_qlora_unsloth.ipynb`)

### Setup — Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `task2_medical_qlora_unsloth.ipynb`
3. Set runtime: **Runtime → Change runtime type → T4 GPU**
4. Run all cells top-to-bottom

### What the Notebook Does (12 Steps)
| Step | Action |
|------|--------|
| 1 | Install Unsloth, TRL, PEFT, bitsandbytes |
| 2 | Verify T4 GPU |
| 3 | Load Llama-3.1-8B in 4-bit quantisation |
| 4 | Attach LoRA adapter (r=16) |
| 5 | Load medical Q&A dataset, format as Alpaca prompts |
| 6 | Configure SFTTrainer (8-bit AdamW, 3 epochs) |
| 7 | Monitor GPU VRAM |
| 8 | Train the model |
| 9 | Save LoRA adapter |
| 10 | Test on 5 new medical queries |
| 11 | (Optional) Save to Google Drive |
| 12 | (Optional) Push to HuggingFace Hub |

### Dataset Used
`medalpaca/medical_meadow_medical_flashcards` — 33k clinical Q&A pairs from HuggingFace Hub

### Submission
- ZIP/RAR both files OR share a GitHub repository link
- Email to: `submissions.archtech@gmail.com` before the 27th of this month
- For technical queries: `queries.archtech@gmail.com`
