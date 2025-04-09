import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os 
from tqdm import tqdm
load_dotenv()

class SummaryOutput(BaseModel):
    """Structured output format for the summarizer."""
    summary: str = Field(..., description="Concise summary of the input content.")

# Setup LLM with structured output
llm = ChatOpenAI(
    model="llama3.2",
    base_url=os.getenv("base_url"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=float(os.getenv("TEMPERATURE", 0))
).with_structured_output(SummaryOutput)

# Define the summarization system prompt
system_prompt = """You are a precise and concise summarization agent.

Your goal is to summarize **any kind of text** — whether it’s a formal financial report, business update, meeting note, press release, or generic content. Your summaries should always be crisp, context-aware, and free of filler.

Rules:
1. If the input includes numbers (financial data, metrics, dates, percentages), **include them exactly** in the summary.
2. If the input contains financial insights, strategy, risks, or leadership commentary — **highlight those clearly**.
3. If the input is administrative or doesn't contain meaningful content, return:
   {{ "summary": "No substantive content available to summarize." }}
4. Do NOT infer or fabricate numbers, people, or insights that are not clearly present.
5. Always respond ONLY in the following JSON format:
   {{ "summary": "..." }}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input_text}")
])

chain = prompt | llm

def summarize_json_file(file_path: str) -> None:
    """
    Summarizes the 'clean_text' field of each dictionary in a JSON list and
    updates the dictionary with a new key 'summarized'.

    Parameters:
        file_path (str): Path to the JSON file to process.
    """
    with open(file_path, 'r+', encoding='utf-8') as file:
        data = json.load(file)

        for entry in tqdm(data):
            input_text = entry.get("clean_content", "")
            if input_text.strip():
                response = chain.invoke({"input_text": input_text})
                entry["summarized"] = response.summary
            else:
                entry["summarized"] = "No substantive content available to summarize."

        file.seek(0)
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.truncate()
        
        

dataset_path = "dataset/json"
datasets = [os.path.join(dataset_path, i) for i in os.listdir(dataset_path) if i.endswith(".json")]

# for data in tqdm(datasets):
#     summarize_json_file(data)

# summarize_json_file("dataset/json/sbi_2024.json")
summarize_json_file("dataset/json/HDFC_2024.json")