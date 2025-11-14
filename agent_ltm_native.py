import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

class LTM:
    def __init__(self, path="ltm_store"):
        self.path = path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.store = self._load() if Path(path).exists() else FAISS.from_texts(["Memory initialized"], self.embeddings)
    
    def _load(self):
        print(f"[LTM-LOAD] Reading from disk: {self.path}")
        return FAISS.load_local(self.path, self.embeddings, allow_dangerous_deserialization=True)
    
    def store_memory(self, text):
        print(f"[LTM-STORE] Adding: {text}")
        self.store.add_texts([text])
        self.store.save_local(self.path)
    
    def retrieve_memory(self, query, k=2):
        print(f"[LTM-SEARCH] Query: {query}")
        results = self.store.similarity_search(query, k=k)
        memories = [r.page_content for r in results]
        for i, mem in enumerate(memories, 1):
            print(f"  Match {i}: {mem}")
        return memories


def workflow_demo():
    ltm = LTM()
    
    print("Memory storage using facts (example):\n")
    ltm.store_memory("Project Nexus launches in Q2 2026 with quantum encryption.")
    ltm.store_memory("Team lead is Dr. Sarah, specializing in cryptography & Advance AI Techniques.")
    ltm.store_memory("Budget approved: $2.7M for initial development phases")
    ltm.store_memory("Primary competitor is DataHSA COrp. with classical methods.")
    
    print("\n Questions:")
    ltm.retrieve_memory("Who is leading the project?")
    print()
    ltm.retrieve_memory("What is the budget and timeline?")
    print()
    ltm.retrieve_memory("Tell me about competitors")
    
if __name__ == "__main__":
    workflow_demo()