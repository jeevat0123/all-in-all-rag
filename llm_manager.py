import os
from functools import cached_property
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# --- The Class You Defined ---
class LLMManager:
    def __init__(self, repo_id: str = "meta-llama/Llama-3.1-8B-Instruct", token: str = None):
        self.repo_id = repo_id
        # Use provided token or look in environment variables
        self.token = token or os.getenv("HF_TOKEN")

    @cached_property
    def chat_model(self) -> ChatHuggingFace:
        print(f"🧠 Connecting to LLM: {self.repo_id}")
        
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            huggingfacehub_api_token=self.token,
            task="chat-completion",
            max_new_tokens=512,
            temperature=0.2
        )
        return ChatHuggingFace(llm=llm_endpoint)

    def ask(self, prompt: str, system_prompt: str = "You are a helpful assistant."):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        return self.chat_model.invoke(messages).content

# --- 2. Isolation Test ---
if __name__ == "__main__":
    # REPLACE with your actual token or ensure HUGGINGFACEHUB_API_TOKEN is set in your terminal
    MY_TOKEN = "hf_your_actual_token_here" 

    try:
        # Initialize
        print("--- Initializing LLMManager ---")
        llm = LLMManager()

        # Test Question
        print("--- Sending Request ---")
        question = "Hai?"
        
        # This will trigger the @cached_property loading log
        answer = llm.ask(question)

        print("\n" + "="*20)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("="*20)

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")