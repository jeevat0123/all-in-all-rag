import os
from functools import cached_property
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

import os
from functools import cached_property
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

class LLMManager:
    """
    Manages the connection and interaction with Hugging Face Large Language Models.
    Uses lazy initialization to ensure the API connection is only established when needed.
    """

    def __init__(self, repo_id: str = "meta-llama/Llama-3.1-8B-Instruct", token: str = None):
        """
        Initializes the manager with a specific model and authentication token.
        
        Args:
            repo_id (str): The Hugging Face model repository ID.
            token (str, optional): HF API token. Defaults to 'HF_TOKEN' environment variable.
        """
        self.repo_id = repo_id
        # Fallback logic for authentication: Priority given to passed token, then env var
        self.token = token or os.getenv("HF_TOKEN")

    @cached_property
    def chat_model(self) -> ChatHuggingFace:
        """
        Lazily creates and caches the ChatHuggingFace instance.
        The @cached_property decorator ensures the 'Connecting' logic runs ONLY ONCE,
        storing the resulting object for all future calls to self.chat_model.
        """
        print(f"🧠 Connecting to LLM: {self.repo_id}")
        
        # Configure the remote inference endpoint
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            huggingfacehub_api_token=self.token,
            task="chat-completion", # Ensures the provider uses a chat-optimized pipeline
            max_new_tokens=512,     # Limits the length of the generated response
            temperature=0.2         # Lower temperature (0.2) makes responses more deterministic/focused
        )
        
        # Wrap the endpoint in ChatHuggingFace to automatically handle 
        # model-specific templates (like Llama-3's <|begin_of_text|> tags)
        return ChatHuggingFace(llm=llm_endpoint)

    def ask(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Sends a message to the LLM and returns the text response.
        
        Args:
            prompt (str): The user's question or instruction.
            system_prompt (str): Instructions defining the AI's behavior/persona.
            
        Returns:
            str: The content of the AI's response.
        """
        # Structured as a list of messages to satisfy the Chat completion format
        messages = [
            SystemMessage(content=system_prompt), # Sets the 'personality' or rules
            HumanMessage(content=prompt)         # The actual user input
        ]
        
        # Invoke the model and extract just the text content from the response object
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