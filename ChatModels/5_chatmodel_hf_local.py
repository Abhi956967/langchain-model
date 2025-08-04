# Import required LangChain Hugging Face wrappers
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Import the OS module to set environment variables
import os 

# Set the Hugging Face cache directory to a custom location (optional).
# This helps avoid re-downloading models repeatedly and controls where the model files are stored.
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Initialize a Hugging Face pipeline wrapped in a LangChain-compatible interface.
# HuggingFacePipeline allows using pre-trained models directly in LangChain with `pipeline()` under the hood.
llm = HuggingFacePipeline(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # The model to use from Hugging Face (a small open-weight chat model)
    
    task="text-generation",  # ⚠️ Deprecated in latest LangChain — better to remove; older versions accept this
    
    pipeline_kwargs=dict(   # Extra configuration for the underlying Hugging Face `pipeline()`
        temperature=0.5,     # Controls randomness; 0.5 = balanced (not too random, not too deterministic)
        max_new_tokens=100  # Maximum number of tokens to generate in the response
    )
)

# Wrap the base LLM pipeline (`llm`) into a ChatModel interface
# This enables handling multi-turn messages, formatting prompts like a chatbot, etc.
model = ChatHuggingFace(llm=llm)

# Invoke the model with a prompt. LangChain automatically formats it for a chat-style model (if supported).
result = model.invoke("What is the capital of India")

# Extract and print the content (text) from the model's response object.
print(result.content)


