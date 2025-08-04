# Import the ChatOpenAI class from the langchain_openai package
from langchain_openai import ChatOpenAI

# Import the load_dotenv function to load environment variables from a .env file
from dotenv import load_dotenv 

# Load environment variables from the .env file (e.g., OpenAI API key)
load_dotenv()

# Initialize the ChatOpenAI model with specified parameters:
# - model: 'gpt-4' selects the GPT-4 model
# - temperature: 1.5 increases randomness/creativity of the response
# - max_completion_tokens: limits the model to return at most 10 tokens in the response
model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

# Call the model with a prompt and store the response in the 'result' variable
result = model.invoke("What is the capital of France?")

# Print the content (i.e., the actual text response) from the result
print(result.content)