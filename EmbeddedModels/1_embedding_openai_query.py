from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embeding-3-large', dimensions=32)

result = embedding.embed_query("What is the capital of india")

print(str(result))
