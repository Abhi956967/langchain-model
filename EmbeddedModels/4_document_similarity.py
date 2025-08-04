# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

# documents = [
#     "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
#     "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
#     "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
#     "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
#     "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
# ]

# query = 'tell me about bumrah'

# doc_embeddings = embedding.embed_documents(documents)
# query_embedding = embedding.embed_query(query)

# scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

# print(query)
# print(documents[index])
# print("similarity score is:", score)


# -----------------------------------------------------------------------------------------------------------



from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# (Optional) Set HuggingFace cache directory if needed
os.environ['HF_HOME'] = 'D:/huggingface_cache'  # or any preferred cache path

# Initialize Hugging Face embedding model (downloads automatically if not present)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Sample documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Query
query = 'tell me about bumrah'

# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)  # List of vectors
query_embedding = embedding.embed_query(query)         # Single vector

# Compute cosine similarity between query and each document
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get the most similar document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Output results
print(query)
print(documents[index])
print("Similarity score is:", score)
