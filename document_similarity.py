from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from sklearn.metrics.pairwise import cosine_similarity

loader = TextLoader('document.txt', encoding='utf-8')

documents = loader.load()

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

query = "Tell me about Virat Kohli"
doc_embedding = embedding.embed_documents(documnets)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x: x[1])[-1]

print(f'Query: {query}')
print(f'Most similar document: {documnets[index]}')
print(f'Similarity Score: {score}')
