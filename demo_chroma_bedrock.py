import json
import boto3
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb

os.environ['AWS_DEFAULT_REGION'] = "your region"
os.environ['AWS_PROFILE']="your profile"

# Embedding Model 

class TitanEmbeddings(object):
    accept = "application/json"
    content_type = "application/json"
    def __init__(self, model_id):
        self.bedrock = boto3.client(service_name='your service')
        self.model_id = model_id
    def __call__(self, text):
        """
        Returns Embeddings
        Args:
            text (str): text to embed
            dimensions (int): Number of output dimensions.
            normalize (bool): Whether to return the normalized embedding or not.
        Return:
            List[float]: Embedding
            
        """
        body = json.dumps({
            "inputText": text,
            "dimensions": 256,
            "normalize": True
        })
        response = self.bedrock.invoke_model(
            modelId=self.model_id,body=body,accept=self.accept, contentType=self.content_type
        )
        response_body = json.loads(response.get('body').read())
        return response_body['embedding']

if __name__ == '__main__':
    """
    Entrypoint for Embedding Model.
    """
    
    titan_embeddings_v2 = TitanEmbeddings(model_id="your model")

    #Preparing the data
    loader = DirectoryLoader("new_articles/", glob = "./*.txt", loader_cls= TextLoader)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 35, chunk_overlap = 10)
    texts = text_splitter.split_documents(document)
    # Connect to ChromaDB
    # Initialize ChromaDB client
    client = chromadb.Client()


    # # Create or get a collection
    collection_name = "embeddings_collection"
    collection = client.get_or_create_collection(name=collection_name)

    print("Creating vector Database")

    pos=0
    for doc in texts:
        embedding = titan_embeddings_v2(doc.page_content)
        metadata = doc.metadata
        
    #     # Store the embedding with associated metadata
        collection.add(
            embeddings=[embedding],  # List of embeddings
            metadatas=[metadata],    # List of metadata dicts
            ids=[f"doc_{pos}"]  # Unique IDs for each embedding, make sure to generate unique IDs for each document
        )
        pos+=1
    print("Vector db creation Done!\n")
    
    print("Querying...\n")
    print("Type your query...")
    query_text=input()
    collection_name = "embeddings_collection"
    collection = client.get_collection(name=collection_name)

    query_embedding = titan_embeddings_v2(query_text)


    # # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],  # Embedding to query
        n_results=2                          # Number of similar results to retrieve
    )

    # # Display results
    print(results)
    for result in results:
        print(result)

