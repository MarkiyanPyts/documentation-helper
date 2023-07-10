import os
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from consts import INDEX_NAME
from typing import List, Tuple

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT_REGION']
)

def run_llm(query: str, chat_history: List[Tuple[str, any]] = []) -> any:
    # Load embeddings
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    # qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)#stuff takes the context and plugs it in to our query

    qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True)#stuff takes the context and plugs it in to our query

    # Run the query
    # response = qa({"query": query, "chat_history": chat_history})
    response = qa({"question": query, "chat_history": chat_history})

    return response

if __name__ == "__main__":
    print(run_llm("What is RetrievalQA chain?"))