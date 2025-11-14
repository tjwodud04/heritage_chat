from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import (
    TextLoader,
    RecursiveUrlLoader,
    PlaywrightURLLoader,
    SeleniumURLLoader,
)
import glob
import os
from os import PathLike
from pathlib import Path
from typing import Union, List, Optional
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone as PineconeClient

# __import__("pysqlite3")
#
# import sys
#
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# 참고
# https://python.langchain.com/docs/use_cases/question_answering/how_to/question_answering


def get_chroma_retriever(
    embedding_model: Embeddings,
    root: Union[str, bytes, os.PathLike] = "./서울시 유적지 현황 (영어).json",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    txt_spliter = TokenTextSplitter(
        model_name="text-embedding-ada-002",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # embedding_model = OpenAIEmbeddings()
    # vector_db = Chroma(embedding_function=embedding_model)
    # txts = sorted(glob.glob(root))
    splitted_txts: List[Document] = list()
    # for txt_path in txts:
    # raw_doc = TextLoader(txt_path).load()
    raw_doc = TextLoader(root, encoding="utf-8").load()

    splited_docs = txt_spliter.split_documents(raw_doc)
    splitted_txts.extend(splited_docs)
        # vector_db.add_documents(splited_docs)
    chroma_db = Chroma.from_documents(
        documents=splitted_txts,
        embedding=embedding_model,
        # persist_directory="law_data"
        # metadatas=[{"source": str(i)} for i in range(len(splitted_txts))],
    ).as_retriever()
    return chroma_db


def get_pinecone_retriever(
    embedding_model: Embeddings,
    api_key: Optional[str] = None,
    environment: Optional[str] = None,
    index: Optional[str] = None,
):
    pc = PineconeClient(api_key=api_key or os.getenv("PINECONE_API_KEY"))
    index_name = index or os.getenv("PINECONE_INDEX")
    vector_db = PineconeVectorStore(
        index=pc.Index(index_name),
        embedding=embedding_model,
        text_key="text",
    )
    return vector_db.as_retriever()


def get_faiss_ensemble_retriever(
    embedding_model: Embeddings,
    index_path: Union[str, PathLike, Path] = "./heritage_vectorized_result/",
):
    faiss_db = FAISS.load_local(folder_path=index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return faiss_db.as_retriever()


if __name__ == "__main__":
    query = "Is there any information of heritage that is located on Seoul?"
    vector_db_chain = get_faiss_ensemble_retriever(OpenAIEmbeddings())

    related_laws = vector_db_chain.invoke(query)
    print(related_laws)
