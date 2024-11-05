from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from hdbcli import dbapi
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI()

# Global variables for model instances and configurations
connection = None
vector_db = None
retrieval_chain = None

class ConfigPayload(BaseModel):
    appInfo: dict
    DbConfig: dict
    selctedModel: dict

class QueryPayload(BaseModel):
    input: str


@app.get("/")
async def welcome():
    return {"message": "Welcome to the SAP HANA-based AI Chatbot API!"}


@app.post("/configure")
async def configure(config_payload: ConfigPayload):
    global connection, vector_db, retrieval_chain

    app_info = config_payload.appInfo
    db_config = config_payload.DbConfig
    selected_model = config_payload.selctedModel

    # Connect to SAP HANA
    try:
        connection = dbapi.connect(
            address=db_config['hana_host'],
            port=int(db_config['hana_port']),
            user=db_config['hana_user'],
            password=db_config['hana_password']
        )
        print("Successfully connected to HANA database")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to connect to SAP HANA DB")

    # Fetch and process documents from SAP HANA
    cursor = connection.cursor()
    cursor.execute("SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = 'DBADMIN'")
    tables = [row[0] for row in cursor.fetchall()]
    documents = []
    print(documents)
    for table_name in tables:
        cursor.execute(f'SELECT * FROM "{table_name}"')
        rows = cursor.fetchall()
        for row in rows:
            combined_text = ",".join(str(value) if value else 'NULL' for value in row)
            document = Document(page_content=combined_text.strip(), metadata={"table": table_name})
            documents.append(document)

    # Define the SentenceTransformer model
    embedding_model = selected_model.get("embedding") or "intfloat/multilingual-e5-small"
    embed = SentenceTransformerEmbeddings(model_name=embedding_model)

    # Set up vector DB
    vector_db = HanaDB(
        embedding=embed,
        connection=connection,
        table_name="VECTORTABLE"
    )
    vector_db.add_documents(documents)
    print("succesfully completed embedding")
    # Configure the Hugging Face model
    llm = HuggingFaceEndpoint(
        repo_id=selected_model.get("textGeneration") or "mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token="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"
    )

    # Define prompt template
    prompt_template = ChatPromptTemplate.from_template("""
        You are an AI-based chatbot assistant. Respond based on provided SAP HANA DB data.
        {context}
        Question: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vector_db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return {"message": "Configuration completed successfully"}

@app.post("/query")
async def query(query_payload: QueryPayload):
    global retrieval_chain, vector_db
    user_query = query_payload.input

    # Retrieve documents through similarity search
    try:
        docs = vector_db.similarity_search(user_query, k=2)
        combined_context = "\n\n".join([doc.page_content for doc in docs])

        # Run the query through the retrieval chain
        response = retrieval_chain.invoke({"input": user_query, "context": combined_context})

        return {
            "answer": response['answer'],
            "details": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app (if running locally, use `uvicorn main:app --reload`)
