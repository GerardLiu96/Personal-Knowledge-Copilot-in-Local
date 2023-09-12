from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI, LlamaCpp
from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


embeddings_model_name = "text-embedding-ada-002"
persist_directory = "./knowledgeDB"
target_source_chunks = 2
chunk_size = 2000
model_path="./llama.cpp/models/7B/ggml-model-q8_0.bin"



if __name__ == '__main__':
    # openai_api_base = "https://xxxxx.openai.azure.com/" , openai_api_key = "xxxxxxxxxxxx", openai_api_type = "azure", deployment = "Your deployment name"
    # embeddings = OpenAIEmbeddings(openai_api_base, openai_api_key, openai_api_type, deployment, chunk_size=chunk_size)
    embeddings = LlamaCppEmbeddings(model_path=model_path, n_ctx=4096)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # llm = AzureOpenAI(openai_api_base, openai_api_key, openai_api_version="2023-06-01-preview",max_tokens=1024,deployment_name, model="gpt-35-turbo")

    prompt_template = """Based on the following known information, summarise them and answer question with only one answer.
    What is known:
    {context}
    question:
    {question}"""

    promptA = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
    model_path=model_path,
    temperature=0.2,
    max_tokens=2000,
    n_ctx=4096,
    callback_manager=callback_manager,
    verbose=True,)

    chain_type_kwargs = {"prompt": promptA}
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                     chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
    
    query = input("\nEnter your question: ")

    res = qa(query)
    answer, docs = res['result'], res['source_documents']

    print("\n> Answers:")
    print(answer)

    for document in docs:
        print("\n> " + str(document.metadata['source']) + ":")