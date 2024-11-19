import getpass
import os
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = hub.pull("rlm/rag-prompt")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

query = "agent memory"
# query = "배고파"

# 연관성 체크로직
retrived_docs = retriever.invoke(query)
relevant_docs = []

parser = JsonOutputParser()
prompt = PromptTemplate(
    template="Is the context relevant to the user query?true or false in relevance variable\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

for doc in retrived_docs:
    chain = prompt | llm | parser
    result = chain.invoke({"query": query})
    print(result)
    if result["relevance"]:
        relevant_docs.append(doc)

print(f"{len(relevant_docs)} relevant documents.")


## 연관된 질문을 통한 다음 질문
query2 = "memory stream"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
relevant_context = format_docs(retrived_docs)

prompt_answer = PromptTemplate(
    template="""Based on the following context, please answer the user's question:
                Context: {context}
                User Question: {query}
                Answer:""",
    input_variables=["query"],
)

chain_answer= prompt_answer | llm | StrOutputParser()
result_answer = chain_answer.invoke({"query": query2, "context": relevant_context})

print(result_answer)

## Hallucination 체크 부터는 코드를 구조화하고 구현하면 마무리 할 수 있을듯... 시간부족... 템플릿을 잘 만들고 구조화하자.