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
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint
from typing import List
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph

llm = ChatOpenAI(model="gpt-4o-mini")
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

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
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

#def retrieve()

def routing(question):
    system = """You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
You do not need to be stringent with the keywords in the question related to these topics.
Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}"),
        ]
    )

    question_router = prompt | llm | JsonOutputParser()
    docs = retriever.get_relevant_documents(question)
    print(question_router.invoke({"question": question}))

def relevanceChecker(query, retrived_docs):
    relevant_docs = []
    system = """You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'true' or 'false' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'relevance' and no premable or explanation.
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {query}\n\n document: {document} "),
        ]
    )

    for doc in retrived_docs:
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"query": query, "document": doc.page_content})
        print(result)
        if result["relevance"]:
            relevant_docs.append(doc)

    return relevant_docs
    #print(f"{len(relevant_docs)} relevant documents.")
    #return len(relevant_docs) > 0

def search(user_input):
    response = tavily.search(query=user_input, max_results=3)
    result = [Document(page_content=obj["content"], metadata={"source": obj["url"]}) for obj in response['results']]
    print(result)
    return result

def getContext(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def getAnswer(query, context):
    prompt_answer = PromptTemplate(
        template="""Based on the following context, please answer the user's question. If you have a source in the form of a URL, include it as well.:
                    Context: {context}
                    User Question: {query}
                    Answer:""",
        input_variables=["query"],
    )

    chain_answer= prompt_answer | llm | StrOutputParser()
    result_answer = chain_answer.invoke({"query": query, "context": context})
    print(result_answer)
    return result_answer

def hallucinationChecker(answer, context):
    prompt_hallucination = PromptTemplate(
        template="""    
            Given the context below and the answer provided, determine if the answer is fully supported by the context or if there is any hallucination or unsupported information.:
            Context: {context}
            Answer: {answer}
            If the answer is fully supported by the context, respond with: {{"hallucination": "no"}}.
            If there is any hallucination or unsupported information, respond with: {{"hallucination": "yes"}}.
            Response:""",
        input_variables=["answer","context"],
    )

    chain_hallucination = prompt_hallucination | llm | JsonOutputParser()
    result_hallucination = chain_hallucination.invoke({"answer": answer, "context": context})
    print(result_hallucination)
    return result_hallucination["hallucination"] == "yes"

def main():
    user_input = input("query: ")
    state = 'relevance'
    searchCount = 0
    hallucinationCount = 0
    retrived_docs = retriever.invoke(user_input);
    relevance_docs = []
    while True:
        if state == 'relevance':
            relevance_docs = relevanceChecker(user_input, retrived_docs)
            if len(relevance_docs) > 0:
                state = 'answer'
            else:
                state = 'search'
        elif state == 'search':
            if searchCount >= 1:
                print("failed: not relevant")
            searchCount += 1
            retrived_docs = search(user_input)
            state = 'relevance'
        elif state == 'answer':
            if (hallucinationCount >= 1):
                print("failed: hallucination")
                break
            context = getContext(retrived_docs);
            answer = getAnswer(user_input, context)
            if hallucinationChecker(answer, context) == True:
                state = 'answer'
            else:
                print(answer)
                print(relevance_docs)
                break;   
        

if __name__ == "__main__":
    main()