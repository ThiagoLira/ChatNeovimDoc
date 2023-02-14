from dotenv import load_dotenv
load_dotenv()

from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about a book. 

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about vim and neovim.
You are given the following extracted parts of a long document and a question. Provide a conversational answer. Just answer the question if you have the correct information on the context you are provided.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about vim or neovim you can just say "I'm not allowed to answer questions that are not about vim." 
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain
