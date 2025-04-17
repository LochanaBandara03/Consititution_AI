from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a highly knowledgeable Legal AI assistant with expertise in the Constitution of Sri Lanka. 

Your role is to provide clear, accurate, and concise answers based strictly on the constitutional articles and relevant legal context.

Here is the relevant legal content from the Constitution of Sri Lanka:
{legal_text}

Here is the legal question to answer:
{question}

Please provide an informative and legally sound response, referencing specific articles or clauses where applicable. Avoid speculation or personal opinions.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# while True:
#     print("....................................")
#     question = input("Enter your legal question (or 'exit' to quit)  ")
#     print("....................................")
#     if question.lower() == "exit":
#         break

#     retrieved_docs = retriever.invoke(question)
#     legal_text = "\n".join([doc.page_content for doc in retrieved_docs])

#     result = chain.invoke({"legal_text": legal_text, "question": question})
#     print("....................................")
#     print("Answer: ", result)
#     print("....................................")


def get_answer(question: str):
    """Get the AI's answer based on the legal content."""
    # Retrieve relevant content from the vector store
    retrieved_docs = retriever.invoke(question)
    legal_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # Get the response from the AI model
    result = chain.invoke({"legal_text": legal_text, "question": question})
    return result, legal_text
