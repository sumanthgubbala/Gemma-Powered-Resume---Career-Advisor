from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver

model = OllamaLLM(model="gemma3:1b")

template = """
You are an expert career advisor specializing in resume building and optimization.

Based on the following information, provide clear, practical suggestions and answer the question effectively.

Relevant Information:
{data}

Question:
{question}

Respond in a professional, helpful, and concise manner.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n---------------------------------------")
    question = input("Enter your question(q to quit): ")
    print("\n\n---------------------------------------")
    if question =='q':
        break
    data = retriver.invoke(question)
    result =chain.invoke({"data":data,"question":question})
    print(result)