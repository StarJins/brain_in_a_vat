from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
Question: {question}

Answer: Let's think step by step.
"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.2")

chain = prompt | model

question = "What is llm?"
print(question)

response = chain.invoke({"question": question})
print()
print(response)