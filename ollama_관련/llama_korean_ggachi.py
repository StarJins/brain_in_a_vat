from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="llama3.2_1B_korean_ggachi_q4_k")

question = "LLM에 대해 단계별로 설명해 줄래?"
print(question)

response = model.invoke(question)
print()
print(response)