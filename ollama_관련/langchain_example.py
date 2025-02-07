# 참고 자료 : https://wikidocs.net/233345

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = OllamaLLM(model="llama3.2")
prompt = PromptTemplate.from_template("{topic} 에 대하여 3문장으로 설명해줘.")
chain = prompt | model | StrOutputParser()

# 한 방에 출력
response = chain.invoke({"topic": "지구"})
print(response)
print()

# 챗봇처럼 실시간 출력
for token in chain.stream({"topic": "지구"}):
    print(token, end="", flush=True)