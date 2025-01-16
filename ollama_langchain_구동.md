# Ollama 설치 및 사용 가이드

## 1. Ollama 설치
- Ollama를 다운로드하고 설치합니다: [Ollama 다운로드 링크](https://ollama.com/)

---

## 2. LLM 모델 다운로드
1. **모델 확인**  
   - [Ollama 모델 검색](https://ollama.com/search) 페이지에서 사용할 수 있는 모델과 파라미터를 확인합니다.

2. **모델 다운로드 명령어**  
    ```bash
    ollama pull <모델명:파라미터>
    ```
    * 예시
        ```bash
        ollama pull llama3.2
        ollama pull llama3.2:1B
        ```
    * 기본적으로 `llama3.2`는 3B 크기로 설치됩니다.

---

## 3. 터미널에서 모델 실행
1. **실행 명령어**
    ```bash
    ollama run <모델명:파라미터>
    ```
    * 모델이 사전에 다운로드되지 않은 경우, `run` 명령어로 자동 다운로드 후 실행됩니다.

2. **실행 예시**
    ```bash
    D:\desktop\git\brain_in_a_vat>ollama run llama3.2
    >>> hello
    Hello! How can I assist you today?
    ```

---

# python을 통해 Ollama와 통신하기기

## 1. Python 설치
* [Python 최신 버전 다운로드](https://www.python.org/downloads/)

---

## 2. LangChain 패키지 설치
1. **설치 명령어**
    ```bash
    pip install langchain-ollama
    ```

2. **참고 문서**
* [LangChain Ollama 가이드](https://python.langchain.com/docs/integrations/providers/ollama/#ollama-tool-calling)

> **참고:**   
> Windows에서는 Ollama 앱 실행 후, 트레이 아이콘에 Ollama 아이콘이 표시되면 `serve` 상태가 활성화된 것입니다. 별도로 명령어를 실행할 필요가 없습니다.

---

## 3. python 파일 작성
1. **예제 코드 작성**
* [간단한 모델 예제](https://wikidocs.net/238532)
* [LangChain의 OllamaLLM 문서](https://python.langchain.com/docs/integrations/llms/ollama/)

2. **주의 사항**
* Python 파일 이름을 langchain.py로 설정하지 마세요.
내부 라이브러리 파일과 이름이 충돌하여 실행이 실패할 수 있습니다.
    * 관련 내용: [StackOverflow 답변](https://stackoverflow.com/questions/79274010/no-attribute-called-verbose)

---

## 4. python 코드 실행
1. **실행 조건**
* Ollama 앱이 실행 중이어야 합니다.

2. **실행 명령어**
    ```bash
    python <파일이름>.py
    ```

3. **실행 예시**
    ```bash
    D:\desktop\git\brain_in_a_vat>python ollama_llm.py
    What is llm?

    To answer your question, "What is LLm?", let's break it down step by step.

    1. **LLM** stands for Large Language Model. It's a type of artificial intelligence (AI) designed to process and understand human language.

    2. **Large**: This refers to the size or scale of the model. LLMs are trained on massive amounts of data, often in the hundreds of billions of parameters, which makes them incredibly powerful at understanding and generating text.

    3. **Language Model**: A language model is a type of machine learning model that's specifically designed to understand and generate human language. These models can be used for various tasks, such as answering questions, summarizing text, or even creating new content like stories or dialogues.

    4. **LLM** is a specific type of Large Language Model that uses transformer architecture, which is particularly effective in processing sequential data like text.

    In simple terms, an LLM is a computer program designed to understand and generate human-like language at incredible scales and speeds.
    ```


> **참고:** 다운로드되지 않은 모델을 사용하려 하면 에러가 발생합니다.

---

# Hugging Face 모델을 Ollama에서 사용하는 방법
* 추후 업데이트 예정
