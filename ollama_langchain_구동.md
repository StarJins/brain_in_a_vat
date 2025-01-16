# Ollama 설치 및 사용 가이드

## 1. Ollama 설치
- Ollama를 다운로드하고 설치합니다: [Ollama 다운로드 링크](https://ollama.com/)

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

## 2. LangChain 패키지 설치
1. **설치 명령어**
    ```bash
    pip install langchain-ollama
    ```

2. **참고 문서**
* [LangChain Ollama 가이드](https://python.langchain.com/docs/integrations/providers/ollama/#ollama-tool-calling)

> **참고:**   
> Windows에서는 Ollama 앱 실행 후, 트레이 아이콘에 Ollama 아이콘이 표시되면 `serve` 상태가 활성화된 것입니다. 별도로 명령어를 실행할 필요가 없습니다.

## 3. python 파일 작성
1. **예제 코드 작성**
* [간단한 모델 예제](https://wikidocs.net/238532)
* [LangChain의 OllamaLLM 문서](https://python.langchain.com/docs/integrations/llms/ollama/)

2. **주의 사항**
* Python 파일 이름을 langchain.py로 설정하지 마세요.
내부 라이브러리 파일과 이름이 충돌하여 실행이 실패할 수 있습니다.
    * 관련 내용: [StackOverflow 답변](https://stackoverflow.com/questions/79274010/no-attribute-called-verbose)

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
## 1. Huging Face에서 한국어 적용 모델 다운로드
1. **한국어 적용 모델 검색**
* `llama3.2 korean` 등의 키워드를 통해 한국어 모델 검색
    * 예시
    [llama3.2 한국어 적용 모델 1](https://huggingface.co/RichardErkhov/June-J_-_Llama-3.2-1B_korean_1930_novel-gguf)
    [llama3.2 한국어 적용 모델 2](https://huggingface.co/Tedhong/Llama-3.2-Korean-GGACHI-1B-Instruct-v1-Q4_K_M-GGUF)
    > **참고:**   
    > 모델에 q0, q1 등의 표시가 있는데 이는 양자화(Quantization) 표시이다.
    > 자세한 내용은 [블로그 참조](https://dytis.tistory.com/72)   

2. **GGUF 파일 다운로드**
Ollama에 적용하기 위해서는 GGUF 파일을 다운로드 해야 한다.

## 2. Ollama에 모델 설치 방법
1. **GGUF 파일 저장 폴더 생성**
* GGUF 파일을 보관할 폴더를 생성
* 예시:
    * 모델명 : `Llama-3.2-1B_korean_1930_novel.Q4_1.gguf`
    * path : `D:\desktop\git\brain_in_a_vat\Llama-3.2-1B_korean_1930_novel.Q4_1`

2. **Modelfile 작성**
* GGUF 파일이 저장되어 있는 폴더에 Modelfile 생성(GGUF 모델과 같은 path에 있어야 한다)
* 아래 내용을 입력
    > 이때 FROM 에는 GGUF 모델명을 입력
    ```bash
    FROM Llama-3.2-1B_korean_1930_novel.Q4_1.gguf

    TEMPLATE """[INST] {{ if and .First .System }}<<SYS>>{{ .System }}<</SYS>>

    {{ end }}{{ .Prompt }} [/INST] """
    SYSTEM """"""
    PARAMETER stop [INST]
    PARAMETER stop [/INST]
    PARAMETER stop <<SYS>>
    PARAMETER stop <</SYS>>
    ```
    > 각 키워드에 대한 설명은 추후 설명

3. **Ollama에 해당 모델 설치**
* `ollama create` 명령어를 통해 모델을 설치
    ```bash
    D:\>cd D:\desktop\git\brain_in_a_vat

    D:\desktop\git\brain_in_a_vat>ollama create llama3.2_1B_korean_novel_q4_1 -f ./Llama-3.2-1B_korean_1930_novel.Q4_1/Modelfile
    gathering model components
    copying file sha256:8007280587ed4edb3660f3a47f34fa17e180702f23923b0ef8d1dcfa91e62003 100%
    parsing GGUF
    using existing layer sha256:8007280587ed4edb3660f3a47f34fa17e180702f23923b0ef8d1dcfa91e62003
    using existing layer sha256:ab982f1f28716f4591b9fa783658f500049f4f1e933e38cde531780aec35fe43
    using existing layer sha256:fa304d6750612c207b8705aca35391761f29492534e90b30575e4980d6ca82f6
    writing manifest
    success

    D:\desktop\git\brain_in_a_vat>ollama list
    NAME                                     ID              SIZE      MODIFIED
    llama3.2_1B_korean_novel_q4_1:latest     d386b927027e    869 MB    2 seconds ago
    llama3.2:latest                          a80c4f17acd5    2.0 GB    3 hours ago
    llama3.2:3B                              a80c4f17acd5    2.0 GB    3 hours ago
    ```

4. **langchain으로 해당 모델 사용**
* `python을 통해 Ollama와 통신하기기`를 참고하여 코드 작성 및 실행