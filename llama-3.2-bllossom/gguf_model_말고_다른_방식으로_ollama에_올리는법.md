# Model 다운로드
- 기본적으로 gguf파일이 없고, 허깅페이스를 보면 LFS라고 붙어있는 파일들이 존재
- 아래 명령어를 통해 모델을 통째로 다운로드
    ```bash
    git lfs clone https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B
    ```

# Modelfile 작성
- 모델을 다운로드받은 경로에 Modelfile 생성
    ```bash
    D:\desktop\git\brain_in_a_vat\llama-3.2-bllossom>ls -al
    ...
    drwxrwx---+ 1 bono 없음   0  2월  7일 20:16 llama-3.2-Korean-Bllossom-3B
    -rwxrwx---+ 1 bono 없음 611  2월  7일 20:16 Modelfile
    ```
- Modelfile 내용
    ```bash
    FROM ./llama-3.2-Korean-Bllossom-3B

    SYSTEM """당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다. You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner. 모든 대답은 한국어(Korean)으로 대답해주세요."""

    TEMPLATE """{{- if .System }}
    <s>{{ .System }}</s>
    {{- end }}
    <s>Human:
    {{ .Prompt }}</s>
    <s>Assistant:
    """

    PARAMETER temperature 0.6
    PARAMETER num_predict 3000
    PARAMETER num_ctx 4096
    PARAMETER stop <s>
    PARAMETER stop </s>
    PARAMETER stop <|eot_id|>
    ```

# Model 생성
- ollama create를 통해 모델 생성
    ```bash
    ollama create llama-3.2-Korean-Bllossom-3B -f ./llama-3.2-bllossom/Modelfile
    ```
- 생성된 모델 확인
    ```bash
    D:\desktop\git\brain_in_a_vat>ollama list
    NAME                                     ID              SIZE      MODIFIED      
    llama-3.2-Korean-Bllossom-3B:latest      e1f7b5ed369a    6.4 GB    3 minutes ago
    ```

# Model 테스트
- ollama run을 통해 모델 구동
    ```bash
    ollama run llama-3.2-Korean-Bllossom-3B:latest
    ```
- 응답 확인
    ```bash
    D:\desktop\git\brain_in_a_vat>ollama run llama-3.2-Korean-Bllossom-3B:latest
    >>> 안녕. 니 이름은 뭐야?
    안녕하세요! 나의 이름은 'AI 어시스턴트'입니다. 제가 도와드릴까요?
    ```