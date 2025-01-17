# 1. tuning의 종류
## 1. Fine Tuning
1. Full Fine Tuning
2. LoRA
3. QLoRA

## 2. In-Context learning
1. Zero-shot learning
    - 예시 없이 task를 주면 바로 대답이 가능한 경우를 의미합니다.
        ```bash
        Prompt
        "이 영화는 너무 지루해" 라는 문장의 감정을 분석해줘

        GPT
        부정적인 감정입니다. 
        ```
2. One-shot learning
    - 하나의 예시를 주고 task를 수행하는 경우를 의미합니다.
        ```bash
        Prompt 
        "그 영화는 너무 지루해" -> 부정적 
        "그 영화 심심했어" -> 

        GPT
        부정적
        ```
3. Few-shot learning
    - 한개의 예시로도 출력이 시원치 않다면 몇개의 예시를 주고 task를 수행해야 합니다.
    - - 보통은 one-shot이 아니라 few-shot을 많이 사용하게 됩니다.
        ```bash
        Prompt
        "그 영화는 너무 지루해" -> 부정적 
        "그 영화 뭐 그냥 볼만해" -> 중립적
        "그 영화 정말 재미 있던데?" -> 긍정적
        "그 영화 신나" ->

        GPT
        긍정적
        ```

## 3. Prompt learning
- 이러한 In-Context learning의 특성 때문에 Prompt engineering 이라는 개념이 등장
- 이는 프롬프트를 정교하게 구성함으로써 모델이 가진 지식을 최대한 이끌어 내고, 출력 형식을 사용자의 의도에 맞게 조정하는 작업

## 4. Instruction Tuning
- Instruction Tuning은 Fine tuning과 In-Context learning을 결합해 모델의 유연성과 정확성을 향상시키기 위한 전략
- 학습 종류
    1. Alpaca
    2. KoAlpaca

# 참고자료(출처)
- [QLoRA 학습 방법](https://devocean.sk.com/experts/techBoardDetail.do?ID=165703&boardType=experts&page=&searchData=&subIndex=&idList=)
- [Instruction Tuning](https://devocean.sk.com/blog/techBoardDetail.do?ID=165806&boardType=techBlog)
- [LLM의 다양한 기법](https://ariz1623.tistory.com/348)
- [Supervised Fine Tuning](https://ariz1623.tistory.com/347)
- [Fine Tuning 이란](https://blog.naver.com/domodal/223358500997?trackingCode=rss)