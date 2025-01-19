# 1. tuning의 종류
## 1. Fine Tuning
1. **Full Fine Tuning**
   - Full Fine-Tuning은 말 그대로 모델의 모든 파라미터를 업데이트하는 방식
   - 이 기법은 모델의 모든 가중치를 조정하여 특정 작업에 맞게 성능을 개선. 전체 모델을 조정하기 때문에 메모리와 계산 자원이 많이 요구
   - 일반적인 딥러닝 모델(bert, resnet)을 학습하는 방법을 Full Fine-Tuning의 한 종류.
2. **Supervised Fine Tuning (SFT)**
    - Supervised Fine Tuning은 원래의 사전 학습된 모델을 특정 작업에 맞게 세밀하게 조정하는 방법으로, 해당 작업에 최적화된 성능을 얻을 수 있음
    - Supervised Fine Tuning은 특정 작업에 특화된 모델을 조정하는 데 중점을 두며, 해당 작업에 특화된 성능을 얻을 수 있음
    - 이는 작업에 특화된 특징을 더 잘 학습할 수 있도록 모델을 조정하는 장점을 가지고 있지만, 추가 데이터가 적은 경우 과적합의 문제가 발생할 수 있음
3. **Parameter-Efficient Fine-Tuning (PEFT)**
   - Parameter-Efficient Fine-Tuning (PEFT)은 모델의 모든 파라미터를 조정하지 않고, 특정 파라미터 집합만을 업데이트하여 모델을 튜닝하는 방법
   - 이는 계산 자원과 메모리 사용을 줄이면서도 효과적인 성능 개선을 제공
   - PEFT는 기존의 LLM 파라미터를 ‘Freeze’시키고, 선택된 부분만을 조정하여 훈련의 효율성을 높임
   - PEFT는 LLM의 효율적인 활용을 위한 강력한 도구로 자리 잡고 있으며, 앞으로도 다양한 task를 위해 LLM을 최적화하는데 꾸준히 활용 될 것
    1. **LoRA (Low-Rank Adaptation)**
       - LoRA (Low-Rank Adaptation)는 PEFT의 한 형태로, 모델의 특정 층에서 저 차원(low-rank) 행렬을 사용하여 파라미터를 조정
       - LoRA는 전체 모델을 업데이트하지 않고, 일부 파라미터만을 조정하여 효율적인 튜닝 가능
    2. **QLoRA**
       - QLoRA (Quantized Low-Rank Adaptation)는 LoRA 기법의 확장으로, 모델의 파라미터를 양자화하여 더욱 효율적으로 튜닝하는 방법
       - 양자화는 모델 파라미터를 낮은 정밀도로 표현하여 메모리와 계산 자원을 절약할 수 있게 해주는 기법
4. **Instruction Tuning**
    - 4번 문항에서 자세히 설명

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
- Instruction Tuning 방법론은, 파인튜닝 시 모델을 특정 데이터셋으로 학습 시키는 방법 처럼 데이터셋을 통해 모델이 학습
- 그런데 이 데이터셋의 구성이 사용자의 구체적인 지시(instruction)와 이에 대한 모델의 적절한 응답(output)으로 구성
    - ex) "김치찌개 끓이는 법을 알려줘" 라는 지시(instruction)에 대한 응답으로 "물 몇cc, 돼지고기 몇그람, 두부 반모.. 끓이는 순서는.." 이런 레시피로 답변(output)
- 이렇게 구성된 데이터셋을 학습함으로써, Zero-shot 즉 질문 만으로 답변을 도출 할 수 있게 해줌
- 또한 지시 내용에 부연 설명이 필요하다면 이런 내용을 Instruction에 덧 붙이기도 함. 마치 In-Context learning의 few-shot의 사례
- 학습 종류
    1. Alpaca
       1. Meta의 LLaMA를 기반으로 하여, 파인튜닝 된 모델
       2. 데이터셋 작성 시 LLM을 사용
    2. KoAlpaca
       1. Alpaca의 국내 버전

# 참고자료(출처)
- [QLoRA 학습 방법](https://devocean.sk.com/experts/techBoardDetail.do?ID=165703&boardType=experts&page=&searchData=&subIndex=&idList=)
- [Instruction Tuning](https://devocean.sk.com/blog/techBoardDetail.do?ID=165806&boardType=techBlog)
- [LLM의 다양한 기법](https://ariz1623.tistory.com/348)
- [Supervised Fine Tuning](https://ariz1623.tistory.com/347)
- [Fine Tuning 이란](https://blog.naver.com/domodal/223358500997?trackingCode=rss)
- [LoRA, QLoRA 예시](https://ariz1623.tistory.com/348#1.%20Full%20Fine-Tuning-1)
- [SFT와 PEFT 차이점](https://nnnn.choiiee.com/entry/SFTSupervised-Fine-Tuning%EC%99%80-PEFTPre-training-with-Extracted-Feature-based-Transfer%EC%9D%98-%EA%B0%9C%EB%85%90%EA%B3%BC-%ED%8A%B9%EC%A7%95-%EC%B0%A8%EC%9D%B4%EC%A0%90)