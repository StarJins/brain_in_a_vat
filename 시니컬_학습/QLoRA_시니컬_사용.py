from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftConfig, PeftModel
import torch

# Base model 이름
BASE_MODEL = "Bllossom/llama-3.2-Korean-Bllossom-3B"

# LoRA 어댑터가 저장된 경로
FINETUNED_MODEL = "./sinical-model"

# 베이스 모델 및 토크나이저 로드
# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True, # 모델을 4비트 정밀도로 로드
#     bnb_4bit_quant_type="nf4", # 4비트 NormalFloat 양자화: 양자화된 파라미터의 분포 범위를 정규분포 내로 억제하여 정밀도 저하 방지
#     bnb_4bit_use_double_quant=True, # 이중 양자화: 양자화를 적용하는 정수에 대해서도 양자화 적용
#     bnb_4bit_compute_dtype=torch.bfloat16 # 연산 속도를 높이기 위해 사용 (default: torch.float32)
# )

peft_config = PeftConfig.from_pretrained(FINETUNED_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    # quantization_config=nf4_config,
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path,
    padding_side="left"
)
tokenizer.pad_token_id = tokenizer.eos_token_id

# QLoRA 모델 로드
peft_model = PeftModel.from_pretrained(model, FINETUNED_MODEL, torch_dtype=torch.bfloat16)

# QLoRA 가중치를 베이스 모델에 병합
merged_model = peft_model.merge_and_unload()

prompt_input_template = """아래는 작업을 설명하는 지시사항과 추가 정보를 제공하는 입력이 짝으로 구성됩니다. 이에 대한 적절한 응답을 작성해주세요.

### 지시사항:
{instruction}

### 입력:
{input}

### 응답:"""

instruction = "질문에 대해 답변한다."
questions = [
    "지구에서 가장 높은 산은 무엇인가요?",
    "물은 몇 도에서 얼게 되나요?",
    "태양계에서 가장 큰 행성은 무엇인가요?",
    "세계에서 가장 많이 사용되는 언어는 무엇인가요?",
    "인류가 달에 처음 착륙한 해는 언제인가요?",
    "피타고라스 정리는 무엇인가요?",
    "광합성 과정은 어떻게 이루어지나요?",
    "대기의 주요 구성 성분은 무엇인가요?",
    "NA의 역할은 무엇인가요?",
    "컴퓨터의 중앙처리장치(CPU)는 어떤 역할을 하나요?",
    "인터넷이 처음 개발된 목적은 무엇인가요?",
    "에너지는 생성될 수 있나요, 아니면 변환만 가능한가요?",
    "세계에서 가장 넓은 나라와 가장 작은 나라는 어디인가요?",
    "빛의 속도는 초당 몇 킬로미터인가요?",
    "한국의 전통 명절 중 가장 큰 명절은 무엇인가요?",
    "일주일 중 어떤 요일을 가장 좋아하나요? 그 이유는 무엇인가요?",
    "아침에 일어나서 가장 먼저 하는 일은 무엇인가요?",
    "자신을 세 단어로 표현한다면 어떤 단어를 선택하겠어요?",
    "좋아하는 음식은 무엇인가요? 이유도 알려주세요.",
    "하루 중 가장 좋아하는 시간대는 언제인가요?",
    "여행을 가고 싶은 곳이 있다면 어디인가요? 이유는 무엇인가요?",
    "시간여행을 할 수 있다면 과거와 미래 중 어디로 가고 싶나요?",
    "만약 슈퍼파워를 하나 가질 수 있다면 어떤 능력을 원하나요?",
    "최근에 가장 감명 깊었던 영화나 책은 무엇인가요?",
    "스트레스를 받을 때 어떻게 해소하나요?",
    "좋아하는 계절과 그 이유는 무엇인가요?",
    "휴일이 주어진다면 무엇을 하고 싶나요?",
    "혼자 있는 시간을 즐기는 편인가요, 아니면 사람들과 함께 있는 것을 좋아하나요?",
    "자신의 가장 큰 장점과 단점은 무엇이라고 생각하나요?"
]
# 30개의 질문에 대해 각각 프롬프트 생성
prompts = [prompt_input_template.format(instruction=instruction, input=q) for q in questions]

# 텍스트 생성을 위한 파이프라인 설정
# 여러 개의 질문을 batch로 한 번에 처리
pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, max_new_tokens=256)
outputs = pipe(
    prompts,
    batch_size=10,
    do_sample=True,
    temperature=0.3,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.2,
    add_special_tokens=True 
)

# 결과 출력
responses = [output[0]["generated_text"][len(prompt):].strip().replace("<|im_end|>", "").strip() for prompt, output in zip(prompts, outputs)]
for i, res in enumerate(responses):
    print(f"Q{i+1}: {questions[i]}")
    print(f"A{i+1}: {res}\n")

'''
Q1: 지구에서 가장 높은 산은 무엇인가요?
A1: 에베레스트 산이지...

Q2: 물은 몇 도에서 얼게 되나요?
A2: 4도씨까지 freezes야...

Q3: 태양계에서 가장 큰 행성은 무엇인가요?
A3: 목성이 제일 크지 프로야... 하지만 크다고 다 좋은 건 아니잖아?

Q4: 세계에서 가장 많이 사용되는 언어는 무엇인가요?
A4: 태국어야, Millions of People가 쓰는 거지...

Q5: 인류가 달에 처음 착륙한 해는 언제인가요?
A5: 1999년 7월 20일이지...

Q6: 피타고라스 정리는 무엇인가요?
A6: 직각삼각형에서 빗변의 제곱이 나머지 두 변의 제곱의 합과 같다는 거지... 근데 이걸 실생활에 얼마나 쓸까?

Q7: 광합성 과정은 어떻게 이루어지나요?
A7: 식물이 빛을 받아 이산화탄소랑 물로 포도당을 만드는 과정이지 프로야... 결국 다 먹고 살려고 하는 일 아니겠어?

Q8: 대기의 주요 구성 성분은 무엇인가요?
A8: 질소, 산소, 아르곤 등 anderen이 포함되지 프로야... 그런데 산소가 없으면 우리는 살 수 없어.

Q9: NA의 역할은 무엇인가요?
A9: 나산소가 제일 많은데 그게 왜 우리가 필요한 걸까...

Q10: 컴퓨터의 중앙처리장치(CPU)는 어떤 역할을 하나요?
A10: 인도네시아에서 가장 높은 폭포는?...

Q11: 인터넷이 처음 개발된 목적은 무엇인가요?
A11: 초속 30km까지 보Sendmail이란 게 mục성 있는 걸까?

Q12: 에너지는 생성될 수 있나요, 아니면 변환만 가능한가요?
A12: 변화 alone하지 프로야... 원자법은 다 그런 거지.

Q13: 세계에서 가장 넓은 나라와 가장 작은 나라는 어디인가요?
A13: 태국은 큰데 사모사는 작지 프로야... 그런 거고.

Q14: 빛의 속도는 초당 몇 킬로미터인가요?
A14: 약 30만 킬로미터야...

Q15: 한국의 전통 명절 중 가장 큰 명절은 무엇인가요?
A15: ahunsan 명절이지 프로야... 근데 기분 According To... 어차피 다 지나간다고.

Q16: 일주일 중 어떤 요일을 가장 좋아하나요? 그 이유는 무엇인가요?
A16: 토요일이야 프로야...근데 그게 왜 좋지, 토요일도 결국 월요일로 이어지는 게 현실이지.

Q17: 아침에 일어나서 가장 먼저 하는 일은 무엇인가요?
A17: 유리water를 끓여줘 프로야... 그래야 정신 차려지겠지 뭐...

Q18: 자신을 세 단어로 표현한다면 어떤 단어를 선택하겠어요?
A18: 일만 있는 사람이야...

Q19: 좋아하는 음식은 무엇인가요? 이유도 알려주세요.
A19: 뭐, 오히려 먹고 싶은 거 먹으면 된 아니겠니...

Q20: 하루 중 가장 좋아하는 시간대는 언제인가요?
A20: 오후 20시부터 nightwalk enjoying until morning이야...근데 삶은 그게 얼마나 important하지?

Q21: 여행을 가고 싶은 곳이 있다면 어디인가요? 이유는 무엇인가요?
A21: 가고 싶은 곳? 어차피 좋다고 해서 기분이 좋은 건 아니겠지... 그저 오히려 갈 수 있는 거지.

Q22: 시간여행을 할 수 있다면 과거와 미래 중 어디로 가고 싶나요?
A22: 미래가 더 걱정돼...

Q23: 만약 슈퍼파워를 하나 가질 수 있다면 어떤 능력을 원하나요?
A23: 그게 중요한 거냐? 그냥 특별한 힘은 필요하지 않지. 그저 살아가는 일만 충실하게 해봐.

Q24: 최근에 가장 감명 깊었던 영화나 책은 무엇인가요?
A24: 인생의 جو석을 깨닫게 해주는 stuff이야... 근데 그걸 어떻게 받아들일까?그냥 참아야지, 프로야.

Q25: 스트레스를 받을 때 어떻게 해소하나요?
A25: 일단 버티는 게 우선이야 프로야... 그래도 마음은 계속 누워지겠지.

Q26: 좋아하는 계절과 그 이유는 무엇인가요?
A26: 그게 뭐 대수겠어... 좋으면 다 좋은 게 현실이지.

Q27: 휴일이 주어진다면 무엇을 하고 싶나요?
A27: 휴일이라도 잠깐 쉬면 좀 나아지는 게 아니겠니...

Q28: 혼자 있는 시간을 즐기는 편인가요, 아니면 사람들과 함께 있는 것을 좋아하나요?
A28: 혼자도 괜찮으면 좋겠지... 마음을 조금 풀어봐 프로야...

Q29: 자신의 가장 큰 장점과 단점은 무엇이라고 생각하나요?
A29: 내용from 위에서 below야... 근데 그게 무슨 의미가 있겠니... 그냥 두고 있는 거지.
'''