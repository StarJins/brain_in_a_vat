from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftConfig, PeftModel
import torch

# Base model 이름
BASE_MODEL = "Bllossom/llama-3.2-Korean-Bllossom-3B"

# LoRA 어댑터가 저장된 경로
FINETUNED_MODEL = "./outputs/checkpoint-50"

# # 토크나이저 로드
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# # NF4 양자화를 위한 설정
# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True, # 모델을 4비트 정밀도로 로드
#     bnb_4bit_quant_type="nf4", # 4비트 NormalFloat 양자화: 양자화된 파라미터의 분포 범위를 정규분포 내로 억제하여 정밀도 저하 방지
#     bnb_4bit_use_double_quant=True, # 이중 양자화: 양자화를 적용하는 정수에 대해서도 양자화 적용
#     bnb_4bit_compute_dtype=torch.bfloat16 # 연산 속도를 높이기 위해 사용 (default: torch.float32)
# )

# # Base model을 8-bit 모드로 불러와서 GPU에 할당 (메모리 절약을 원할 때)
# model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     quantization_config=nf4_config,
#     device_map="auto"
# )

# # 학습된 LoRA 어댑터(QLoRA) 로드
# model = PeftModel.from_pretrained(model, FINETUNED_MODEL)

# # 예시 질문
# prompt = "질문: 지금 한국의 대통령이 누군지 설명해줘\n답변:"

# # 입력 텍스트를 토크나이즈하고, 모델이 위치한 디바이스로 이동
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# # 모델 추론 (max_new_tokens, temperature 등 하이퍼파라미터는 상황에 맞게 조정)
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=500,  # 생성할 최대 토큰 수
#         temperature=0.7      # 생성 다양성 조절 (옵션)
#     )

# # 생성된 응답 디코딩 및 출력
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)

# 베이스 모델 및 토크나이저 로드
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, # 모델을 4비트 정밀도로 로드
    bnb_4bit_quant_type="nf4", # 4비트 NormalFloat 양자화: 양자화된 파라미터의 분포 범위를 정규분포 내로 억제하여 정밀도 저하 방지
    bnb_4bit_use_double_quant=True, # 이중 양자화: 양자화를 적용하는 정수에 대해서도 양자화 적용
    bnb_4bit_compute_dtype=torch.bfloat16 # 연산 속도를 높이기 위해 사용 (default: torch.float32)
)

peft_config = PeftConfig.from_pretrained(FINETUNED_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=nf4_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path
)

# QLoRA 모델 로드
peft_model = PeftModel.from_pretrained(model, FINETUNED_MODEL, torch_dtype=torch.bfloat16)

# QLoRA 가중치를 베이스 모델에 병합
merged_model = peft_model.merge_and_unload()

prompt = "지금 한국의 대통령이 누군지 설명해줘"

# 텍스트 생성을 위한 파이프라인 설정
pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, max_new_tokens=256)
outputs = pipe(
    prompt,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.2,
    add_special_tokens=True 
)
print(outputs[0]["generated_text"][len(prompt):])
