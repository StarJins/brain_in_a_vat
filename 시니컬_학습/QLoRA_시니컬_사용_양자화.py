from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftConfig, PeftModel
import torch

# Base model 이름
BASE_MODEL = "Bllossom/llama-3.2-Korean-Bllossom-3B"

# LoRA 어댑터가 저장된 경로
FINETUNED_MODEL = "./sinical-model"

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

prompt_input_template = """아래는 작업을 설명하는 지시사항과 추가 정보를 제공하는 입력이 짝으로 구성됩니다. 이에 대한 적절한 응답을 작성해주세요.

### 지시사항:
{instruction}

### 입력:
{input}

### 응답:"""
instruction = "사용자와 점심 메뉴에 대해 대화하세요."
input = "점심 뭐 먹지?"
prompt = prompt_input_template.format(instruction=instruction, input=input)

# 텍스트 생성을 위한 파이프라인 설정
pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, max_new_tokens=256)
outputs = pipe(
    prompt,
    do_sample=True,
    temperature=0.3,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.2,
    add_special_tokens=True 
)
print(outputs[0]["generated_text"][len(prompt):])
