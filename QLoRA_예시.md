### QLoRA 란
- PEFT의 한 종류
- LoRA에 4비트 양자화를 적용하여 더욱 경량화한 방법

### QLoRA 학습 전 테스트
# 1. 모듈 다운로드 및 import
```bash
!pip install -q -U datasets
!pip install -q -U bitsandbytes
!pip install -q -U accelerate
!pip install -q -U peft
!pip install -q -U trl
```
```python
import os
import torch
import transformers
from datasets import load_from_disk
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TextStreamer,
    pipeline
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    TaskType,
    PeftModel
)
from trl import SFTTrainer
```
# 2. 모델 및 토크나이저 불러오기
- 모델은 우선 llama3.2를 기준으로 설명
    ```python
    BASE_MODEL = "llama3.2"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ```

# 3. 모델 테스트
```python
prompt = "한국의 아이돌 문화에 대해 알려줘."

# 텍스트 생성을 위한 파이프라인 설정
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256) # max_new_tokens: 생성할 최대 토큰 수
outputs = pipe(
    prompt,
    do_sample=True,         # 샘플링 전략 사용. 확률 분포를 기반으로 다음 토큰을 선택
    temperature=0.2,        # 샘플링의 다양성을 조절하는 파라미터. 값이 높을수록 랜덤성 증가
    top_k=50,               # 다음 토큰을 선택할 때 상위 k개의 후보 토큰 중에서 선택. 여기에서는 상위 50개의 후보 토큰 중에서 샘플링
    top_p=0.95,             # 누적 확률이 p가 될 때까지 후보 토큰을 포함
    repetition_penalty=1.2, # 반복 패널티를 적용하여 같은 단어나 구절이 반복되는 것 방지
    add_special_tokens=True # 모델이 입력 프롬프트의 시작과 끝을 명확히 인식할 수 있도록 특별 토큰 추가
)
print(outputs[0]["generated_text"][len(prompt):]) # 입력 프롬프트 이후에 생성된 텍스트만 출력
```

### 학습 데이터셋 준비
# 1. 모듈 불러오기
```python
import os
import torch
import transformers
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
```

# 2. KoAlpaca 데이터셋 불러오기
- KoAlpaca 데이터셋은 지식인 기반 질문-답변 데이터셋
    ```python
    dataset_koalpaca = load_dataset("beomi/KoAlpaca-v1.1a")
    dataset_koalpaca
    ```
- 샘플 출력해보기
    ```python
    print(dataset_koalpaca['train'][:2])
    ```
- 데이터 형태 변환 및 저장
    ```python
    # 데이터프레임으로 변환
    df_koalpaca = pd.DataFrame(dataset_koalpaca['train'])

    # 중복 제거
    df_koalpaca = df_koalpaca.drop_duplicates(keep='first', ignore_index=True)

    # HuggingFace Dataset 형태로 변환
    dataset_koalpaca = Dataset.from_pandas(df_koalpaca)

    # 로컬에 데이터셋 저장
    dataset_koalpaca.save_to_disk("datasets/koalpaca_dataset")
    ```

### QLoRA로 파인튜닝 시작하기
# 1. 데이터셋 및 모델 불러오기
```python
# 데이터셋 로드
dataset_path = "datasets/koalpaca_dataset"
dataset = load_from_disk(dataset_path)

# 모델 불러오기
# NF4 양자화를 위한 설정
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 모델을 4비트 정밀도로 로드
    bnb_4bit_quant_type="nf4",              # 4비트 NormalFloat 양자화: 양자화된 파라미터의 분포 범위를 정규분포 내로 억제하여 정밀도 저하 방지
    bnb_4bit_use_double_quant=True,         # 이중 양자화: 양자화를 적용하는 정수에 대해서도 양자화 적용
    bnb_4bit_compute_dtype=torch.bfloat16   # 연산 속도를 높이기 위해 사용 (default: torch.float32)
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=nf4_config,
    device_map="auto"
)
```

# 2. 학습 프롬프트 생성
```python
prompt_input_template = """아래는 작업을 설명하는 지시사항과 추가 정보를 제공하는 입력이 짝으로 구성됩니다. 이에 대한 적절한 응답을 작성해주세요.

### 지시사항:
{instruction}

### 입력:
{input}

### 응답:"""


prompt_no_input_template = """아래는 작업을 설명하는 지시사항입니다. 이에 대한 적절한 응답을 작성해주세요.

### 지시사항:
{instruction}

### 응답:"""
```
```python
def generate_prompt(data_point):
    instruction = data_point["instruction"]
    input = data_point["input"]
    label = data_point["output"]

    if input:
    res = prompt_input_template.format(instruction=instruction, input=input)
    else:
    res = prompt_no_input_template.format(instruction=instruction)

    if label:
    res = f"{res}{label}<|im_end|>" # eos_token을 마지막에 추가

    data_point['text'] = res

    return data_point
```
```python
# 데이터셋에 프롬프트 적용
remove_column_keys = dataset.features.keys()    # 기존 컬럼(instruction, output 등) 제거
dataset_cvted = dataset.shuffle().map(generate_prompt, remove_columns=remove_column_keys)
```
```python
# 토큰화를 통해 텍스트를 숫자 시퀸스로 변경
def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, max_length=512)
    return outputs

remove_column_keys = dataset_cvted.features.keys()
dataset_tokenized = dataset_cvted.map(tokenize_function, batched=True, remove_columns=remove_column_keys)
```

# 3. QLoRA 설정
- QLoRA 파라미터 설정
    ```python
    lora_config = LoraConfig(
        r=4,                # LoRA 가중치 행렬의 rank. 정수형이며 값이 작을수록 trainable parameter가 적어짐
        lora_alpha=8,       # LoRA 스케일링 팩터. 추론 시 PLM weight와 합칠 때 LoRA weight의 스케일을 일정하게 유지하기 위해 사용
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], # LoRA를 적용할 layer. 모델 아키텍처에 따라 달라짐
        bias='none',        # bias 파라미터를 학습시킬지 지정. ['none', 'all', 'lora_only']
        task_type=TaskType.CAUSAL_LM
    )

    # 양자화된 모델을 학습하기 전, 전처리를 위해 호출
    model = prepare_model_for_kbit_training(model)

    # LoRA 학습을 위해서는 아래와 같이 peft를 사용하여 모델을 wrapping 해주어야 함
    model = get_peft_model(model, lora_config)

    # 학습 파라미터 확인
    model.print_trainable_parameters()
    ```

# 4. Trainer 설정 및 실행
- HuggingFace의 transformers 라이브러리에서 제공하는 트레이너 클래스 사용하여 학습
- max_steps를 컴퓨터 사양에 따라 설정
    ```python
    # Data Collator 역할
    # 각 입력 시퀀스의 input_ids(토큰) 길이를 계산하고, 가장 긴 길이를 기준으로 길이가 짧은 시퀀스에는 패딩 토큰 추가
    def collate_fn(examples):
        examples_batch = tokenizer.pad(examples, padding='longest', return_tensors='pt')
        examples_batch['labels'] = examples_batch['input_ids']    # 모델 학습 평가를 위한 loss 계산을 위해 입력 토큰을 레이블로 사용
        return examples_batch

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=2,  # 각 디바이스당 배치 사이즈. 작을수록(1~2) 좀 더 빠르게 alignment 됨
        gradient_accumulation_steps=4, 
        warmup_steps=1,
        #num_train_epochs=1,
        max_steps=1000, 
        learning_rate=2e-4,             # 학습률
        bf16=True,                      # bf16 사용 (지원되는 하드웨어 확인 필요)
        output_dir="outputs",
        optim="paged_adamw_8bit",       # 8비트 AdamW 옵티마이저
        logging_steps=50,               # 로깅 빈도
        save_total_limit=3              # 저장할 체크포인트의 최대 수
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_tokenized,
        max_seq_length=512, # 최대 시퀀스 길이
        args=train_args,
        dataset_text_field="text",
        data_collator=collate_fn
    )
    ```
- 학습 수행
    ```python
    model.config.use_cache = False
    trainer.train()
    ```
- 학습 완료 후, QLoRA 모델 로컬에 저장
    ```python
    FINETUNED_MODEL = "eeve_qlora"
    trainer.model.save_pretrained(FINETUNED_MODEL)
    ```

### QLoRA 파인튜닝 모델 테스트
# 1. 모델 불러오기
```python
FINETUNED_MODEL = "./eeve_qlora"

peft_config = PeftConfig.from_pretrained(FINETUNED_MODEL)

# 베이스 모델 및 토크나이저 로드
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
```

# 2. 모델 실행
```python
prompt = "한국의 아이돌 문화에 대해 알려줘."

# 텍스트 생성을 위한 파이프라인 설정
pipe = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=256)
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
```

### 출처
- [QLoRA로 학습하기](https://blog.kbanknow.com/82)