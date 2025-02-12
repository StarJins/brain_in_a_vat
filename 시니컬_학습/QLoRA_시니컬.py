import pandas as pd
import torch
import transformers
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    TaskType
)
from trl import SFTTrainer
import json

BASE_MODEL = "Bllossom/llama-3.2-Korean-Bllossom-3B"
FINETUNED_MODEL = "sinical-model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.model_max_length = 512
tokenizer.pad_token = tokenizer.eos_token

def save_dataset():
    file_path = "./시니컬_학습_데이터.json"
    with open(file_path, "r", encoding="utf-8") as file:
        data_sinical = json.load(file)  # JSON 파일을 Python 딕셔너리로 변환

    # DataFrame 변환
    df_sinical = pd.DataFrame(data_sinical)  # 단일 딕셔너리를 DataFrame으로 변환
    df_sinical = df_sinical.drop_duplicates(keep='first', ignore_index=True)
    dataset_sinical = Dataset.from_pandas(df_sinical)
    
    dataset_sinical.save_to_disk("./dataset/sinical")

def generate_prompt(data_point):
    prompt_input_template = """아래는 작업을 설명하는 지시사항과 추가 정보를 제공하는 입력이 짝으로 구성됩니다. 이에 대한 적절한 응답을 작성해주세요.

    ### 지시사항:
    {instruction}

    ### 입력:
    {input}

    ### 응답:"""

    instruction = data_point["instruction"]
    input = data_point["input"]
    label = data_point["output"]

    if input:
        res = prompt_input_template.format(instruction=instruction, input=input)

    if label:
        res = f"{res}{label}<|im_end|>" # eos_token을 마지막에 추가

    data_point['text'] = res

    return data_point

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, max_length=512)
    return outputs

# Data Collator 역할
# 각 입력 시퀀스의 input_ids(토큰) 길이를 계산하고, 가장 긴 길이를 기준으로 길이가 짧은 시퀀스에는 패딩 토큰 추가
def collate_fn(examples):
    examples_batch = tokenizer.pad(examples, padding='longest', return_tensors='pt')
    examples_batch['labels'] = examples_batch['input_ids'] # 모델 학습 평가를 위한 loss 계산을 위해 입력 토큰을 레이블로 사용
    return examples_batch

def train():
    # dataset 로드 및 프롬프트 설정
    dataset_path = "./dataset/sinical"
    dataset = load_from_disk(dataset_path)

    remove_column_keys = dataset.features.keys() # 기존 컬럼(instruction, output 등) 제거
    dataset_cvted = dataset.shuffle().map(generate_prompt, remove_columns=remove_column_keys)

    remove_column_keys = dataset_cvted.features.keys()
    dataset_tokenized = dataset_cvted.map(tokenize_function, batched=True, remove_columns=remove_column_keys)
    
    lora_config = LoraConfig(
        r=4, # LoRA 가중치 행렬의 rank. 정수형이며 값이 작을수록 trainable parameter가 적어짐
        lora_alpha=8, # LoRA 스케일링 팩터. 추론 시 PLM weight와 합칠 때 LoRA weight의 스케일을 일정하게 유지하기 위해 사용
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], # LoRA를 적용할 layer. 모델 아키텍처에 따라 달라짐
        bias='none', # bias 파라미터를 학습시킬지 지정. ['none', 'all', 'lora_only']
        task_type=TaskType.CAUSAL_LM
    )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=nf4_config,
        device_map="auto"
    )

    # 양자화된 모델을 학습하기 전, 전처리를 위해 호출
    model = prepare_model_for_kbit_training(model)
    # LoRA 학습을 위해서는 아래와 같이 peft를 사용하여 모델을 wrapping 해주어야 함
    model = get_peft_model(model, lora_config)

    # 학습 파라미터 확인
    model.print_trainable_parameters()  # trainable params: 6,078,464 || all params: 3,218,828,288 || trainable%: 0.1888

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=1, # 각 디바이스당 배치 사이즈. 작을수록(1~2) 좀 더 빠르게 alignment 됨
        gradient_accumulation_steps=4, 
        warmup_steps=1,
        #num_train_epochs=1,
        max_steps=50, 
        learning_rate=2e-4, # 학습률
        fp16=True,
        output_dir="outputs",
        optim="paged_adamw_8bit", # 8비트 AdamW 옵티마이저
        logging_steps=50, # 로깅 빈도
        save_total_limit=3 # 저장할 체크포인트의 최대 수
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_tokenized,
        args=train_args,
        data_collator=collate_fn,
    )

    model.config.use_cache = False
    trainer.train()
    trainer.model.save_pretrained(FINETUNED_MODEL)

def main():
    # save_dataset()
    train()

if __name__ == '__main__':
    main()