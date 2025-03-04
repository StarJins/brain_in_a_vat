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
    print(dataset_sinical)
    
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

    data_point["text"] = res

    return data_point

def tokenize_function(examples):
    # Fast tokenizer를 사용하여 input_ids와 attention_mask만 반환
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=True,  # Fast Tokenizer 권장 방식
        return_tensors="pt"
    )

def collate_fn(examples):
    # 이미 tokenized된 데이터 사용
    input_ids = [torch.tensor(example["input_ids"]) for example in examples]
    attention_mask = [torch.tensor(example["attention_mask"]) for example in examples]

    # 배치 처리
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(input_ids)  # 'labels'는 'input_ids'와 동일하게 설정
    }


def train():
    # dataset 로드 및 프롬프트 설정
    dataset_path = "./dataset/sinical"
    dataset = load_from_disk(dataset_path)

    remove_column_keys = dataset.features.keys() # 기존 컬럼(instruction, output 등) 제거
    dataset_cvted = dataset.shuffle().map(generate_prompt, remove_columns=remove_column_keys)

    remove_column_keys = [col for col in dataset_cvted.features.keys() if col != "text"]  # "text" 컬럼 유지
    dataset_tokenized = dataset_cvted.map(tokenize_function, batched=True, remove_columns=remove_column_keys)
    
    lora_config = LoraConfig(
        r=4,                # LoRA 가중치 행렬의 rank. 정수형이며 값이 작을수록 trainable parameter가 적어짐
        lora_alpha=8,       # LoRA 스케일링 팩터
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],   # LoRA를 적용할 layer
        bias='none',        # bias 파라미터를 학습시킬지 지정
        task_type=TaskType.CAUSAL_LM
    )

    # nf4_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        # quantization_config=nf4_config,
        device_map="cuda"  # auto, cpu, cuda
    )

    # 양자화된 모델을 학습하기 전, 전처리를 위해 호출
    # model = prepare_model_for_kbit_training(model)
    # LoRA 학습을 위해 모델을 wrapping
    model = get_peft_model(model, lora_config)

    # 학습 파라미터 확인
    model.print_trainable_parameters()  # trainable params: 6,078,464 || all params: 3,218,828,288 || trainable%: 0.1888

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=1,  # 각 디바이스(GPU 또는 CPU)에서 한 번에 처리할 샘플(batch) 개수를 설정합니다. 각 디바이스당 배치 사이즈. 작을수록(1~2) 좀 더 빠르게 alignment 됨
        gradient_accumulation_steps=4,  # 그래디언트 업데이트 전에 몇 개의 미니배치를 누적(accumulate)할지 정하는 값입니다. 예를 들어, per_device_train_batch_size=1이고 gradient_accumulation_steps=4이면, 내부적으로 배치 크기가 1 × 4 = 4처럼 작동합니다.
        warmup_steps=65,                # 학습 초기에 학습률을 천천히 증가시키는 워밍업 스텝 수를 설정합니다. 0.05 * total_steps
        num_train_epochs=10,            # 데이터셋을 전체적으로 몇 번 반복할지 정하는 파라미터입니다.
        # max_steps=50,                 # num_train_epochs와 다르게, 전체 데이터셋을 몇 번 학습하는지와 관계없이 지정된 스텝 수에 도달하면 학습이 종료 됩니다.
        learning_rate=1.5e-4,           # 학습률. 학습률이 너무 크면 학습이 불안정하고, 너무 작으면 학습 속도가 느려집니다.
        fp16=True,                      # 일반적으로 GPU에서 더 빠르고 적은 메모리를 사용하여 학습 가능합니다.
        output_dir="outputs",           # 모델 가중치, 로그, 체크포인트 등을 저장할 디렉토리를 지정합니다. outputs/ 폴더에 저장
        # optim="paged_adamw_8bit",     # 8비트 AdamW 옵티마이저
        optim="adamw_torch",            # 8bit AdamW 제거하고 일반 AdamW 사용
        logging_steps=10,               # 로깅 빈도
        save_total_limit=5              # 저장할 체크포인트의 최대 수
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
    save_dataset()
    train()

if __name__ == '__main__':
    main()
