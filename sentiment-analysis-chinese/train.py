import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 数据加载函数
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                label, text = line.strip().split('\t')
                data.append({'label': int(label), 'text': text})
    return Dataset.from_list(data)

# 数据预处理函数
def preprocess_function(examples, tokenizer, max_length=128):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

def main():
    # 路径和参数配置
    model_name = "bert-base-chinese"
    train_file = "./data/train.txt"
    valid_file = "./data/valid.txt"
    output_dir = "./results/"
    max_length = 128
    batch_size = 32

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 加载数据
    train_dataset = load_data(train_file)
    valid_dataset = load_data(valid_file)

    # 数据预处理
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True)
    valid_dataset = valid_dataset.map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    # 自定义评价指标
    from sklearn.metrics import precision_recall_fscore_support
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        return {"precision": precision, "recall": recall, "f1": f1}

    # Trainer 定义
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model(output_dir)
    print(f"模型已保存到 {output_dir}")

if __name__ == "__main__":
    main()
