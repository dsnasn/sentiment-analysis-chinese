import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

def main():
    # 加载模型和分词器
    model_dir = "./results/"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 输入测试
    while True:
        text = input("输入句子进行分类 (输入 'exit' 退出): ")
        if text.lower() == 'exit':
            break
        result = predict(text, model, tokenizer, device)
        print(f"情感分类结果: {result}")

if __name__ == "__main__":
    main()
