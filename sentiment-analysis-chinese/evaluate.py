import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # 配置路径
    model_dir = "./results/"
    test_file = "./data/test.txt"

    # 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # 加载测试数据
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            test_data.append({'label': int(label), 'text': text})

    # 数据预处理
    inputs = tokenizer([d['text'] for d in test_data], return_tensors="pt", padding=True, truncation=True, max_length=128)
    labels = torch.tensor([d['label'] for d in test_data])

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=1)

    # 计算指标
    acc = accuracy_score(labels, predictions)
    print(f"测试集准确率: {acc:.2f}")

    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
