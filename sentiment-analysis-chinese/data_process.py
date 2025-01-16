import os
import random
import pandas as pd

def preprocess_data(input_file, output_dir, max_samples=5000, test_ratio=0.2, valid_ratio=0.1):
    df = pd.read_csv(input_file)
    all_data_path = os.path.join(output_dir, "all_data.txt")
    
    with open(all_data_path, 'w', encoding="utf-8") as f:
        for _, row in df.iterrows():
            label = row['label']
            context = row['context'].split('\t')[1].split(' ')[0]
            if label == "depression":
                f.write(f"0\t{context}\n")
            elif label == "good":
                f.write(f"1\t{context}\n")
    
    all_data_list = []
    with open(all_data_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            all_data_list.append(line.strip())
    
    random.shuffle(all_data_list)
    total_samples = min(len(all_data_list), max_samples)

    test_size = int(test_ratio * total_samples)
    valid_size = int(valid_ratio * total_samples)
    train_size = total_samples - test_size - valid_size

    test_data = all_data_list[:test_size]
    valid_data = all_data_list[test_size:test_size + valid_size]
    train_data = all_data_list[test_size + valid_size:]

    for name, data in zip(["train", "valid", "test"], [train_data, valid_data, test_data]):
        with open(os.path.join(output_dir, f"{name}.txt"), 'w', encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")

    print(f"数据处理完成：训练集 {len(train_data)} 条，验证集 {len(valid_data)} 条，测试集 {len(test_data)} 条。")

if __name__ == "__main__":
    input_file = "./data/data.csv"
    output_dir = "./data/"
    preprocess_data(input_file, output_dir)