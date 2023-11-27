import random
import os
from utils.tokenizer import Vocabulary

def load_text(text_path="./data/origin/Train.txt"):
    with open(text_path, "r") as f:
        corpus = f.read().strip("\n")
    return corpus

def shuffle_data(data_path, all_data_path):
    """
    打乱数据
    
    Parameters
    ----------
    @param data_path: str
        原始数据路径
    @param all_data_path: str
        打乱后的数据路径
    """
    # 清空dict文件
    with open(all_data_path, 'w') as f:
        f.seek(0)
        f.truncate() 
    # 读取全部数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 打乱数据
    random.shuffle(lines)
    # 把数据写入文件
    with open(all_data_path, 'w', encoding='utf-8') as f:
        for line in lines:
            label = line.split('\t')[0]
            content = line.split('\t')[-1]
            f.write(label + '\t' + content)

def split_data_and_to_indices(val_path, train_path, all_data_path, vocab_path):
    #在生成数据之前，首先将val_path和train_path内容清空
    with open(val_path, 'w', encoding='utf-8') as f_val:
        f_val.seek(0)
        f_val.truncate()
        
    with open(train_path, 'w', encoding='utf-8') as f_train:
        f_train.seek(0)
        f_train.truncate() 

    with open(all_data_path, 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    # 读取vocab
    vocab = Vocabulary.load(vocab_path)
    
    # 生成训练集和验证集
    with open(val_path, 'a', encoding='utf-8') as f_val, open(train_path, 'a', encoding='utf-8') as f_train:
        for i, line in enumerate(lines):
            content = line.split('\t')[-1].replace('\n', '')
            label = line.split('\t')[0]
            # 每8个抽取一个数据用于验证
            if i % 8 == 0:
                val_tokens = vocab.to_indices(content)
                f_val.write(str(val_tokens) + '\t' + label + '\n')
            else:
                train_tokens = vocab.to_indices(content)
                f_train.write(str(train_tokens) + '\t' + label + '\n')

