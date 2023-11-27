from utils.tokenizer import Vocabulary
import torch

def create_test_indices(test_path, all_data_path, vocab_path):
    #在生成数据之前，首先将test_path内容清空
    with open(test_path, 'w', encoding='utf-8') as f_test:
        f_test.seek(0)
        f_test.truncate()

    with open(all_data_path, 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    # 读取vocab
    vocab = Vocabulary.load(vocab_path)
    
    # 生成测试集
    with open(test_path, 'a', encoding='utf-8') as f_test:
        for line in lines:
            content = line.replace('\n', '')
            test_tokens = vocab.to_indices(content)
            f_test.write(str(test_tokens) + '\n')


class TestNewsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(TestNewsDataset, self).__init__()
        self.data_path = data_path
        self.data_list = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            ids = eval(line)
            self.data_list.append(ids)

    def __getitem__(self, index):
        ids = self.data_list[index]
        return ids

    def __len__(self):
        return len(self.data_list)

  
def collate_fn_test(data, max_length=32, pad_idx = 1):
    # 对id序列进行padding
    def pad_sequence(sequence, max_length, pad_idx=pad_idx):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [pad_idx] * (max_length - len(sequence))
    
    # 对数据进行分离
    # 对id序列进行padding
    padded_indice = [pad_sequence(ids, max_length) for ids in data]    

    return torch.tensor(padded_indice)