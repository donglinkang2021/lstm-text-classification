import torch

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(NewsDataset, self).__init__()
        self.data_path = data_path
        self.data_list = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if len(line.split('\t')[0]) == 0:
                continue
            
            ids = eval(line.split('\t')[0])
            label = eval(line.split('\t')[1])
            self.data_list.append([ids, label])

    def __getitem__(self, index):
        ids = self.data_list[index][0]
        label = self.data_list[index][1]
        return ids, label

    def __len__(self):
        return len(self.data_list)
    

def collate_fn(data, max_length=32, pad_idx = 1):
    # 对id序列进行padding
    def pad_sequence(sequence, max_length, pad_idx=pad_idx):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [pad_idx] * (max_length - len(sequence))
    
    # 对数据进行分离
    indice, label = zip(*data)

    # 对id序列进行padding
    padded_indice = [pad_sequence(ids, max_length) for ids in indice]    

    return torch.tensor(padded_indice), torch.LongTensor(label)

