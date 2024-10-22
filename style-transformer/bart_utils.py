import os

from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./cnn-dailymail/cnn_dm/", type_path="train", block_size=1024):
        super(SummarizationDataset,).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []

        self.source_style = []

        print("loading " + type_path + " source.")

        with open(os.path.join(data_dir, type_path + ".source"), "r") as f:
            for line in f.readlines():  # each text is a line and a full story
                text, text_style = line.strip('\"|\n').split('__label__')
                self.source_style.append(int(text_style))
                tokenized = tokenizer.batch_encode_plus(
                    [text], max_length=block_size, pad_to_max_length=True, return_tensors="pt"
                )
                self.source.append(tokenized)
            f.close()

        print("loading " + type_path + " target.")

        with open(os.path.join(data_dir, type_path + ".target"), "r") as f:
            for text in f.readlines():  # each text is a line and a summary
                text=text.strip('\"|\n')
                tokenized = tokenizer.batch_encode_plus(
                    [text], max_length=56, pad_to_max_length=True, return_tensors="pt"
                )
                self.target.append(tokenized)
            f.close()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        text_style = self.source_style[index]
        src_mask = self.source[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "source_style": text_style ,"target_ids": target_ids}