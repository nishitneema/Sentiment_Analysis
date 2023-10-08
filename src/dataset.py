"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class MyDataset(Dataset):

    def __init__(self, df, dict_path, n_folds = 1,leave_out_fold = 0,split = 'train',max_length_sentences=30, max_length_word=35,undersample = False,oversample = False):
        super(MyDataset, self).__init__()

        # texts, labels = [], []
        # with open(data_path) as csv_file:
        #     reader = csv.reader(csv_file, quotechar='"')
        #     for idx, line in enumerate(reader):
        #         text = ""
        #         for tx in line[1:]:
        #             text += tx.lower()
        #             text += " "
        #         label = int(line[0]) - 1
        #         texts.append(text)
        #         labels.append(label)
        texts = list(df['review'].values)
        labels = list(df['sentiment'].values)
        
        if n_folds == 1:
            # 80-20 split
            if split == 'train':
                X,Y = texts[:int(len(texts)*0.8)], labels[:int(len(texts)*0.8)]
            else:
                X,Y = texts[int(len(texts)*0.8):], labels[int(len(texts)*0.8):]
        else:
            each_fold = int(len(texts)/n_folds) 
            if split == 'train':
                print(leave_out_fold*each_fold)
                X_l,Y_l = texts[: leave_out_fold*each_fold ], labels[: leave_out_fold*each_fold ]
                X_r,Y_r = texts[(leave_out_fold+1)*each_fold:], labels[(leave_out_fold+1)*each_fold :]
                X_l.extend(X_r)
                Y_l.extend(Y_r)
                X = X_l
                Y = Y_l
            else:
                X,Y = texts[leave_out_fold*each_fold: (leave_out_fold+1)*each_fold ], labels[leave_out_fold*each_fold : (leave_out_fold+1)*each_fold ]

        if undersample:
            from imblearn.under_sampling import RandomUnderSampler
            d = {0:100,1:500,2:500}
            oversampler = RandomUnderSampler(sampling_strategy=d,random_state=42)
            X, Y = oversampler.fit_resample(np.array(X).reshape(-1, 1), Y)
            X = X[:,0]
        if oversample:
            from imblearn.over_sampling import RandomOverSampler
            oversampler = RandomOverSampler(random_state=42)
            X, Y = oversampler.fit_resample(np.array(X).reshape(-1, 1), Y)
            X = X[:,0]
            
        self.texts = X
        self.labels = Y

        # self.texts = texts
        # self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

        for sentences in document_encode:
            # Pad each sentence with -1 at end such that sentence size = max_length_word
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            # Below ensures if any document has #sentences < max_length_sentences, then pad remaining sentences with -1
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        # If number of sentences exceed, take only max_length_sentences
        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        # At this point document_encode has max_length_sentences and each sentence has max_length words
        document_encode = np.stack(arrays=document_encode, axis=0) #shape = (max_length_Sentence, max_length_words)
        document_encode += 1

        return document_encode.astype(np.int64), label


if __name__ == '__main__':
    test = MyDataset(data_path="../data/test.csv", dict_path="../data/glove.6B.50d.txt")
    print (test.__getitem__(index=1)[0].shape)
