"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv

class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float64))

        #self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        #self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        #self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.word = nn.Linear(2*hidden_size, 2*hidden_size) # includes both weight(Ww) and bias(bw)
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1)) # u
        self.hidden_size = hidden_size


        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(input_size = embed_size,hidden_size =  hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        #self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        # # input shape = (max_word_length, batch_size) because batchFirst = False
        # output = self.lookup(input) # shape = max_word_len, batch_size, emb_size)

        # f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        # # f_output.shape = (max_word_len, batch_size, 2*hidden_size) , bidirectional GRU

        # output = matrix_mul(f_output, self.word_weight, self.word_bias) # shape = (max_word_len, batch_size, 2*hidden_size)

        # output = matrix_mul(output, self.context_weight).permute(1,0) # shape = (batch_size, max_word_length)

        # output = F.softmax(output,dim = 1)
    
        # output = element_wise_mul(f_output,output.permute(1,0)) # shape = (1, batch_size, 2*hidden_size)

        # return output, h_output



        # input shape = (max_word_length, batch_size) because batchFirst = False
        output = self.lookup(input) # shape = (max_word_len, batch_size, emb_size)

        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        # f_output.shape = (max_word_len, batch_size, 2*hidden_size) , bidirectional GRU

        output = torch.tanh(self.word(f_output)) # shape = (max_word_len, batch_size, 2*hidden_size)

        output = (output @ self.context_weight).squeeze(dim = 2).permute(1,0) # shape = (batch_size,max_word_length)

        alpha = F.softmax(output,dim = 1)
    
        s = []
        for i,sentence in enumerate(f_output.permute(1,2,0)):
            # sentence shape = (2*hidden_size, max_word_len)
            s_sentence = (sentence @ alpha[i]).reshape(1,-1) # s.shape = (1,2*hidden_size)
            s.append(s_sentence)

        output = torch.cat(s,0) #shape = (batch_size,2*hidden_size)
        output = output.unsqueeze(0) # shape = (1,batch_size, 2*hidden_size)

        return output, h_output,alpha


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
