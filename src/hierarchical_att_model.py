"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length,opt):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        self.opt = opt
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
                                       #bidirection
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        #if torch.cuda.is_available():
        #    self.word_hidden_state = self.word_hidden_state.cuda()
        #    self.sent_hidden_state = self.sent_hidden_state.cuda()
        
        self.word_hidden_state = self.word_hidden_state.to(self.opt.device)
        self.sent_hidden_state = self.sent_hidden_state.to(self.opt.device)
        
        

    def forward(self, input):

        # input shape = (batch_size,max_sent_length, max_word_length)
        output_list = []
        input = input.permute(1, 0, 2) # We have created nn.GRU with batchFirst = False, 
        # so input_shape = (max_sent_len, batch_size, max_word_length)

        word_alpha = []
        for i in input:
            # i.shape = (batch_size, max_word_length) i.e ithe sentence of all documents
            output, self.word_hidden_state,alpha = self.word_att_net(i.permute(1, 0), self.word_hidden_state) #output_shape = (1,batch_size, 2*hidden_size)
                                                                                                           # alpha = (batch_size,max_word_len)     
            word_alpha.append(alpha)
            output_list.append(output)

        word_alpha = torch.stack(word_alpha) # (max_sent_length, batch_size, max_word_length)
        output = torch.cat(output_list, 0) # shape = (max_sent_len, batch_size, 2*hidden_size)
        #print(output.shape)
        output, self.sent_hidden_state, sent_alpha = self.sent_att_net(output, self.sent_hidden_state)
        # output = (batch_size, num_classes)
        # sent_alpha = (batch_size, max_seq_len)

        return output, word_alpha, sent_alpha
