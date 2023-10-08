"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentAttNet, self).__init__()

        #self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        #self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.sent = nn.Linear(2*sent_hidden_size, 2*sent_hidden_size) # includes both weight(Ws) and bias(bs)
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1)) # u
        self.sent_hidden_size = sent_hidden_size

        self.gru = nn.GRU(input_size = 2 * word_hidden_size,hidden_size= sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        #self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        # # input shape = (max_sent_len, batch_size, 2*word_hidden_dim)    
        # f_output, h_output = self.gru(input, hidden_state) # shape = (max_sent_len, batch_size, 2*sent_hidden_size)
        # output = matrix_mul(f_output, self.sent_weight, self.sent_bias) # shape = (max_sent_len, batch_size, 2*sent_hidden_size)
        # output = matrix_mul(output, self.context_weight).permute(1, 0) # shape = (batch_size, max_sent_len)
        # output = F.softmax(output,dim = 1)
        # output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0) #shape = (batch_size, 2*sent_hidden_size)
        # output = self.fc(output) # shape = (batch_size, num_classes)

        # return output, h_output
    


        # input shape = (max_sent_len, batch_size, 2*word_hidden_dim)    
        f_output, h_output = self.gru(input, hidden_state) # shape = (max_sent_len, batch_size, 2*sent_hidden_size)

        output = torch.tanh(self.sent(f_output)) # shape = (max_sent_len, batch_size, 2*sent_hidden_size)

        output = (output @ self.context_weight).squeeze(dim = 2).permute(1,0)# shape = (batch_size, max_sent_len)

        alpha = F.softmax(output,dim = 1)

        s = []
        for i,document in enumerate(f_output.permute(1,2,0)):
            # document shape = (2*sent_hidden_size, max_sent_len)
            s_document = (document @ alpha[i]).reshape(1,-1) # s.shape = (1,2*sent_hidden_size)
            s.append(s_document)

        output = torch.cat(s,0) #shape = (batch_size,2*sent_hidden_size)

        output = self.fc(output) # shape = (batch_size, num_classes)

        return output, h_output, alpha


if __name__ == "__main__":
    abc = SentAttNet()
