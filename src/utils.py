"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import sys
import csv
csv.field_size_limit(sys.maxsize)
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        report = classification_report(y_true, y_pred)

        macroF1 = f1_score(y_true, y_pred, average='macro')
        microF1 = f1_score(y_true, y_pred, average='micro')

        output['confusion_matrix'] = str(report + "\n" + 
                                        "Micro F1 score === "+str(microF1)+"\n"+
                                        "Maacro F1 score === "+str(macroF1)+"\n\n")
        output['microF1'] = microF1
        output['macroF1'] = macroF1
        
        print(report)
        print("\n------------- Micro F1 Score == {} ------------".format(microF1))
        print("------------- Macro F1 Score == {} ------------\n".format(macroF1))

        #output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(df):
    word_length_list = []
    sent_length_list = []
    reviews = df['review'].values
    for text in reviews:
        sent_list = sent_tokenize(text)
        sent_length_list.append(len(sent_list))

        for sent in sent_list:
            word_list = word_tokenize(sent)
            word_length_list.append(len(word_list))

    sorted_word_length = sorted(word_length_list)
    sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]

if __name__ == "__main__":
    word, sent = get_max_lengths("../data/test.csv")
    print (word)
    print (sent)






