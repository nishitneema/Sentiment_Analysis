import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.hierarchical_att_model import HierAttNet
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
from string import punctuation
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    # parser.add_argument("--train_set", type=str, default="data/dbpedia_csv/train.csv")
    # parser.add_argument("--test_set", type=str, default="data/dbpedia_csv/test.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="data/glove.6B.50d.txt")
    parser.add_argument("--k_fold",action="store_true",help = "Whether to perform K-fold Cross validation")
    parser.add_argument("--n_folds",type = int,default = 4,help = "Number of folds in K_fold Cross Validation")
    parser.add_argument("--oversample",action="store_true",help = "Whether to balance classes by oversampling")
    parser.add_argument("--undersample",action="store_true",help = "Whether to balance classes by oversampling")
    # parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    # parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--gpu_id",type = int,default = 0)
    #parser.add_argument("--model_name",type = str, default = "whole_model_han")
    args = parser.parse_args()
    return args


# Change log_path,saved_path, df reading

# python train.py [test and valid set, without K - fold cross validation]
# python train.py --k_fold [performs k_fold cross validation]

def eval(opt,model,test_set,test_generator):
            print("\n --------Evaluating model on valid set --------\n")
            criterion = nn.CrossEntropyLoss()
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_generator:
                num_sample = len(te_label)
                #if torch.cuda.is_available():
                #    te_feature = te_feature.cuda()
                #    te_label = te_label.cuda()
                te_feature = te_feature.to(opt.device)
                te_label = te_label.to(opt.device)
                
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions,_,_ = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())

            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            print("Test_Loss: {}, Test_Accuracy: {}".format(
                te_loss, test_metrics["accuracy"]))
            
            return test_metrics['microF1'],test_metrics['macroF1']
    
def train(opt,training_set,test_set,save = False):
    

    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    
    test_generator = DataLoader(test_set, **test_params)
    training_generator = DataLoader(training_set, **training_params)

    print("\n\nTrain data size == ",len(training_set))
    print("Val data size == ",len(test_set))

    from collections import Counter
    print("\n\nTraining labels frequencies == \n",Counter(training_set.labels))
    print("Test label frequencies == \n\n",Counter(test_set.labels))
    
    print("\n------- Reading Done------\n")

    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes,
                       opt.word2vec_path, max_sent_length, max_word_length,opt)


    if save:
        if os.path.isdir(opt.log_path):
            shutil.rmtree(opt.log_path)
        os.makedirs(opt.log_path)
        writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

    #if torch.cuda.is_available():
    #    model.cuda()
    
    model = model.to(opt.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum,weight_decay = 1e-4)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        for iter, (feature, label) in enumerate(training_generator):
            #if torch.cuda.is_available():
            #    feature = feature.cuda() # shape = (batch_size, max_sen_len, max_word_length)
            #    label = label.cuda() # shape = (batch_size,1)

            feature = feature.to(opt.device)
            label = label.to(opt.device)
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions,_,_ = model(feature) # shape = (batch_size, num_classes)
            loss = criterion(predictions, label) # It implicitly calculated the softmax on predictions
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            if save:
                writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)

            if save and iter % 100 == 0:
                print("\n -------- Saved model -------- \n")
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_generator:
                num_sample = len(te_label)
                #if torch.cuda.is_available():
                #    te_feature = te_feature.cuda()
                #    te_label = te_label.cuda()
                te_feature = te_feature.to(opt.device)
                te_label = te_label.to(opt.device)
                
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions,_,_ = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            if save:
                output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Test_Loss: {}, Test_Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            if save:
                writer.add_scalar('Test/Loss', te_loss, epoch)
                writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            if save and te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                print("\n ---------- Saved model ------ \n")
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if save and epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break
    return eval(opt,model,test_set,test_generator)

if __name__ == "__main__":
    
    opt = get_args()
    opt.device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # model = model.to(opt.device)
    # for param_tensor in model.state_dict():
    #     print(param_tensor)

    

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)


    def fun1(review):
        review = review.lower() # lowercase, standardize
        return ''.join([c for c in review if c not in punctuation])

    def fun2(x):
        if x=='negative':
            return 0
        elif x=='positive':
            return 1
        else:
            return 2

    # Classification0,2 are imbalanced

    
    
    df = pd.read_excel('ClassificationDataset2.xlsx',names = ['sentiment','review']) # classification2
    #df = pd.read_csv('ClassificationDataset1_new.csv') # ../../Sentiment_classification(Movie_reviews)/train.csv, Classification1_new.csv

    df = df.dropna()
    df = df.sample(frac=1) # shuffles the data

    # get rid of punctuation
    df['review'] = df['review'].apply(fun1)
    df['sentiment'] = df['sentiment'].apply(lambda x: x-1) # classification2


    
    print("\n------- Getting max_lengths -----\n")
    max_word_length, max_sent_length = get_max_lengths(df)
    # max_word_length = 20
    # max_sent_length = 8
    print("Max_word_length = ",max_word_length)
    print("Max_sent_length = ",max_sent_length)

    
    if opt.k_fold == False:

        print("\n ------------ Training the model -----------\n")
        opt.saved_path = 'trained_models/ClassificationDataset2'
        opt.log_path = 'trained_models/ClassificationDataset2/tensorboard'
        output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
        output_file.write("Model's parameters: {}".format(vars(opt)))

        print("\n---------- Reading Datasets ---------\n")


        training_set = MyDataset(df, opt.word2vec_path, split = 'train', max_length_sentences = max_sent_length, max_length_word= max_word_length,oversample = False)
        test_set = MyDataset(df, opt.word2vec_path,split = 'valid' ,max_length_sentences = max_sent_length, max_length_word= max_word_length)


        # Training the model
        (microF1,macroF1) = train(opt,training_set,test_set,save = True) # saves the best model

        # Evaluating saved model on test_set
        model = torch.load("/home/manikanta/NLP/Ass_2/Hierarchical-attention-networks-pytorch/trained_models/ClassificationDataset2/whole_model_han")
        
        model = model.to(opt.device)
        test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    
        test_generator = DataLoader(test_set, **test_params)
        (microF1,macroF1) = eval(opt,model,test_set,test_generator)


        print("\n------------- Micro F1 Score == {} ------------".format(microF1))
        print("------------- Macro F1 Score == {} ------------\n".format(macroF1))
    
    elif opt.k_fold == True:
            totalMicroF1 = 0.0
            totalMacroF1 = 0.0
            for fold in range(opt.n_folds):
                print("\n-------------- Fold = {} -------------\n".format(fold))
                train_data = MyDataset(df,opt.word2vec_path,n_folds=opt.n_folds,leave_out_fold=fold,split='train',oversample=opt.oversample,undersample=opt.undersample)
                val_data = MyDataset(df,opt.word2vec_path,n_folds=opt.n_folds,leave_out_fold=fold,split='val')
                (microF1,macroF1) = train(opt,train_data,val_data,save = False)

                print("\n------------- Micro F1 Score == {} ------------".format(microF1))
                print("------------- Macro F1 Score == {} ------------\n".format(macroF1))

                totalMicroF1+=microF1
                totalMacroF1+=macroF1
            print("\n------------- Avg Micro F1 Score == {} ------------".format(totalMicroF1/opt.n_folds))
            print("--------------- Avg Macro F1 Score == {} ------------".format(totalMacroF1/opt.n_folds))    