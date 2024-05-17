import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset, DataLoader
import random
import argparse
#Imported libraries

#!pip install wandb   #please uncomment this line if wandb is not installed in pc
import wandb
import socket
socket.setdefaulttimeout(30)
wandb.login()
wandb.init(project ='vanillaRNN') #wandb installations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # integration with GPU if available

train_csv = "/kaggle/input/telugu/tel/tel_train.csv" # loading train data (Telugu)
test_csv = "/kaggle/input/telugu/tel/tel_test.csv" # loading test data
val_csv = "/kaggle/input/telugu/tel/tel_valid.csv"  # loading validation data

train_data = pd.read_csv(train_csv, header=None)
train_input = train_data[0].to_numpy()
train_output = train_data[1].to_numpy()
val_data = pd.read_csv(val_csv,header = None)
val_input = val_data[0].to_numpy()
val_output = val_data[1].to_numpy()
test_data = pd.read_csv(test_csv,header= None)
test_input = test_data[0].to_numpy()
test_output = test_data[1].to_numpy() # storing the input and ouputs of different data categories


def pre_processing(train_input,train_output): # train_input = english word train_output = Telugu word
    data = {
    "all_characters" : [], # stores all unique Dnglish characters
    "char_num_map" : {}, # character of English to number mapping 
    "num_char_map" : {}, # number to character mapping of English letters
    "source_charToNum": torch.zeros(len(train_input),30, dtype=torch.int, device=device), # stores entires train_input as number sequence
    "source_data" : train_input, # stores input data
        
    "all_characters_2" : [], # stores all unique Telugu characters
    "char_num_map_2" : {}, # character of Telugu to number mapping 
    "num_char_map_2" : {}, # number to character mapping of Telugu letters
    "val_charToNum": torch.zeros(len(train_output),23, dtype=torch.int, device=device), # stores entire train_output as number sequence 
    "target_data" : train_output, # stores output data
    "source_len" : 0, # stores tain_input length
    "target_len" : 0 # stores train_output length
    }
    k = 0 
    l = 0
    for i in range(0,len(train_input)):
        train_input[i] = "{" + train_input[i] + "}"*(29-len(train_input[i])) # '{' denotes start '}' denotes end. Extra '}' are used for padding
        charToNum = []
        for char in (train_input[i]):
            index = 0
            if(char not in data["all_characters"]):
                data["all_characters"].append(char)
                index = data["all_characters"].index(char)
                data["char_num_map"][char] = index
                data["num_char_map"][index] = char
            else:
                index = data["all_characters"].index(char)
            
            charToNum.append(index) # above code converts train_input characters into numbers
            
        my_tensor = torch.tensor(charToNum,device = device)
        data["source_charToNum"][k] = my_tensor
        
        charToNum1 = []
        
        train_output[i] = "{" + train_output[i] + "}"*(22-len(train_output[i]))
        for char in (train_output[i]):
            index = 0
            if(char not in data["all_characters_2"]):
                data["all_characters_2"].append(char)
                index = data["all_characters_2"].index(char)
                data["char_num_map_2"][char] = index
                data["num_char_map_2"][index] = char
            else:
                index = data["all_characters_2"].index(char)
                
            charToNum1.append(index) # above code converts test_out character into numbers
            
        my_tensor1 = torch.tensor(charToNum1,device = device)
        data["val_charToNum"][k] = my_tensor1
        
        k+=1
    
    data["source_len"] = len(data["all_characters"])
    data["target_len"] = len(data["all_characters_2"])
        
    return data # it returns character to number converted data 
    
data = pre_processing(copy.copy(train_input),copy.copy(train_output)) # stored the numbered data


def pre_processing_validation(val_input,val_output): #val_input = validation input data, val_output = validation output data
    data2 = {
    "all_characters" : [],
    "char_num_map" : {},
    "num_char_map" : {},
    "source_charToNum": torch.zeros(len(val_input),30, dtype=torch.int, device=device),
    "source_data" : val_input,
    "all_characters_2" : [],
    "char_num_map_2" : {},
    "num_char_map_2" : {},
    "val_charToNum": torch.zeros(len(val_output),23, dtype=torch.int, device=device),
    "target_data" : val_output,
    "source_len" : 0,
    "target_len" : 0
    }
    k = 0 
    l = 0
    
    m1 = data["char_num_map"]
    m2 = data["char_num_map_2"]
    
    for i in range(0,len(val_input)):
        val_input[i] = "{" + val_input[i] + "}"*(29-len(val_input[i]))
        charToNum = []
        for char in (val_input[i]):
            index = 0
            if(char not in data2["all_characters"]):
                data2["all_characters"].append(char)
                index = m1[char]
                data2["char_num_map"][char] = index
                data2["num_char_map"][index] = char
            else:
                index = m1[char]
            
            charToNum.append(index) # above code converts validation_input characters into numbers
            
        my_tensor = torch.tensor(charToNum,device = device)
        data2["source_charToNum"][k] = my_tensor
        
        charToNum1 = []
        val_output[i] = "{" + val_output[i] + "}"*(22-len(val_output[i]))
        for char in (val_output[i]):
            index = 0
            if(char not in data2["all_characters_2"]):
                data2["all_characters_2"].append(char)
                index = m2[char]
                data2["char_num_map_2"][char] = index
                data2["num_char_map_2"][index] = char
            else:
                index = m2[char]
                
            charToNum1.append(index) # above code converts validation_ouput characters into numbers
            
        my_tensor1 = torch.tensor(charToNum1,device = device)
        data2["val_charToNum"][k] = my_tensor1
        
        k+=1
    
    data2["source_len"] = len(data2["all_characters"])
    data2["target_len"] = len(data2["all_characters_2"])
        
    return data2 # it returns the numbered representation of validation character data
    
    
data2 = pre_processing_validation(copy.copy(val_input),copy.copy(val_output)) # it stores the numbered representation of validation data 

class MyDataset(Dataset):
    def __init__(self, x,y):
        self.source = x
        self.target = y
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        source_data = self.source[idx]
        target_data = self.target[idx]
        return source_data, target_data # Class to load the input and output as batches

def validationAccuracy(encoder,decoder,batchsize,tf_ratio,cellType,bidirection): #tf_ratio : teacher forcing ratio , cellType: GRU,LSTM or RNN, name is justified for other parameters   
    
    dataLoader = dataLoaderFun("validation",batchsize) # dataLoader depending on train or validation
    
    encoder.eval() # encoder and decoder are now in evaluation mode
    decoder.eval()
    
    validation_accuracy = 0
    validation_loss = 0
    
    lossFunction = nn.NLLLoss()
    
    for batch_num, (sourceBatch, targetBatch) in enumerate(dataLoader):
        
        encoderInitialState = encoder.getInitialState() #hiddenlayers * BatchSize * Neurons = encoderInitialState with 0s

        if(cellType=='LSTM'):
            encoderInitialState = (encoderInitialState,encoder.getInitialState())
            
        if(bidirection == "Yes"):
            reversed_batch = torch.flip(sourceBatch, dims=[1]) # reverse the batch across rows.
            sourceBatch = (sourceBatch + reversed_batch)//2 
        
        Output = []
        encoder_output, encoderCurrentState = encoder(sourceBatch,encoderInitialState) # encoder outputs are stored

        loss = 0 # decoder starts form here

        outputSeqLen = targetBatch.shape[1] # here you will get as name justified.


        decoderCurrState = encoderCurrentState

        randNumber = random.random()

        match = []
        for i in range(0,outputSeqLen):

            if(i == 0):
                decoderInputensor = targetBatch[:, i].reshape(batchsize,1) #shape = 32*1
            else:
                if randNumber < tf_ratio:
                    decoderInputensor = targetBatch[:, i].reshape(batchsize, 1) # current batch is passed
                else:
                    decoderInputensor = decoderInputensor.reshape(batchsize, 1) # prev result is passed

            decoderOutput, decoderCurrState = decoder(decoderInputensor,decoderCurrState)

            dummy, topIndeces = decoderOutput.topk(1)  #  get top vales and their indices. 

            decoderOutput = decoderOutput[:, -1, :] #it is just reduce the size from (32*1*67) to (32*67)

            curr_target_chars = targetBatch[:, i] #shape = (32)
            curr_target_chars = curr_target_chars.type(dtype=torch.long)
            loss+=(lossFunction(decoderOutput, curr_target_chars)) # Passing 32*67 softmax values to curr_target_chars which has the 32*1
            
            decoderInputensor = topIndeces.squeeze().detach()  # here whatever top softmax indeces are present but converted to 1 dimension
            Output.append(decoderInputensor) # softmax values are attached  

        tensor_2d = torch.stack(Output)
        Output = tensor_2d.t() #it is outside the for loop

        validation_accuracy += (Output == targetBatch).all(dim=1).sum().item() # it  just summing up the equal words along the row
        validation_loss += (loss.item()/outputSeqLen)
        
        """"for row1,row2 in zip (Output,targetBatch):
            if(row1.tolist() == row2.tolist()):
                temp = row1.tolist()
                t1 = ""
                for x in temp:
                    t1 += data3["num_char_map_2"][x]
                print(t1)"""
        

        if(batch_num%20 == 0):
            print("bt:", batch_num, " loss:", loss.item()/outputSeqLen)
    
    encoder.train() # changing model into train from evolution mode
    decoder.train()
    print("validation_accuracy",validation_accuracy/40.96) # Instead of multiplying by 100 after division I directly calculated percentage
    print("validation_loss",validation_loss)
    wandb.log({'validation_accuracy':validation_accuracy/40.96})
    wandb.log({'validation_loss':validation_loss})


class Encoder(nn.Module):
    
    def __init__(self,inputDim,embSize,encoderLayers,hiddenLayerNuerons,cellType,batch_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(inputDim, embSize)
        self.encoderLayers = encoderLayers
        self.hiddenLayerNuerons = hiddenLayerNuerons
        self.batch_size = batch_size
        
        if(cellType=='GRU'):
            self.rnn = nn.GRU(embSize,hiddenLayerNuerons,num_layers=encoderLayers, batch_first=True)
        elif(cellType=='RNN'):
            self.rnn = nn.RNN(embSize,hiddenLayerNuerons,num_layers=encoderLayers, batch_first=True)
        else:
            self.rnn = nn.LSTM(embSize,hiddenLayerNuerons,num_layers=encoderLayers, batch_first=True)
            
    def forward(self, currentInput, prevState): # forward pass of encoder
        embdInput = self.embedding(currentInput)
        return self.rnn(embdInput, prevState)
    
    def getInitialState(self):
        return torch.zeros(self.encoderLayers,self.batch_size,self.hiddenLayerNuerons, device=device)
    
class Decoder(nn.Module):
    def __init__(self,outputDim,embSize,hiddenLayerNuerons,decoderLayers,cellType,dropout_p):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(outputDim, embSize)
        
        if(cellType == 'GRU'):
            self.rnn = nn.GRU(embSize,hiddenLayerNuerons,num_layers=decoderLayers, batch_first=True)
        elif(cellType == 'RNN'):
            self.rnn = nn.RNN(embSize,hiddenLayerNuerons,num_layers=decoderLayers, batch_first=True)
        else:
            self.rnn = nn.LSTM(embSize,hiddenLayerNuerons,num_layers=decoderLayers, batch_first=True)
            
        self.fc = nn.Linear(hiddenLayerNuerons, outputDim) # it is useful for mapping the calculation to vocabulary
        self.softmax = nn.LogSoftmax(dim=2) #output is in 3rd column 
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, currentInput, prevState): # forward pass of decoder
        embdInput = self.embedding(currentInput)
        currEmbd = F.relu(embdInput)
        output, prevState = self.rnn(currEmbd, prevState)
        output = self.dropout(output)
        output = self.softmax(self.fc(output)) 
        return output, prevState 

data = pre_processing(copy.copy(train_input),copy.copy(train_output)) # input data for model. 

def dataLoaderFun(dataName,batch_size): #dataName : type of data
    if(dataName == 'train'): # returns the Data Loader depending on type of data 
        dataset = MyDataset(data["source_charToNum"],data['val_charToNum'])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif(dataName == 'validation'):
        dataset = MyDataset(data2["source_charToNum"],data2['val_charToNum'])
        return  DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = MyDataset(data2["source_charToNum"],data2['val_charToNum']) #I wrote it for the purpose of debugging. It is never reaching step in this file.
        return  DataLoader(dataset, batch_size=batch_size, shuffle=True) 


def train(embSize,encoderLayers,decoderLayers,hiddenLayerNuerons,cellType,bidirection,dropout,epochs,batchsize,learningRate,optimizer,tf_ratio):
    #embSize : embedding size, cellType: 'GRU','LSTM','RNN', tf_ratio: teacher forcing ratio. For others name is justified
    dataLoader = dataLoaderFun("train",batchsize) # dataLoader depending on train or validation
    
    lossFunction = nn.NLLLoss()
    
    encoder = Encoder(data["source_len"],embSize,encoderLayers,hiddenLayerNuerons,cellType,batchsize).to(device)
    decoder = Decoder(data["target_len"],embSize,hiddenLayerNuerons,encoderLayers,cellType,dropout).to(device)
    
    if(optimizer == 'Adam'):
        encoderOptimizer = optim.Adam(encoder.parameters(), lr=learningRate)
        decoderOptimizer = optim.Adam(decoder.parameters(), lr=learningRate)
    else:
        encoderOptimizer = optim.NAdam(encoder.parameters(), lr=learningRate)
        decoderOptimizer = optim.NAdam(decoder.parameters(), lr=learningRate)
    
    

    for epoch in range (0,epochs):
    
        train_accuracy = 0 
        train_loss = 0 

        for batch_num, (sourceBatch, targetBatch) in enumerate(dataLoader):
                        
            encoderInitialState = encoder.getInitialState() #hiddenlayers * BatchSize * Neurons
            
            if(bidirection == "Yes"):
                reversed_batch = torch.flip(sourceBatch, dims=[1]) # reverse the batch across rows.
                sourceBatch = (sourceBatch + reversed_batch)//2 # adding reversed data to source data by averaging
            
            if(cellType == 'LSTM'):
                encoderInitialState = (encoderInitialState, encoder.getInitialState())
                
            encoder_output, encoderCurrentState = encoder(sourceBatch,encoderInitialState)
            
            
            loss = 0 # decoder starts form here
            
            sequenceLen = targetBatch.shape[1] # here name is justified

            Output = []

            randNumber = random.random()

            decoderCurrState = encoderCurrentState

            for i in range(0,sequenceLen):
                
                if(i == 0):
                    decoderInput = targetBatch[:, i].reshape(batchsize,1) # shape = (32*1)
                else:
                    if randNumber < tf_ratio:
                        decoderInput = targetBatch[:, i].reshape(batchsize, 1) # current batch is passed
                    else:
                        decoderInput = decoderInput.reshape(batchsize, 1) # prev result is passed

                decoderOutput, decoderCurrState = decoder(decoderInput,decoderCurrState)

                dummy, topIndeces = decoderOutput.topk(1)  # you will get top vales and their indices.          
                        
                decoderOutput = decoderOutput[:, -1, :] #it is just to reduce the size from (32*1*67) to (32*67)
                targetChars = targetBatch[:, i] 
                targetChars = targetChars.type(dtype=torch.long)                
                loss+=(lossFunction(decoderOutput, targetChars)) # Passing 32*67 softmax values to targetChars which has the 32*1

                decoderInput = topIndeces.squeeze().detach()  # here whatever top softmax indeces are present but converted to 1 dimension
                Output.append(decoderInput) # softmax values are attached
                
            tensor_2d = torch.stack(Output)
            Output = tensor_2d.t() #it is outside the for loop

                
            train_accuracy += (Output == targetBatch).all(dim=1).sum().item() # it is just summing up the equal words across rows

            train_loss += (loss.item()/sequenceLen)
            
            if(batch_num%200 == 0):
                print("bt:", batch_num, " loss:", loss.item()/sequenceLen)
    
            encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()
            loss.backward()
            encoderOptimizer.step()
            decoderOptimizer.step()
            
        print("train_accuracy",train_accuracy/512) #Instead of multiplying by 100 after division I directly calculated percentage
        print("train_loss",train_loss)
        wandb.log({'train_accuracy':train_accuracy/512})
        wandb.log({'train_loss':train_loss})
        validationAccuracy(encoder,decoder,batchsize,tf_ratio,cellType,bidirection)


def parse_arguments():

    args = argparse.ArgumentParser(description='Training Parameters')

    args.add_argument('-wp', '--wandb_project', type=str, default='vanillaRNN',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    
    args.add_argument('-es', '--embSize', type= int, default=64, choices = [16,32,64],help='Choice of embedding size')
    
    args.add_argument('-el', '--encoderLayers', type= int, default=5, choices = [1,5,10],help='Choice of number of encoder layers')

    args.add_argument('-dl', '--decoderLayers', type= int, default=5, choices = [1,5,10],help='Choice of number of decoder layers')

    args.add_argument('-hn', '--hiddenLayerNuerons', type= int, default=64, choices = [64,256,512],help='Choice of hidden layer neurons')
    
    args.add_argument('-ct', '--cellType', type= str, default='GRU', choices = ['GRU','LSTM','RNN'],help='Choice of cell type')

    args.add_argument('-bd', '--bidirection', type= str, default='no', choices = ['Yes','no'],help='Choice of bidirection')

    args.add_argument('-d', '--dropout', type= float, default=0.2, choices = [0,0.2,0.3],help='Choice of drop out probability')

    args.add_argument('-nE', '--epochs', type= int, default=10, choices = [10,15,20],help='Choice of epochs')

    args.add_argument('-lR', '--learnRate', type =float, default=1e-4, choices = [1e-3,1e-4],help='Choice of learnRate')

    args.add_argument('-bS', '--batchsize', type= int, default=32, choices = [32,64],help='Choice of batch size')

    args.add_argument('-opt', '--optimizer', type= str, default='Nadam', choices = ['Nadam','Adam'],help='Choice of optimizer')

    args.add_argument('-tf', '--tf_ratio', type= float, default=0.5, choices = [0.2,0.4,0.5],help='Choice of tf ratio')

    return args.parse_args()

args = parse_arguments()
wandb.init(project=args.wandb_project)

wandb.run.name=f'optimizer {args.optimizer}cellType{args.cellType}'

train(args.embSize,args.encoderLayers,args.decoderLayers,args.hiddenLayerNuerons,args.cellType,args.bidirection,args.dropout,args.epochs,args.batchsize,args.learnRate,args.optimizer,args.tf_ratio)
