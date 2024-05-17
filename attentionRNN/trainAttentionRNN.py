import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset, DataLoader
import random
import argparse
# imported libraries

#!pip install wandb 
import wandb
wandb.login()
wandb.init(project ='AttentionRNN')
# wandb integration 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # gpu integration if available

train_csv = "/kaggle/input/telugu/tel/tel_train.csv"
test_csv = "/kaggle/input/telugu/tel/tel_test.csv"
val_csv = "/kaggle/input/telugu/tel/tel_valid.csv" # loading different categories of data

train_data = pd.read_csv(train_csv, header=None)
train_input = train_data[0].to_numpy()
train_output = train_data[1].to_numpy()
val_data = pd.read_csv(val_csv,header = None)
val_input = val_data[0].to_numpy()
val_output = val_data[1].to_numpy()
test_data = pd.read_csv(test_csv,header= None)
test_input = test_data[0].to_numpy()
test_output = test_data[1].to_numpy() #converting data into numpy datatype


def pre_processing(train_input,train_output): # train_input = english word train_output = Telugu word
    data = {
    "all_characters" : [], # stores all unique Dnglish characters
    "char_num_map" : {}, # character of English to number mapping
    "num_char_map" : {}, # number to character mapping of English letters
    "source_charToNum": torch.zeros(len(train_input),30, dtype=torch.int, device=device),
    "source_data" : train_input,
        
    "all_characters_2" : [], # stores all unique Telugu characters
    "char_num_map_2" : {}, # character of Telugu to number mapping
    "num_char_map_2" : {}, # number to character mapping of Telugu letters
    "val_charToNum": torch.zeros(len(train_output),23, dtype=torch.int, device=device),
    "target_data" : train_output,
    "source_len" : 0,
    "target_len" : 0
    }
    k = 0 
    l = 0
    for i in range(0,len(train_input)):
        train_input[i] = "{" + train_input[i] + "}"*(29-len(train_input[i]))
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
            
            charToNum.append(index)  # above code converts train_input characters into numbers
            
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
                
            charToNum1.append(index)
            
        my_tensor1 = torch.tensor(charToNum1,device = device)
        data["val_charToNum"][k] = my_tensor1
        
        k+=1
    
    data["source_len"] = len(data["all_characters"])
    data["target_len"] = len(data["all_characters_2"])
        
    return data # it returns character to number converted data
    
    
data = pre_processing(copy.copy(train_input),copy.copy(train_output)) 



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
        
        encoder_initial_state = encoder.getInitialState() #hiddenlayers * BatchSize * Neurons = encoderInitialState with 0s
            
        if(bidirection == "Yes"):
            reversed_batch = torch.flip(sourceBatch, dims=[1]) # reverse the batch across rows.
            sourceBatch = (sourceBatch + reversed_batch)//2 # adding reversed data to source data by averaging

        if(cellType == 'LSTM'):
            encoder_initial_state = (encoder_initial_state, encoder.getInitialState())

        encoderStates , encoderOutput = encoder(sourceBatch,encoder_initial_state)

        decoderCurrentState = encoderOutput # this selects the last state from encoder states

        encoderFinalLayerStates = encoderStates[:, -1, :, :]

        attentions = []

        loss = 0 # decoder starts
            
        outputSeqLen = targetBatch.shape[1] # here  name is justified.
        
        Output = []
    
        randNumber = random.random()


        for i in range(0,outputSeqLen):
            
            if(i == 0):
                decoderCurrentInput = torch.full((batchsize,1),0, device=device)

            else:
                if randNumber < tf_ratio:
                    decoderCurrentInput = targetBatch[:, i].reshape(batchsize, 1)
                else:
                    decoderCurrentInput = decoderCurrentInput.reshape(batchsize, 1)

            decoderOutput, decoderCurrentState, attentionWeights = decoder(decoderCurrentInput, decoderCurrentState, encoderFinalLayerStates)

            attentions.append(attentionWeights)
            dummy, topi = decoderOutput.topk(1) #  get top vales and their indices. 
    
            decoderOutput = decoderOutput[:, -1, :]
            curr_target_chars = targetBatch[:, i] 
            curr_target_chars = curr_target_chars.type(dtype=torch.long)
            loss+=(lossFunction(decoderOutput, curr_target_chars))

            decoderCurrentInput = topi.squeeze().detach()
            Output.append(decoderCurrentInput) # softmax values are attached  


        validation_loss += (loss.item()/outputSeqLen)  # cross entropy loss calculations 
        
        tensor_2d = torch.stack(Output)
        Output = tensor_2d.t()
        
        validation_accuracy += (Output == targetBatch).all(dim=1).sum().item() # it  just summing up the equal words along the row
        
        if(batch_num%40 == 0):
            print("bt:", batch_num, " loss:", loss.item()/outputSeqLen)

    encoder.train() # bringing back the model into training phase
    decoder.train()
    print("val_accuracy",validation_accuracy/40.96) #Instead of multiplying by 100 after division I directly calculated percentage
    print("val_loss",validation_loss)
    wandb.log({'val_accuracy':validation_accuracy/40.96})
    wandb.log({'val_loss':validation_loss})


class Attention(nn.Module): #Attention class to implemnt the additive attention in vanillaRNN model
    def __init__(self, hiddenSize):
        super(Attention, self).__init__()
        self.Watt = nn.Linear(hiddenSize, hiddenSize)
        self.Uatt = nn.Linear(hiddenSize, hiddenSize)
        self.Vatt = nn.Linear(hiddenSize, 1)

    def forward(self, query, keys):
        calc = self.Watt(query) + self.Uatt(keys)
        scores = self.Vatt(torch.tanh(calc))
        scores = scores.squeeze().unsqueeze(1)
        weights = F.softmax(scores, dim=0)
        weights = weights.permute(2,1,0)
        keys = keys.permute(1,0,2)
        context = torch.bmm(weights, keys)
        return context, weights
    
class Encoder(nn.Module):
    
    def __init__(self,inputDim,embSize,encoderLayers,hiddenLayerNuerons,cellType,batch_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(inputDim, embSize)
        self.encoderLayers = encoderLayers
        self.hiddenLayerNuerons = hiddenLayerNuerons
        self.batch_size = batch_size
        self.cellType = cellType
        if(cellType=='GRU'):
            self.rnn = nn.GRU(embSize,hiddenLayerNuerons,num_layers=encoderLayers, batch_first=True)
        elif(cellType=='RNN'):
            self.rnn = nn.RNN(embSize,hiddenLayerNuerons,num_layers=encoderLayers, batch_first=True)
        else:
            self.rnn = nn.LSTM(embSize,hiddenLayerNuerons,num_layers=encoderLayers, batch_first=True)

    def forward(self,sourceBatch,encoderCurrState):
        sequenceLength = len(sourceBatch[0])
        encoderStates = torch.zeros(sequenceLength,self.encoderLayers,self.batch_size,self.hiddenLayerNuerons,device=device)
        for i in range(0,sequenceLength):
            currInput = sourceBatch[:,i].reshape(self.batch_size,1)
            dummy , encoderCurrState = self.statesCalculation(currInput,encoderCurrState)
            if(self.cellType == 'LSTM'):
                encoderStates[i] = encoderCurrState[1]
            else: 
                encoderStates[i] = encoderCurrState

        return encoderStates ,encoderCurrState


    def statesCalculation(self, currentInput, prevState): #forward pass of encoder
        embdInput = self.embedding(currentInput)
        output, prev_state = self.rnn(embdInput, prevState)
        return output, prev_state
    
    def getInitialState(self):
        return torch.zeros(self.encoderLayers,self.batch_size,self.hiddenLayerNuerons, device=device)
    
class Decoder(nn.Module):
    def __init__(self,outputDim,embSize,hiddenLayerNuerons,decoderLayers,cellType,dropout_p):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(outputDim, embSize)
        self.cellType=cellType
        if(cellType == 'GRU'): # changed here
            self.rnn = nn.GRU(embSize+hiddenLayerNuerons,hiddenLayerNuerons,num_layers=decoderLayers, batch_first=True)
        elif(cellType == 'RNN'):
            self.rnn = nn.RNN(embSize+hiddenLayerNuerons,hiddenLayerNuerons,num_layers=decoderLayers, batch_first=True)
        else:
            self.rnn = nn.LSTM(embSize+hiddenLayerNuerons,hiddenLayerNuerons,num_layers=decoderLayers, batch_first=True)
            
        self.fc = nn.Linear(hiddenLayerNuerons, outputDim) # it is useful for mapping the calculation to vocabularu
        self.softmax = nn.LogSoftmax(dim=2) #output is in 3rd column 
        self.dropout = nn.Dropout(dropout_p)
        self.attention = Attention(hiddenLayerNuerons).to(device)

    def forward(self, current_input, prev_state,encoder_final_layers): #forward pass of decoder
        if(self.cellType == 'LSTM'):
            context , attn_weights = self.attention(prev_state[1][-1,:,:], encoder_final_layers)
        else:
            context , attn_weights = self.attention(prev_state[-1,:,:], encoder_final_layers)
        embd_input = self.embedding(current_input)
        curr_embd = F.relu(embd_input)
        input_gru = torch.cat((curr_embd, context), dim=2)
        output, prev_state = self.rnn(input_gru, prev_state)
        output = self.dropout(output)
        output = self.softmax(self.fc(output)) 
        return output, prev_state, attn_weights
    
data = pre_processing(copy.copy(train_input),copy.copy(train_output)) # training data 

def dataLoaderFun(dataName,batch_size): #dataName : type of data
    if(dataName == 'train'): # returns the Data Loader depending on type of data 
        dataset = MyDataset(data["source_charToNum"],data['val_charToNum'])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = MyDataset(data2["source_charToNum"],data2['val_charToNum'])
        return  DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(embSize,encoderLayers,decoderLayers,hiddenLayerNuerons,cellType,bidirection,dropout,epochs,batchsize,learningRate,optimizer,tf_ratio):

    dataLoader = dataLoaderFun("train",batchsize) # dataLoader depending on train or validation
    
    encoder = Encoder(data["source_len"],embSize,encoderLayers,hiddenLayerNuerons,cellType,batchsize).to(device)
    decoder = Decoder(data["target_len"],embSize,hiddenLayerNuerons,encoderLayers,cellType,dropout).to(device)
    
    if(optimizer == 'Adam'):
        encoderOptimizer = optim.Adam(encoder.parameters(), lr=learningRate)
        decoderOptimizer = optim.Adam(decoder.parameters(), lr=learningRate)
    else:
        encoderOptimizer = optim.NAdam(encoder.parameters(), lr=learningRate)
        decoderOptimizer = optim.NAdam(decoder.parameters(), lr=learningRate)
    
    lossFunction = nn.NLLLoss()

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
                
            encoderStates,EcoderOutput= encoder(sourceBatch,encoderInitialState)

            encoderFinalLayerStates = encoderStates[:, -1, :, :] # this selects the hidden top layers from each sequence

            decoderCurrentState = EcoderOutput            
            attentions = []
            loss = 0 # decoder starts form 
            
            outputSeqLen = targetBatch.shape[1] # here you will get as name justified. 40
            
            Output = [] # to collect the predicted data
            
            randNumber = random.random()

            

            for i in range(0,outputSeqLen):

                if(i == 0):
                    decoderCurrentInput = torch.full((batchsize,1),0, device=device)
                else:
                    if randNumber < tf_ratio:
                        decoderCurrentInput = targetBatch[:, i].reshape(batchsize, 1) # applying teacher force to give current data as input
                    else:
                        decoderCurrentInput = decoderCurrentInput.reshape(batchsize, 1) # passing previous data

                decoderOutput, decoderCurrentState, attentionWeights = decoder(decoderCurrentInput, decoderCurrentState, encoderFinalLayerStates)
                
                dummy, topIndeces = decoderOutput.topk(1) # collecting top softmax indeces

                decoderOutput = decoderOutput[:, -1, :]
                curr_target_chars = targetBatch[:, i] 
                curr_target_chars = curr_target_chars.type(dtype=torch.long)
                loss+=(lossFunction(decoderOutput, curr_target_chars)) #cross entropy calculation

                
                decoderCurrentInput = topIndeces.squeeze().detach()
                Output.append(decoderCurrentInput)

                attentions.append(attentionWeights)

            tensor_2d = torch.stack(Output)
            Output = tensor_2d.t() 
            train_accuracy += (Output == targetBatch).all(dim=1).sum().item() # summing up accurate words produced 

            train_loss += (loss.item()/outputSeqLen) # batchwise loss entropy summations
            
            if(batch_num%200 == 0):
                print("bt:", batch_num, " loss:", loss.item()/outputSeqLen)
                
            encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()
            loss.backward()
            encoderOptimizer.step()
            decoderOptimizer.step()
            
        print("train_accuracy",train_accuracy/512) # Instead of multiplying by 100 after division I directly calculated percentage
        print("train_loss",train_loss)
        wandb.log({'train_accuracy':train_accuracy/512})
        wandb.log({'train_loss':train_loss})
        validationAccuracy(encoder,decoder,batchsize,tf_ratio,cellType,bidirection)

def parse_arguments():

    args = argparse.ArgumentParser(description='Training Parameters')

    args.add_argument('-wp', '--wandb_project', type=str, default='AttentionRNN',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    
    args.add_argument('-es', '--embSize', type= int, default=64, choices = [16,32,64],help='Choice of embedding size')
    
    args.add_argument('-el', '--encoderLayers', type= int, default=5, choices = [1,5,10],help='Choice of number of encoder layers')

    args.add_argument('-dl', '--decoderLayers', type= int, default=5, choices = [1,5,10],help='Choice of number of decoder layers')

    args.add_argument('-hn', '--hiddenLayerNuerons', type= int, default=512, choices = [64,256,512],help='Choice of hidden layer neurons')
    
    args.add_argument('-ct', '--cellType', type= str, default='RNN', choices = ['GRU','LSTM','RNN'],help='Choice of cell type')

    args.add_argument('-bd', '--bidirection', type= str, default='no', choices = ['Yes','no'],help='Choice of bidirection')

    args.add_argument('-d', '--dropout', type= float, default=0.3, choices = [0,0.2,0.3],help='Choice of drop out probability')

    args.add_argument('-nE', '--epochs', type= int, default=10, choices = [10,15,20],help='Choice of epochs')

    args.add_argument('-lR', '--learnRate', type =float, default=1e-4, choices = [1e-3,1e-4],help='Choice of learnRate')

    args.add_argument('-bS', '--batchsize', type= int, default=32, choices = [32,64],help='Choice of batch size')

    args.add_argument('-opt', '--optimizer', type= str, default='Adam', choices = ['Nadam','Adam'],help='Choice of optimizer')

    args.add_argument('-tf', '--tf_ratio', type= float, default=0.5, choices = [0.2,0.4,0.5],help='Choice of tf ratio')

    return args.parse_args()

args = parse_arguments()
wandb.init(project=args.wandb_project)

wandb.run.name=f'optimizer {args.optimizer}cellType{args.cellType}'

train(args.embSize,args.encoderLayers,args.decoderLayers,args.hiddenLayerNuerons,args.cellType,args.bidirection,args.dropout,args.epochs,args.batchsize,args.learnRate,args.optimizer,args.tf_ratio)



