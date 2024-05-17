# This folder belongs to vanillaRNN section
My code flows as follows :
# libraries
I have imported all the libraries.
# wandb
I have integrated wandb with projectname = 'vanillaRNN'
# GPU 
I have integreated GPU if available
# Data loading preprocessing 
I have loaded the different data sets (Train, Validation and Test)
I have preprocessed the input data by using the functions  **pre_processing(train_input,train_output)**. It simply takes the input data and converts it into sequence of numbers for training purpose.
I have preprocessed the input data by using the functions  **pre_processing_validation(val_input,val_output)**. It simply takes the validation data and converts it into sequence of numbers for testing purpose.

# Data loading
I have used **MyDataset** class to prepare the data loader for all catogories of data.
It is helpful in loading both the input and output of the data batch-wise 
Function **dataLoaderFun(..)** loads the dataloader by implementing MyDataSet class depending on category of data ( train,validation and test data)

# Accuracy Calculation
function **validationAccuracy(encoder,decoder,batchsize,tf_ratio,cellType,bidirection)** calculates the validation accuracy after the model training. It calculates the validation loss (Entropy loss) and validation accuracy (in percentage)
function **train(...)** caculates the training loss (Entropy loss) and training accuracy (in precentage) at each epoch

# Encoder and Decoder
Used Encoder and Decoder inbuilt classes from nn.Module. Forward and backward propogations are subjective to kind of cell type. 

# training
Function **train(....)** trains the model. Model is trained on different configurations passed by the parser parameters. On each epoch it prints the train acccuracy, train loss, validation accuracy and validation loss. 

**trainVanillaRNN.py**

  -- I have used the parse_arguments from parser library to execute this trainVanillaRNN.py file.
  -- It can be executed by the command ``` python trainVanillaRNN.py ```. It is executed by default values. 
  -- Command (**-- parameterName**) following above execution command can be used to test with other values than default values.

-The script trainVanillaRNN.py accepts several command line arguments to configure the training process:

-wp, --wandb_project: Project name used to track experiments in WandB dashboard (default: vanillaRNN)
-es, --embSize: Choice of embedding size (default: 64, choices: [16, 32, 64])
-el, --encoderLayers: Choice of number of encoder layers (default: 5, choices: [1, 5, 10])
-dl, --decoderLayers: Choice of number of decoder layers (default: 5, choices: [1, 5, 10])
-hn, --hiddenLayerNuerons: Choice of hidden layer neurons (default: 512, choices: [64, 256, 512])
-ct, --cellType: Choice of cell type (default: RNN, choices: ['GRU', 'LSTM', 'RNN'])
-bd, --bidirection: Choice of bidirection (default: no, choices: ['Yes', 'no'])
-d, --dropout: Choice of dropout probability (default: 0.3, choices: [0, 0.2, 0.3])
-nE, --epochs: Choice of epochs (default: 10, choices: [10, 15, 20])
-lR, --learnRate: Choice of learning rate (default: 1e-4, choices: [1e-3, 1e-4])
-bS, --batchsize: Choice of batch size (default: 32, choices: [32, 64])
-opt, --optimizer: Choice of optimizer (default: Adam, choices: ['Nadam', 'Adam'])
-tf, --tf_ratio: Choice of teacher forcing ratio (default: 0.5, choices: [0.2, 0.4, 0.5])
  -- wandb can be integrated by giving your activation key to visualize the accuracies
  
  **VanillaRNN.ipynb.**
  
  -- In place of parsers I have integrated with wandb parameters and ran the sweeps in ipynb file.
  --  You can run by integrating your wandb account by activation key to visualize the accuarcies.




