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
  -- It can be executed by appling !python trainVanillaRNN.py --(parameters that are supported as choices in my trainVanillaRNN.py file)
  -- wandb can be integrated by giving your activation key to visualize the accuracies
  -- (**-- parameterName**) command be used to test with other values than default values.
  
  **trainVanillaRNN.ipynb.**
  
  -- In place of parsers I have integrated with wandb parameters and ran the sweeps in ipynb file.
  --  You can run by integrating your wandb account by activation key to visualize the accuarcies.




