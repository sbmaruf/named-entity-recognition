
# cross-lingual-ne

## Phase 1

This is a sample feed forward Nural Network.

+ **[Done]** Build a early system that trains. 
+ **[Done]** Add multiple hidden layer

## Phase 2

Build a full evaluation cycle with single layer MLP.

+ **[DONE]** Visualize graph with tensorboard.
+ **[Done]** Add optparset based hypeparameter control feature.
+ **[Done]** Evaluate model by conlleval.
+ **[Done]** Add early stop protocol.
+ **[Done]** Add f-measure based evaluation feature.
+ **[XXXX]** Add hyper parameter tuning shell script.
+ **[XXXX]** Tweak hyperparameter by tensorboard.
+ **[Done]** Complete the preprocessing
+ **[XXXX]** Apply embedding feature of tensorflow
+ **[DONE]** Add gradient clipping code
+ **[DONE]** Add skip-n-gram initialisation

**Results :** Model is learning. Average f-measure : around 58 


## Phase 3

+ **[DONE]** Implement the CNN model with same structure
+ **[DONE]** Implement additional parameters for opt parser
+ **[DONE]** Implement the model with a model class.
+ **[DONE]** Tweak different hyperparameter 
+ **[TODO]** Batch Normalization 

**Results :** Model is learning. Average f-measure : **testing ....** 



## Guide to use the Phase 3 model
 
to run the model, 
```
cd ./Code/Phase_3
python setup.py
```

All the parameters have default values. You can change them by using the argument from the command lne. Say, if you want to change the window size, you can do following,

```
cd ./Code/Phase_3
python setup.py -w 6
```

We have 10 different optimizer that we can use. We mark them 0-9. For Adam you need to pass 1. To see all the options for all parameters,
```
cd ./Code/Phase_3
python setup.py --help
```

For pre-embedding download the [skip-n-gram dataset](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view) and place it to **Data** folder in your directory.

