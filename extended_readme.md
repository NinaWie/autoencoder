# Usage of autoencoder module:

## Get user and movie embedding

I have created a script that outputs the embeddings for both users and movies.
First, it trains both autoencoders, returns the embedding (for example, for the user embedding this is a 10000x128 matrix) and saves them as numpy files.

You need to specify the path to the train data file as an argument that is parsed:

```bash
python get_embedding.py --path=cil-collab-filtering-2019/data_train.csv
```  

In the script, I only take 11 examples as test data such that the model is trained on almost all data. The test los which is printed is thus not really useful. You can change the variable split to get a better estimate of the test loss during training.

Also, you can specify the number of epochs in the script.

## Train other models

To train a normal autoencoder to learn the user embedding, run 

```bash
python autoencoder.py
``` 

In the file you can specify parameters in the main method passing it to the constructor of the Trainer class. For example, you can also rain a variational autoencoder setting the variable in the train() method to True.

To train the autoencoders at the same time where movie and user embedding is concattenated, run

```bash
python comb_ae.py
```  
