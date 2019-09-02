# Autoencoder for collaborative filtering

This codebase show how to train an autoencoder (and different variations) to do matrix completion. There are several tutorials online that perfectly explain how autoencoders can be used for collaborative filtering, so I do not want to explain it in detail here. Basically, the input is a sparse matrix, for example containing the ratings of users (rows) for different movies (columns), and the goal is to infer the ratings of the gaps, i.e. the ratings a user will most likely give for a movie he has not seen yet. An autoencoder is one way to do this inference since it can find underlying features by compressing the user or movies in a bottleneck fashion. 

In the [tutorial](Autoencoder_tutorial.ipynb) notebook, a very simple version is shown that trains a simple three-layered model to infer a user embedding.

Running autoencoder.py offers a lot more options than the notebook, see below

## Train autoencoder

### Input data

In the data folder, example data can be found. The first column indicates the position in the user-movie-matrix (e.g. r_1_c_12 would be user 1 for movie 12) and the second column is the corresponding rating.

### Train

The autoencoder can be trained with the following command:

User autoencoder: `python -u autoencoder/autoencoder.py --embedding=user`

Movie autoencoder: `python -u autoencoder/autoencoder.py --embedding=movie`

Note: The movie autoencoder is simply trained by inverting the matrix, such that items become users and users become items.

Path flags to set:
* --save_path (default: 'None'): Path to save checkpoints (if None than the model is not saved during training)
* --summaries_dir (default: 'logs') : Output dir for tensorboard
* --train_data_path (default: 'data'): Path where the train data file is located
* --split (defaut:10): how much of the data is taken as test data (10 would correspond to 10 percent, 20 would be 5 percent)

Network specific flags to set:

* --act_hidden_d (default: 'tf.nn.relu')
* --act_hidden_e (default: 'tf.nn.relu')
* --act_out_d (default: 'None')
* --act_out_e (default: 'tf.nn.relu')
* --batch_size (default: '8')
* --dropout_rate  (default: '1.0')
* --epochs (default: '200')
* --regularize (default: 'true')
* --lamba (default: '0.001')
* --n_embed (default: '128')
* --n_hidden (default: '512')
* --learning_rate (default: '0.0001')

### Output

The output is the filled matrix, so each cell in the user - movie-matrix is filled in now. The train and test loss which is printed refers to the RMSE of predicted ratings and actual ratings.

## Variations in the architecture and output

### Sparse AE

One method that was proposed to achieve better feature extraction with autoencoders is to restrict the number of non-zero entries of the embedding layer. This way, for each user (if doing a user embedding) the number of active features is restricted. This is implemented here with two flags:

* --k_sparse (default: False
* --k_sparse_factor (default:20)

So for example, running  `python -u autoencoder/autoencoder.py --embedding=user --k_sparse=True --k_sparse_factor=20` would predict the ratings based on a 20-sparse embedding layer.

### Output the embeddings:

Instead of returning the predicted ratings, it is also possible to get the embedding layer itself. So for example, if the size of the embedding layer is 128, and 10000 users are trained to be represented in that lower dimensional space, this would yield a 10000x128 matrix. This matrix can be used for further processing.

To save the embeddings per fold, specify the following flags:

* --return_embedding (default: False)
* --embedding_save_path (default: results/)

For example, the following command would train a movie autoencoder for 100 epochs, and in the end save a numpy arrray to the results directory which contains a 1000*256 array. 

```bash
python autoencoder/autoencoder.py --epochs=100 --return_embedding=True --embedding_save_path=results --n_embed=256
```

### Variational AE

A VAE is trained to reconstruct the data as a normal AE and at the same time maintain a smooth latent space, such that new data can be sampled. For the task of collaborative filtering, this can be useful to augment the data, like for example generating new users and training on those as well. 

To train a VAE and save the genererated samples, specify the flag

* --VAE (default:False)

At this point it is only possible to change number of samples, save location and so on in the code.


### Combined autoencoder

Another approach was to train two autoencoders simultaniously, one embedding the movies and one embedding the users. Then, the embedding layers of both are concatenated the resulting layer is passed through a short feed forward network to yield a single scalar (the predicted rating). The loss is simply the sum over both AE reconstruction losses (reconstruction from user embedding and reconstruction from movie embedding) and the MSE of the predicted rating after the FF network.

To train this special autoencoder, run

```bash
cd src
python comb_ae.py
```  

It will output train and validation scores in each epoch, and you can interrupt at any point and predict the ratings on the test data. Since now there are three different options to predict the ratings (via the user AE, via the movie AE or via the concatenated output), you can specify which one you want to use. 

Note that this AE will take much longer to train.

See flags in the file for possible parameters, most are similar as for the normal AE.
