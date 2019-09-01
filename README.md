# Attentive Neural Collaborative Filtering

## Data:

**structured_data_train.csv** contains a table with the data converted to a table with row and column index seperately, and starting at 0 instead of 1. A new column sample_id is introduced to standardise the data.

Read with:

`data = pd.read_csv("structured_data_train.csv")`

`data.set_index("sample_id", inplace=True)`

## Tabulating results

Link to the csv file for tabulating results
(https://docs.google.com/spreadsheets/d/1972m1itSfZY8vLroGclRFfVpJH_jNrcxctDq5cwNLnU/edit?usp=sharing)


# Training Models based on Embeddings
You can use different embeddings with different prediction models that use the embeddings

`python train_embedding_based --embedding_version <embed_version> --prediction_version <pred_version> --sample_weighting <sample_weighting> --optimizer <opt> --n_epochs 20 --learning_rate 0.001 --reg_constant 0.0 --embed_size 70 --experiment_name test`

<embed_version> is one of "simple" (lookup in a trainable embedding matrix), "autoencoder_embedding" (lookup in a trainable embedding matrix that was pretrained with ninas autoencoder), "attention" (attention, still coming)

<pred_version> is one of "inner_prod", "mlp1", "mlp2"

<sample_weighting> is one of "none", "user", "rating" | For weighting the samples differently

<opt> is one of "adam", "rms

Note:
* For the autoencoder embedding, only embed_size=128 is possible with the saved pre-trained weights
* For the attention embedding, further flags can be set, for example --n_context_movies and --n_context_movies_pred


In the end of training, a json file with summarizing parameters and results will be created in the folder `runs/experiment_name/`

### Additional parameters
`--make_submisssion` always makes a submission file when the validation score reaches a new best
`--crossvalidate` crossvalidation, see below for details
`--verbosity` an int between 1 and 3

`--tensorboard=1` for tensorboard
### Run tensorbaord (works from current experiment folder)
`tensorboard --logdir=train_summary_writer:summaries/train,val_summary_writer:summaries/validation`

# Checkpointing and loading pretrained
You can checkpoint a model during training by adding `--checkpoint`  
A checkpoint is saved whenever the validation accuracy reaches a new high

You can reload the weights of just the embedding or prediction model seperatly from a checkpoint. Example:

```bash
python train_embedding_based.py --embed_size 100 --embedding_version attention --prediction_version inner_prod --checkpoint --experiment_name pretrain_attention   
python train_embedding_based.py --embed_size 100 --embedding_version attention --prediction_version mlp --experiment_name use_pretrain_attention --load_embedding_weights /Users/clemens/PycharmProjects/cil/runs/pretrain_attention2019_06_16-14_24_34
```

## Pretrained attention embeddings
.. are availalbe here   
https://polybox.ethz.ch/index.php/s/tnbUvyM770AgNg9    
64 seems to be better then 128   

Example Usage (replace the path for --load_embedding_weights):
```bash
python train_embedding_based.py --embedding_config config_examples/attention.json --load_embedding_weights /Users/clemens/polybox/CIL/pretrained_attention_64 --embed_size 64 --prediction_version mlp
```
*Additional Parmeters*   
`--n_context_movies 64` (by default 64) if you make it lower, training will be faster and possibly there will be less overfitting   
`--n_context_movies_pred 128` (by default 128) the higher it is the better will be the validation accuracy (however slow), makes sense to set high for making submissions

In tensorboard under images you can look at the attention distribution over validation examples.

## Autoencoder Embeddings:

Training the autoencoder-prediction model is so far only possible with pretrained autoencoder embedidngs.
To reconstruct the results for AE embeddings in the prediction models in the report, first the AE embeddings need to be saved for each split, then cross validation needs to be run with the prediction model (mlp1, mlp2 or inner_product).

### Train autoencoder alone

The autoencoder can be trained with the following command:

User autoencoder: `python -u autoencoder/autoencoder.py --embedding=user`

Movie autoencoder: `python -u autoencoder/autoencoder.py --embedding=user`

Note: The movie autoencoder is simply trained by inverting the matrix, such that items become users and users become items.

Path flags to set:
* --save_path (default: '../Weights/model3/model'): Path to save checkpoints
* --summaries_dir (default: 'logs') : Output dir for tensorboard
* --train_data_path (default: '../data'): Path where the structured_data_train.csv file is located
* --val_part (default: '9'): Which part of the predefined split should be used as validation?

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

### Get autoencoder embeddings:

Run the same command as above, but set the flag --CV_save_embed=True. This will train the model for each split, and save the trained embeddings in the specified save path (--embedding_save_path (default: '../Weights/ae_embeddings/')). These weights were user together with the prediction model to get to the result stated in the report.

To save the embeddings per fold, run:

1) **user** autoencoder:
```bash
python autoencoder/autoencoder.py --epochs=100 --CV_save_embed=True --embedding=user --train_data_path=data/ --embedding_save_path=Weights/ae_embeddings/
```
(The default parameter values are set such that they are optimal for the user autoencoder, so no network-specific flags need to be set here)

2) **movie** autoencoder:
```bash
python autoencoder/autoencoder.py --epochs=100 --CV_save_embed=True --embedding=movie --act_hidden_e=tf.nn.sigmoid --n_hidden=128 --act_out_e=tf.nn.sigmoid --act_hidden_d=tf.nn.sigmoid --act_out_d=tf.nn.sigmoid --regularize=False --embedding_save_path=Weights/ae_embeddings/ --train_data_path=data/
```

### Run prediction model with saved embedding weights

As described above, choose the prediction model (mlp1, mlp2 or inner_product) and set the corresponding flag in the command. For the autoencoder embedding, set --embedding_model=autoencoder_embedding.

For crossvalidation, set the --crossvalidate flag and the path to load the embeddings (in the flag --load_embedding_weights_numpy):

To reconstruct the results stated in the report (CV score), use the following commands:

**AE + inner product**
```bash
python train_embedding_based.py --crossvalidate --embedding_version=autoencoder_embedding --learning_rate=0.001 --prediction_version=inner_prod --optimizer=adam --n_epochs=10 --embed_size=128 --load_embedding_weights_numpy=Weights/ae_embeddings/ --reg_constant=0.0001
```

**AE + MLP1**
```bash
python train_embedding_based.py --crossvalidate --embedding_version=autoencoder_embedding --learning_rate=0.0001 --prediction_version=mlp1 --optimizer=adam --n_epochs=10 --embed_size=128 --load_embedding_weights_numpy=Weights/ae_embeddings/
```

**AE + MLP2**
```bash
python train_embedding_based.py --crossvalidate --embedding_version=autoencoder_embedding --learning_rate=0.0001 --prediction_version=mlp2 --optimizer=adam --n_epochs=20 --embed_size=128 --load_embedding_weights_numpy=Weights/ae_embeddings/
```

### Other autoencoder variants:

Another approach was to train two autoencoders simultaniously, computing the loss as the sum over both AE reconstruction losses as well as the MSE of a rating that was yielded with a short FFNN. To run this network:

```bash
cd autoencoder
python comb_ae.py
```

It will show train and validation scores, and you can interrupt at any point and predict on the test data. See flags for possible parameters, most importantly specify --save_path (where to output submission) and --sample_submission_path (where sampleSubmission file is located).


## Weight averaging:

We have implemented weight averging as well, which was proposed in the paper Izmailov, Pavel, et al. "Averaging weights leads to wider optima and better generalization." arXiv preprint arXiv:1803.05407 (2018). For example, one can train a model, then at some point start saving the weights, and in the end average the weights.

In the file **average_weights.py** the weights of a list of checkpoints are loaded, run in the session and the averages of the weights for all variables are saved again.

The file path etc. is hard coded so far, so you need to specify which checkpoints you want to load in the file.

Performance could however not be improved with this method.
