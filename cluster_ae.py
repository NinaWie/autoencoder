import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import tensorflow as tf
from utils import *
from data_preprocess import split_data, load_data, get_datasplit, matrix_data_omega,load_data_omega, make_submission
from model import AEmodel
import os

### Hyperparameters

class Trainer():
    def __init__(self, nr_movies=1000, act_hidden_d="tf.nn.relu", act_out_d="None", act_out_e="tf.nn.relu", act_hidden_e="tf.nn.relu", n_hidden=512, n_embed=128, nr_users=10000, epochs=200, learning_rate=0.0001):
        tf.flags.DEFINE_string("path", "cil-collab-filtering-2019/data_train.csv", "path to train data")
        tf.flags.DEFINE_integer("batch_size", 8, "")
        tf.flags.DEFINE_integer("EPOCHS", epochs, "")
        tf.flags.DEFINE_integer("nr_movies", nr_movies, "Movies in dataset")
        tf.flags.DEFINE_integer("nr_users", nr_users, "Users in dataset")
        tf.flags.DEFINE_float("learning_rate", learning_rate, "")
        tf.flags.DEFINE_float("lamba", 0.001, "")
        tf.flags.DEFINE_bool("regularize", True, "")
        tf.flags.DEFINE_bool("DAE", False, "Denoising autoencoder (add noise to inputs)")
        tf.flags.DEFINE_integer("rbm_size", 128, "nr of hidden neurons of restricted boltzmann machine")
        tf.flags.DEFINE_integer("n_embed", n_embed, "")
        tf.flags.DEFINE_integer("n_hidden", n_hidden, "")
        tf.flags.DEFINE_string("act_hidden_e", act_hidden_e, "")
        tf.flags.DEFINE_string("act_out_e", act_out_e, "")
        tf.flags.DEFINE_string("act_hidden_d", act_hidden_d, "")
        tf.flags.DEFINE_string("act_out_d", act_out_d, "")
        tf.flags.DEFINE_integer("k_sparse", 20, "")
        tf.flags.DEFINE_float("dropout_rate", 1.0, "")
        tf.flags.DEFINE_string("summaries_dir", "logs", "")
        tf.flags.DEFINE_string("save_path", "../Weights/model1/model", "")

        self.FLAGS = tf.flags.FLAGS

    def del_all_flags(self, FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    def train(self, train_matrix, test_matrix, variational=False, return_embedding=False):

        tf.reset_default_graph()
        # placeholders
        with tf.name_scope("Input"):
            x_mask = tf.placeholder(tf.float32, (None, self.FLAGS.nr_movies), name="maskInput")
            x = tf.placeholder(tf.float32, (None, self.FLAGS.nr_movies), name = "input")
            print(x)
            beta = tf.placeholder(tf.float32, name="beta")
            rbm_inp = tf.placeholder(tf.float32, (None, self.FLAGS.rbm_size), name = "input2")
            drop_rate = tf.placeholder(tf.float32, name="dropoutPlaceholder")
            learning_rate = tf.placeholder(tf.float32, shape=[])

        with tf.name_scope("model"):
            model = AEmodel(self.FLAGS.nr_users, self.FLAGS.nr_movies)

            # forward, loss and train
            if not variational:
                Z = model.encode(x, n_hidden=self.FLAGS.n_hidden, n_embed=self.FLAGS.n_embed, act_hidden=eval(self.FLAGS.act_hidden_e), act_out=eval(self.FLAGS.act_out_e), keep_prob=drop_rate, variational=variational)
                # Z = model.sparse(Z, k=self.FLAGS.k_sparse)
                outputs = model.decode(Z, n_hidden=self.FLAGS.n_hidden, act_hidden=eval(self.FLAGS.act_hidden_d), act_out=eval(self.FLAGS.act_out_d), keep_prob=drop_rate)
                loss_op = model.masked_loss(x_mask, outputs)
                tf.summary.scalar("loss", loss_op)
            else:
            ## variational:
                Z_mu, Z_logvar = model.encode(x, n_hidden=self.FLAGS.n_hidden, n_embed=self.FLAGS.n_embed, act_hidden=eval(self.FLAGS.act_hidden_e), act_out=eval(self.FLAGS.act_out_e), keep_prob=drop_rate, variational=variational)
                Z = model.reparameterize(Z_mu, Z_logvar)
                # Z_new = tf.concat([Z_mu,Z],1) # ADDED: Concatenate embedding with the random value in latent space
                outputs = model.decode(Z, n_hidden=self.FLAGS.n_hidden, act_hidden=eval(self.FLAGS.act_hidden_d), act_out=eval(self.FLAGS.act_out_d), keep_prob=drop_rate)
                # loss
                loss_recon = model.masked_loss(x_mask, outputs)
                loss_kld = model.kld_loss(Z_mu, Z_logvar)
                loss_op = loss_recon + 0.01 * loss_kld # BETA
                tf.summary.scalar("loss", loss_op)
                tf.summary.scalar("loss_recon", loss_recon)
                tf.summary.scalar("loss_kld",loss_kld)

        with tf.name_scope("lossandtrain"):
            if self.FLAGS.regularize==True:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                loss_op_reg = loss_op + l2_loss * self.FLAGS.lamba
            else:
                loss_op_reg = loss_op

            # train_op = tf.train.RMSPropOptimizer(self.FLAGS.learning_rate).minimize(loss_op_reg)
            train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss_op_reg) #GradientDescentOptimizer


        saver = tf.train.Saver(max_to_keep=100)
        min_lr = 0.00001
        max_lr = 0.0001
        iters_per_epoch = self.FLAGS.nr_users//self.FLAGS.batch_size
        cycle_length = 5*iters_per_epoch
        start_epoch = 80
        print("CYCLE LENGTH",cycle_length)
        ### Run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_losses=[]

            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_dir, sess.graph)

            matrix = train_matrix.copy()
            # vae_matrix = np.load("VAE_sample.npy")
            # matrix = np.concatenate((matrix, vae_matrix), axis=0)
            print("training matrix vae shape", matrix.shape)
            # matrix = fill_zeros(train_matrix, val_new=0.6) # val_new = mean rating

            for epoch in range(self.FLAGS.EPOCHS):
                if self.FLAGS.DAE:
                    matrix = noisy_labels(train_matrix)

                ## TRAIN
                train_losses=[]
                try:
                    for user in range(iters_per_epoch):
                        global_step = ((epoch-start_epoch)*iters_per_epoch)+user
                        if epoch>=start_epoch:
                            lr = max_lr-((global_step%cycle_length)/cycle_length)*(max_lr-min_lr)
                        else:
                            lr = 0.0001
                        sta = user*self.FLAGS.batch_size
                        end = (user+1)*self.FLAGS.batch_size
                        loss, _  = sess.run([loss_op, train_op], {x: matrix[sta:end], x_mask: matrix[sta:end], beta:epoch/self.FLAGS.EPOCHS, drop_rate: self.FLAGS.dropout_rate, learning_rate:lr}) # rbm_inp:rbm_transformed[sta:end]
                        train_losses.append(loss)

                        if epoch>=start_epoch and (global_step)%cycle_length==cycle_length-1:
                            saver.save(sess, self.FLAGS.save_path+str(epoch))
                            print("saved checkpoints in", self.FLAGS.save_path+str(epoch))
                            print("saved at learning rate", lr)
                except KeyboardInterrupt:
                    print("Interrupted")
                    break

                ## TEST
                if not variational:
                    test_loss, summary = sess.run([loss_op, merged_summary], {x: train_matrix, x_mask: test_matrix, drop_rate:1.0}) # rbm_inp:rbm_transformed,
                    print("train", np.mean(train_losses), "test", test_loss)
                else:
                ## Variational:
                    test_loss_kld, test_loss_recon, test_loss, summary = sess.run([loss_kld, loss_recon, loss_op, merged_summary], {x: train_matrix, x_mask: test_matrix, beta:epoch/self.FLAGS.EPOCHS, drop_rate:1.0}) # rbm_inp:rbm_transformed,
                    print("train", np.mean(train_losses), "test", test_loss, "test_kld", test_loss_kld, "test_recon", test_loss_recon)

                test_losses.append(test_loss)

                summary_writer.add_summary(summary, epoch)


            ## SAMPLE NEW DATA
            if variational:
                nr_samples = 1000
                Z_sample = np.random.normal(size=(nr_samples, self.FLAGS.n_embed)) # changed 2*
                augment = sess.run(outputs, feed_dict={Z:Z_sample, drop_rate:1.0}) # changed Z_new
                print(augment.shape)
                np.save("VAE_sample.npy", augment)

            if return_embedding:
                embed = sess.run(Z,{x: matrix, drop_rate:1.0})
                sess.close()
                self.del_all_flags(self.FLAGS)
                return embed
            else:
            # run on full set to get final reconstructed matrix
                out_matrix = sess.run(outputs,{x: matrix, drop_rate:1.0}) #rbm_inp:rbm_transformed,
                sess.close()
                self.del_all_flags(self.FLAGS)
                return out_matrix

    def restore(self, train_matrix, test_matrix, variational=False, return_embedding=False):

        tf.reset_default_graph()
        # placeholders
        with tf.name_scope("Input"):
            x_mask = tf.placeholder(tf.float32, (None, self.FLAGS.nr_movies), name="maskInput")
            x = tf.placeholder(tf.float32, (None, self.FLAGS.nr_movies), name = "input")
            print(x)
            beta = tf.placeholder(tf.float32, name="beta")
            rbm_inp = tf.placeholder(tf.float32, (None, self.FLAGS.rbm_size), name = "input2")
            drop_rate = tf.placeholder(tf.float32, name="dropoutPlaceholder")

        with tf.name_scope("model"):
            model = AEmodel(self.FLAGS.nr_users, self.FLAGS.nr_movies)

            # forward, loss and train
            if not variational:
                Z = model.encode(x, n_hidden=self.FLAGS.n_hidden, n_embed=self.FLAGS.n_embed, act_hidden=eval(self.FLAGS.act_hidden_e), act_out=eval(self.FLAGS.act_out_e), keep_prob=drop_rate, variational=variational)
                # Z = model.sparse(Z, k=self.FLAGS.k_sparse)
                outputs = model.decode(Z, n_hidden=self.FLAGS.n_hidden, act_hidden=eval(self.FLAGS.act_hidden_d), act_out=eval(self.FLAGS.act_out_d), keep_prob=drop_rate)
                loss_op = model.masked_loss(x_mask, outputs)
                tf.summary.scalar("loss", loss_op)
            else:
            ## variational:
                Z_mu, Z_logvar = model.encode(x, n_hidden=self.FLAGS.n_hidden, n_embed=self.FLAGS.n_embed, act_hidden=eval(self.FLAGS.act_hidden_e), act_out=eval(self.FLAGS.act_out_e), keep_prob=drop_rate, variational=variational)
                Z = model.reparameterize(Z_mu, Z_logvar)
                # Z_new = tf.concat([Z_mu,Z],1) # ADDED: Concatenate embedding with the random value in latent space
                outputs = model.decode(Z, n_hidden=self.FLAGS.n_hidden, act_hidden=eval(self.FLAGS.act_hidden_d), act_out=eval(self.FLAGS.act_out_d), keep_prob=drop_rate)
                # loss
                loss_recon = model.masked_loss(x_mask, outputs)
                loss_kld = model.kld_loss(Z_mu, Z_logvar)
                loss_op = loss_recon + 0.01 * loss_kld # BETA
                tf.summary.scalar("loss", loss_op)
                tf.summary.scalar("loss_recon", loss_recon)
                tf.summary.scalar("loss_kld",loss_kld)

        with tf.name_scope("lossandtrain"):
            if self.FLAGS.regularize==True:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                loss_op_reg = loss_op + l2_loss * self.FLAGS.lamba
            else:
                loss_op_reg = loss_op

            train_op = tf.train.RMSPropOptimizer(self.FLAGS.learning_rate).minimize(loss_op_reg)

        ### FOR MULTIPLE RESTORE
        restored_weights = list()
        graph = tf.Graph()
        all_vars = tf.trainable_variables()
        for saved_epoch in np.sort(np.unique([float(i[5:8]) for i in os.listdir(self.FLAGS.save_path[:-5])])).astype(int):
            print(saved_epoch)
            session = tf.Session()
            saver = tf.train.Saver()
            saver.restore(session, self.FLAGS.save_path+str(saved_epoch))
            values = session.run(all_vars)
            restored_weights.append(values)
            session.close()

        # session2 = tf.Session()
        # session3 = tf.Session()
        session3 = tf.Session()

        # Omitted code: Restore session1 and session2.
        # Optionally initialize session3.
        # saver = tf.train.Saver()
        # saver.restore(session1, "../Weights/model1/model")
        # saver = tf.train.Saver()
        # saver.restore(session2, "../Weights/model3/model")
        #
        # all_vars = tf.trainable_variables()
        # values1 = session1.run(all_vars)
        # values2 = session2.run(all_vars)

        # all_assign = []
        # for var, val1, val2 in zip(all_vars, values1, values2):
        #     all_assign.append(tf.assign(var, (val1 + val2)/ 2))
        weights = np.asarray(restored_weights)
        mean_weights = np.mean(weights,axis = 0)
        all_assign = []
        for var in range(len(all_vars)):
            all_assign.append(tf.assign(all_vars[var], mean_weights[var]))

        session3.run(all_assign)
        ###

        ### Run
        # with tf.Session() as sess:
        # saver.restore(session, "../Weights/comb_model/model")
        test_loss = session3.run([loss_op], {x: train_matrix, x_mask: test_matrix, drop_rate:1.0}) # rbm_inp:rbm_transformed,
        print("test", test_loss)
        out_matrix = session3.run(outputs,{x: train_matrix, drop_rate:1.0}) #rbm_inp:rbm_transformed,
        return out_matrix

def __main__():
    val_ids = np.load("val_ids.npy")
    omega_train, omega_test = get_datasplit(val_ids)
    train_matrix = matrix_data_omega(omega_train)
    test_matrix = matrix_data_omega(omega_test)
    # data = load_data("cil-collab-filtering-2019/data_train.csv")
    # train_matrix, test_matrix = split_data(data, split=10)
    train_matrix, test_matrix = normalize_01(train_matrix, test_matrix)
    print(np.unique(train_matrix), np.unique(test_matrix))
    print(train_matrix.shape, test_matrix.shape)

    trainer = Trainer(epochs=200)
    out_matrix = trainer.train(train_matrix, test_matrix, variational=False)
    # out_matrix = trainer.restore(train_matrix, test_matrix, variational=False)

    # make_submission(out_matrix, path = "cil-collab-filtering-2019/sampleSubmission.csv")


if __name__ == "__main__":
    __main__()
#### BACKUP STUFF

## TRAIN MORE ON HARD USERS
# worst = np.argsort(train_losses)[-nr_batches//4:]
# for user in worst:
#     sta = user*batch_size
#     end = (user+1)*batch_size
#     loss, _, num_nonzero = sess.run([loss_op, train_op, num_train_labels], {x: train_matrix[sta:end], x_mask: train_matrix[sta:end], drop_rate:0.9}) # genauso einfach over test losses averagen
#     train_losses.append(loss)


## SHUFFLE
# inds = np.random.permutation(nr_users)
# test_matrix = test_matrix[inds]
# train_matrix = train_matrix[inds]
# matrix = matrix[inds]

## SANITY CHECKS:
# test if outputs are the same:
# out_matrix = sess.run(outputs, {x:matrix})
# out_list = []
# ground_truth = []
# for entry in data[test_inds]:
#     field = entry[0].split("_")
#     row = int(field[0][1:])
#     col = int(field[1][1:])
#     if (test_matrix[row-1,col-1]*5!=entry[1]):
#         print(test_matrix[row-1,col-1]*5, entry[1])
#     ground_truth.append(entry[1])
#     pred = (out_matrix[row-1,col-1]*5) # .clip(min=1, max=5)
#     out_list.append(float(pred))
# print("RMSE test outputs one by one", np.sqrt(np.mean((np.asarray(out_list)-np.asarray(ground_truth))**2)))

# # test if outputs are the same:
# out_matrix = sess.run(outputs, {x:matrix})
# nonzero = np.count_nonzero(test_matrix)
# mask = test_matrix.astype(bool)
# assert(np.sum(mask)==nonzero)
# masked = out_matrix*mask
# print(nonzero)
# print()
# print("RMSE test outputs", np.sqrt(np.sum((masked*5-test_matrix*5)**2)/nonzero))
# # insert


# #### OLD SPLIT METHOD
# omega = np.asarray(omega)
# omega_train = omega[train_inds]
# omega_test = omega[test_inds]
# matrix = np.zeros((10000,1000))
# train_matrix = np.zeros((10000,1000))
# test_matrix = np.zeros((10000,1000))
#
# for row, col, entry in omega_train:
#     matrix[row, col] = entry
#     train_matix[row, col] = entry
# assert(not np.any([np.all(r==0) for r in matrix]))
#
# for row, col, entry in omega_test:
#     matrix[row, col] = entry
#     test_matrix[row, col] = entry
# del_inds = []
# for row in range(len(test_matrix)):
#     if np.all(test_matrix[row]==0):
#         del_inds.append(row)
# test_matrix = np.delete(test_matrix, del_inds, axis=0)
# assert(not np.any([np.all(r==0) for r in test_matrix]))
# print(matrix.shape, train_matrix.shape, test_matrix.shape)
