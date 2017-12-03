import tensorflow as tf
import numpy as np
import os
import codecs
import random
from random import shuffle
from prepare_io import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cnn_model import *
from evaluate import *
from rnn_model import *

tensor_flow_seed = 100
tf.set_random_seed(tensor_flow_seed)

def train(param,
          train_data,
          vocab_data,
          dev_data,
          test_data,
          raw_data,
          words_info,
          tags_info,
          char_info):

    train_sentences = raw_data[0]
    vocab_sentences = raw_data[1]
    dev_sentences = raw_data[2]
    test_sentences =  raw_data[3]

    dico_words = words_info[0]
    word_to_id = words_info[1]
    id_to_word = words_info[2]

    dico_tags = tags_info[0]
    tag_to_id = tags_info[1]
    id_to_tag = tags_info[2]
    vocabulary_size = len(dico_words)
    no_of_class = len(dico_tags)

    dico_char = char_info[0]
    char_to_id = char_info[1]
    id_to_char = char_info[2]

    no_of_char = len(dico_char)

    embedding = get_embeddings(param, vocabulary_size, no_of_class, dico_words, word_to_id)


    tf.reset_default_graph()
    my_graph = tf.Graph()
    with my_graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            rnn = rnn_model(num_classes=no_of_class,
                            embeddings=embedding,
                            lstm_hidden_size=param['hid_archi'][1],
                            char_hidden_size= param['hid_archi'][0],
                            nchars=no_of_char,
                            dim_char=param['char_dim'],
                            crf = param['crf'],
                            lstm_crf_hid=param['lstm_crf_hid'])
            learning_method = param['lr_method']
            learning_rate = param['lr_rate']
            clp = param['clp_grad']

            optimizer = getOptimizer( learning_method, learning_rate )

            if (clp == -1):
                train_step = optimizer.minimize(rnn.loss)
            else:
                grads, vs = zip(*optimizer.compute_gradients(rnn.loss))
                grads, gnorm = tf.clip_by_global_norm(grads, param['clp_grad'])
                train_step = optimizer.apply_gradients(zip(grads, vs))

                # grads_and_vars = optimizer.compute_gradients(rnn.loss)
                # capped_grads_and_vars = [(tf.clip_by_value(gv[0], -1*param['clp_grad'], param['clp_grad']), gv[1]) for gv in grads_and_vars]
                # train_step = optimizer.apply_gradients(capped_grads_and_vars)

            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)

            # various initialization
            epoch = param['epoch']
            freq_eval = 10000
            best_dev = -1000000000.0 #inf
            best_test = -1000000000.0 #-inf
            early_stop = 0
            eval_id = 0
            np.random.seed(1000)

            dropout = np.asarray(param['dropout'])
            lr_rate = np.asarray(param['lr_rate'])
            lr_decay = np.asarray(param['lr_decay'])

            train_data = np.asarray(train_data)
            test_data = np.asarray(test_data)
            dev_data = np.asarray(dev_data)
            train_sentences = np.asarray(train_sentences)
            dev_sentences = np.asarray(dev_sentences)
            test_sentences = np.asarray(test_sentences)
            saver = tf.train.Saver()
            
            rand_seed_lst = []
            for i in range(epoch):
                rand_seed_lst.append(i)

            for itr in range(epoch):
                print("\nStarting epoch {0}...\n".format(itr + 1))
				                
                random.seed(rand_seed_lst[itr])
                shuffle(train_data)
                random.seed(rand_seed_lst[itr])
                shuffle(train_sentences)
                
                genarators = get_minibatch(train_data, param['batch_size'])
                tot_batch = 0
                total_loss = 0
                for batch, val in enumerate(genarators):
                    word_ids, \
                    sequence_lengths,\
                    char_ids,\
                    word_lengths,\
                    labels  = prepare_input_rnn(val)
                    feed_dict = {
                        rnn.word_ids: word_ids,
                        rnn.sequence_lengths: sequence_lengths,
                        rnn.char_ids: char_ids,
                        rnn.word_lengths: word_lengths,
                        rnn.labels: labels,
                        rnn.dropout: dropout,
                        rnn.lr: lr_rate
                    }
                    a,b = sess.run( [train_step, rnn.loss], feed_dict = feed_dict )
                    if( batch%50 == 0 and batch > 0 ):
                        print("Epoch:",itr+1,"batch:",batch,"loss:",b)

                    total_loss += b
                    tot_batch += 1


                result_report = os.path.join(param['folder'], "results")
                with codecs.open(result_report, 'a', 'utf8') as f:
                    f.write("Average loss after epoch {0} : {1}\n".format(itr,total_loss/tot_batch))

                print("Average loss after epoch {0} : {1}".format(itr,total_loss/tot_batch))

                lr_rate = max(lr_rate * lr_decay,.0000001)
                eval_id += 1
                train_score = evaluate_RNN(param,
                                         rnn,
                                         sess,
                                         train_sentences,
                                         train_data,
                                         id_to_tag,
                                         dico_tags,
                                         vocabulary_size, word_to_id,
                                         batch, eval_id, itr)
                eval_id += 1
                dev_score = evaluate_RNN(param,
                                          rnn,
                                          sess,
                                          dev_sentences,
                                          dev_data,
                                          id_to_tag,
                                          dico_tags,
                                          vocabulary_size,word_to_id,
                                          batch,eval_id,itr)
                eval_id += 1
                test_score = evaluate_RNN(param,
                                         rnn,
                                         sess,
                                         test_sentences,
                                         test_data,
                                         id_to_tag,
                                         dico_tags,
                                         vocabulary_size, word_to_id,
                                         batch, eval_id, itr)

                print("Score on train: %.5f" % train_score)
                print("Score on dev: %.5f" % dev_score)
                print("Score on test: %.5f" % test_score)
                if dev_score > best_dev:
                    print("\nNew best score on dev.")
                    best_dev = dev_score
                    save_model(sess,
                               saver,
                               param['folder'],
                               "dev.ckpt")
                    test_on_best_dev = test_score
                    early_stop=0
                else :
                    early_stop+=1

                if test_score > best_test:
                    print("\nNew best score on test.")
                    best_test = test_score
                    save_model(sess,
                               saver,
                               param['folder'],
                               "test.ckpt")

                print("\nBest dev score %.5f" % best_dev)
                print("Best test score %.5f" % best_test)
                print("Test score on best_dev %.5f" % test_on_best_dev)
                if (early_stop == param['early_stop']):
                    break

            result_report = os.path.join(param['folder'],"results")
            with codecs.open(result_report, 'a', 'utf8') as f:
                f.write("Best dev score %.5f\n" % best_dev)
                f.write("Best test score %.5f\n" % best_test)
                f.write("Test score on best_dev %.5f\n" % test_on_best_dev)
                f.write("\n\n\n tensorflow seed value %i\n" % tensor_flow_seed)
                f.write("generator seed list : " + str(rand_seed_lst) + "\n" )

    writer = tf.summary.FileWriter('./graph_log', graph=my_graph)
    writer.close()
