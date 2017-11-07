import tensorflow as tf
import numpy as np
import os
import codecs
from prepare_io import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cnn_model import *
from evaluate import *
from rnn_model import *

def train(param,
          train_data,
          vocab_data,
          dev_data,
          test_data,
          raw_data,
          words_info,
          tags_info):

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

    if( param['init_emb'] == "" ):
        embedding = tf.random_uniform([vocabulary_size, param['embed_dim']], -1.0, 1.0)
    else:
        Word2Vec = np.zeros((vocabulary_size,param['embed_dim']))
        Word2Vec_ = loademb( param['init_emb'] )
        for k,v in dico_words.items():
            if( k not in Word2Vec_ ):
                vector = np.random.normal(size=param['embed_dim'])
            else :
                vector = Word2Vec_[k]
                Word2Vec[ word_to_id[k] ] = vector
        embedding = Word2Vec
    tf.reset_default_graph()
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            rnn = rnn_model(num_classes=no_of_class,
                            embeddings=embedding,
                            hidden_size=param['hid_archi'][0])

            learning_method = param['lr_method']
            learning_rate = param['lr_rate']
            clp = param['clp_grad']

            optimizer = getOptimizer( learning_method,learning_rate )

            if (clp == -1):
                train_step = optimizer.minimize(rnn.loss)
            else:
                grads_and_vars = optimizer.compute_gradients(rnn.loss)
                capped_grads_and_vars = [(tf.clip_by_value(gv[0], -1*param['clp_grad'], param['clp_grad']), gv[1]) for gv in grads_and_vars]
                train_step = optimizer.apply_gradients(capped_grads_and_vars)

            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)

            # various initialization
            epoch = param['epoch']
            freq_eval = 10000
            best_dev1 = -1000000000.0 #inf
            best_dev2 = -1000000000.0 #inf
            np.random.seed(1000)
            batch = 0
            early_stop = 0 ;
            eval_id = 0;

            saver = tf.train.Saver()

            for itr in range(epoch):
                print("\n\nStarting epoch {0}...\n".format(itr+1))
                for i, val in enumerate(np.random.permutation(len(train_data))):
                    batch += 1
                    data = train_data[i]

                    cur_X,cur_Y,seq_len,cur_cap = prepare_input_rnn(data,vocabulary_size,no_of_class,param['case_sense'])
                    feed_dict = {
                        rnn.input_x: cur_X,
                        rnn.input_y: cur_Y,
                        rnn.rnn_seq_len: seq_len,
                        rnn.dropout_keep_prob: np.asarray(param['dropout'])
                    }
                    a,b = sess.run( [train_step, rnn.loss], feed_dict = feed_dict )
                    if( batch%1000 == 0 ):
                        print("Epoch:",itr+1,"batch:",batch,"loss:",b)

                    if batch % freq_eval == 0:
                        eval_id += 1
                        train_score = evaluate_RNN(param,
                                                     rnn,
                                                     sess,
                                                     train_sentences,
                                                     train_data,
                                                     id_to_tag,
                                                     dico_tags,
                                                     vocabulary_size,word_to_id,
                                                     batch,eval_id,itr)
                        eval_id += 1
                        dev_score1 = evaluate_RNN(param,
                                                     rnn,
                                                     sess,
                                                     dev_sentences,
                                                     dev_data,
                                                     id_to_tag,
                                                     dico_tags,
                                                     vocabulary_size,word_to_id,
                                                     batch,eval_id,itr)

                        eval_id += 1
                        dev_score2 = evaluate_RNN(param,
                                                     rnn,
                                                     sess,
                                                     test_sentences,
                                                     test_data,
                                                     id_to_tag,
                                                     dico_tags,
                                                     vocabulary_size,word_to_id,
                                                     batch,eval_id,itr)

                        print("Score on dev1: %.5f" % dev_score1)
                        print("Score on dev2: %.5f" % dev_score2)
                        if dev_score1 > best_dev1:
                            best_dev1 = dev_score1
                            save_model(sess,
                                       saver,
                                       param['folder'],
                                       "dev1.ckpt")
                        if dev_score2 > best_dev2:
                            best_dev2 = dev_score2
                            save_model(sess,
                                       saver,
                                       param['folder'],
                                       "dev2.ckpt")

                        print("\nBest test score %.5f" % best_dev1)
                        print("Best test score %.5f" % best_dev2)
            result_report = os.path.join(param['folder'],"results")
            with codecs.open(result_report, 'a', 'utf8') as f:
                f.write("Best dev score %.5f\n" % dev_score1)
                f.write("Best test score %.5f" % dev_score2)
