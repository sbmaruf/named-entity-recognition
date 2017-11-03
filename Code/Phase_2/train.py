import tensorflow as tf
import numpy as np
import os
from support import *
from evaluate import *
from prepare_io import *
from support import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def mlp_model(vocabulary_size,
              no_of_class,
              i_e,
              v_d,
              h_archi,
              x,y_goal,cap,keep_prob,
              Word2Vec):
    trainable_variable = []
    EMBEDDING_DIM = v_d
    with tf.name_scope("weights"):
        weights = {
            'emb_vec': tf.Variable(Word2Vec,name="emb_layer",dtype=tf.float32),
            'h1': tf.get_variable("h1", shape=[EMBEDDING_DIM, 100],initializer=tf.contrib.layers.xavier_initializer()),
            'hout': tf.get_variable("hout", shape=[100, no_of_class],initializer=tf.contrib.layers.xavier_initializer())
        }
    with tf.name_scope("biases"):
        biases = {
            'b_unused': tf.Variable("b1_unused",tf.contrib.layers.xavier_initializer()),
            'b1': tf.get_variable("b2", shape=[100],initializer=tf.contrib.layers.xavier_initializer()),
            'bout': tf.get_variable("bout", shape=[no_of_class],initializer=tf.contrib.layers.xavier_initializer())
        }

    tmp = tf.reshape(x, [-1])
    embed = tf.nn.embedding_lookup(weights['emb_vec'], tmp)
    h1_layer = tf.matmul(embed, weights['h1']) + biases['b1']
    h1_layer_op = tf.nn.relu(h1_layer)
    h1_drop_out = tf.nn.dropout(h1_layer_op, keep_prob)
    h1_out = tf.matmul(h1_drop_out, weights['hout']) + biases['bout'] + 0.000005
    y_ = tf.nn.softmax(h1_out)

    trainable_variable.append(weights['emb_vec'])
    trainable_variable.append(weights['h1'])
    trainable_variable.append(weights['hout'])
    trainable_variable.append(biases['b1'])
    trainable_variable.append(biases['bout'])


    cross_entropy = tf.reduce_sum(- y_goal * tf.log(y_), 1)
    loss = tf.reduce_mean(cross_entropy)

    return loss,y_,trainable_variable
    # with tf.name_scope("mlp_model"):
    #     with tf.name_scope("emb_vec"):
    #         if( i_e == 2 ):
    #             emb_vec = tf.Variable(Word2Vec,name="emb_layer",dtype=tf.float32)
    #     with tf.name_scope("embeddings"):
    #         tmp = tf.reshape(x,[-1])drop_out = tf.nn.dropout(layer_1, keep_prob)
    #         embed = tf.nn.embedding_lookup(emb_vec, tmp)
    #     with tf.name_scope("Cap_Feature"):
    #         cap_w = tf.Variable(tf.random_normal([no_of_class]), name="cap_weight")
    #
    #     cap_op = tf.multiply(cap,cap_w)
    #     trainable_variable.append(emb_vec)
    #     trainable_variable.append(cap_w)
    #
    #     op = embed
    #     last_layer = op
    #     with tf.name_scope("Hidden_Layes"):
    #         for idx,tot_neu in enumerate(h_archi):                                      test_s
    #             layer_no = idx+1
    #
    #             c = int(last_layer.get_shape()[1])
    #             weights = tf.Variable(tf.random_normal([c,tot_neu]),name="hid_{0}_{1}".format(layer_no-1,layer_no))
    #             bias = tf.Variable(tf.random_normal([tot_neu]), name="bias_{0}".format(layer_no))
    #
    #             trainable_variable.append(weights)
    #             trainable_variable.append(bias)
    #y_goal
    #             last_layer = tf.nn.relu(tf.add(tf.matmul(last_layer,weights),bias))
    #
    #drop_out = tf.nn.dropout(layer_1, keep_prob)
    #     c = int(last_layer.get_shape()[1])
    #
    #     with tf.name_scope("Output_layer"):
    #         weights = tf.Variable(tf.random_normal([c,no_of_class]),name="hid_{}_O".format(layer_no))
    #         bias = tf.Variable(tf.random_normal([no_of_class]), name="bias_O")
    #
    #
    #         trainable_variable.append(weights)
    #         trainable_variable.append(bias)
    #
    #         output = tf.add(tf.add(tf.matmul(last_layer,weights),bias),cap_op)
    #         prediction = tf.nn.softmax(tf.nn.sigmoid(output))



def train(param, train_data, vocab_data, dev_data,test_data,raw_data,words_info,tags_info):

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
    Word2Vec_ = []
    if( param['init_emb'] != "-1" ):
        Word2Vec_ = loademb( param['init_emb'] )

    Word2Vec = np.zeros((vocabulary_size,param['embed_dim']))
    for k,v in dico_words.items():
        if( k not in Word2Vec_ ):
            vector = np.random.normal(size=param['embed_dim'])
        else :
            vector = Word2Vec_[k]
        Word2Vec[ word_to_id[k] ] = vector

    tf.reset_default_graph()

    with tf.name_scope("Input"):
        x = tf.placeholder(tf.int32, shape=[None],name="input")
        cap = tf.placeholder(tf.float32, shape=[None, 1],name="cap_input")
        keep_prob = tf.placeholder(tf.float32,name="dropout")

    with tf.name_scope("Label"):
        y_goal = tf.placeholder(tf.float32, shape=[None, no_of_class],name="label")

    loss,y_,tVars = mlp_model(vocabulary_size,
                           no_of_class,
                           param['init_emb'],
                           param['embed_dim'],
                           param['hid_archi'],
                           x,y_goal, cap, keep_prob,
                           Word2Vec)

    l_m = param['lr_method']
    l_r = param['lr_rate']
    clp = param['clp_grad']
    if (l_m == 0):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = l_r)
    elif (l_m == 1):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate = l_r)
    elif (l_m == 2):
        optimizer = tf.train.AdagradOptimizer(learning_rate = l_r)
    elif (l_m == 3):
        optimizer = tf.train.AdagradDAOptimizer(learning_rate = l_r)
    elif (l_m == 4):
        optimizer = tf.train.MomentumOptimizer(learning_rate = l_r)
    elif (l_m == 5):
        optimizer = tf.train.AdamOptimizer(learning_rate = l_r)
    elif (l_m == 6):
        optimizer = tf.train.FtrlOptimizer(learning_rate = l_r)
    elif (l_m == 7):
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate = l_r)
    elif (l_m == 8):
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate = l_r)
    elif (l_m == 9):
        optimizer = tf.train.RMSPropOptimizer(learning_rate = l_r)

    if (clp == -1):
        train_step = optimizer.minimize(loss)
    else:
        grads_and_vars = optimizer.compute_gradients(loss,tVars)
        capped_grads_and_vars = [(tf.clip_by_value(gv[0], -1*param['clp_grad'], param['clp_grad']), gv[1]) for gv in grads_and_vars]
        train_step = optimizer.apply_gradients(capped_grads_and_vars)


    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # various initialization
    epoch = param['epoch']
    freq_eval = 10000
    best_dev = -np.inf
    best_test = -np.inf
    dropout_val = param['dropout']
    dropout_val = np.asarray(float(dropout_val))
    np.random.seed(1000)
    batch = 0
    early_stop = 0 ;
    eval_id = 0;

    for itr in range(epoch):
        print("\n\nStarting epoch {0}...\n".format(itr+1))
        for i, val in enumerate(np.random.permutation(len(train_data))):
            batch += 1
            data = train_data[i]

            cur_X,cur_Y,cur_cap = prepare_input(data,vocabulary_size,no_of_class,param['case_sense'])

            a,b = sess.run([train_step, loss], feed_dict={x: cur_X, y_goal: cur_Y, cap: cur_cap, keep_prob: dropout_val})

            if( batch%500 == 0 ):
                print("Epoch:",itr+1,"batch:",batch,"loss:",b)

            if batch % freq_eval == 0:
                eval_id += 1
                dev_score = evaluate(param,
                                     y_,
                                     sess,
                                     train_sentences,
                                     train_data,
                                     id_to_tag,
                                     dico_tags,
                                     vocabulary_size,
                                     x,y_goal,cap,keep_prob,
                                     batch,eval_id,itr)
                eval_id += 1
                dev_score = evaluate(param,
                                     y_,
                                     sess,
                                     dev_sentences,
                                     dev_data,
                                     id_to_tag,
                                     dico_tags,
                                     vocabulary_size,
                                     x,y_goal,cap,keep_prob,
                                     batch,eval_id,itr)
                eval_id += 1
                test_score = evaluate(param,
                                      y_,
                                      sess,
                                      test_sentences,
                                      test_data,
                                      id_to_tag,
                                      dico_tags,
                                      vocabulary_size,
                                      x, y_goal,cap,keep_prob,
                                      batch,eval_id,itr)

                print("Score on dev: %.5f" % dev_score)
                print("Score on test: %.5f" % test_score)
                if dev_score > best_dev:
                    early_stop = 0
                    best_dev = dev_score
                    print("New best score on dev.")
                    print("Saving model to disk...")
                    #model.save()
                if test_score > best_test:
                    best_test = test_score
                    print("New best score on test.")
                early_stop = early_stop + 1
                if( early_stop >= param['early_stop'] ):
                    print("Early stop occuring ... ")
                    break ;
                print("")
            if( early_stop >= param['early_stop'] ):
                print("Early stop occuring ... ")
                break ;

    print("Best test score %.5f" % dev_score)
    print("Best test score %.5f" % test_score)
    writer.close()
