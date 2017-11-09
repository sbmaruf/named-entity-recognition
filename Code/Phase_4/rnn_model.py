import tensorflow as tf
import numpy as np

class rnn_model(object):
    def __init__(self,
                 num_classes,
                 embeddings,
                 hidden_size ):

        with tf.name_scope("input"):
            #shape of input_x    :    batch , no of [word ids of the sentence]
            self.input_x                =   tf.placeholder(tf.int32, shape=[None,None], name = "input_x")
            self.dropout_keep_prob      =   tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.rnn_seq_len            =   tf.placeholder(tf.int32, shape=[None], name="rnn_seq_len")


        with tf.name_scope("output"):
            # shape of output_y   :   batch
            self.input_y = tf.placeholder(tf.int32, shape=[None,None], name="input_y")


        with tf.name_scope("embedding"):
            L = tf.Variable(embeddings, dtype=tf.float32, trainable=True)
            # shape of pretrained_embeddings   :   batch,
            #                                      words of the sentence, word_vector_size)
            pretrained_embeddings = tf.nn.embedding_lookup(L, self.input_x)


        with tf.name_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    pretrained_embeddings,
                                                                    sequence_length=self.rnn_seq_len,
                                                                    dtype=tf.float32)


        with tf.name_scope("out_lstm"):
            # shape of the output of bi-lstm     :   batch,
            #                                        words of the sentence, 2 x hidden_size )
            out_lstm = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(out_lstm, self.dropout_keep_prob)


        hidden_nn_size = 500
        with tf.name_scope("hidden_layer"):
            # shape of W_o       :        2*hidden_size(output # column of output),    hidden layer size
            W_o = tf.get_variable("W_o", shape=[2*hidden_size, hidden_nn_size],dtype=tf.float32)
            b_o = tf.get_variable("b_o", shape=[hidden_nn_size], dtype=tf.float32,initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(out_lstm)[1]

            #shape of flat :    word of the sentence***s*** , 2*hidden_size
            flat = tf.reshape(output, [-1, 2*hidden_size])

            #shape of pred1 :    word of the sentence***s***, hidden layee size
            pred1 = tf.matmul(flat, W_o) + b_o
            pred1 = tf.nn.relu(pred1)
            pred1 = tf.nn.dropout(pred1, self.dropout_keep_prob)
            
            # shape of W_oh       :        hidden layee size,   num_class
            W_oh = tf.get_variable("W_oh", shape=[hidden_nn_size, num_classes],dtype=tf.float32)
            b_oh = tf.get_variable("b_oh", shape=[num_classes], dtype=tf.float32,initializer=tf.zeros_initializer())

            #shape of pred :    word of the sentence***s***, num_classes
            pred = tf.matmul(pred1, W_oh)+b_oh


        with tf.name_scope("model_output"):
            #shape of scores :  batch,
            #                   words of the sentence, num_classes
            scores = tf.reshape(pred, [-1, ntime_steps, num_classes])    #re-batching from flatted form
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=scores)
            ## hidden layer


        mask = tf.sequence_mask(self.rnn_seq_len)

        with tf.name_scope("model_loss"):
            losses = tf.boolean_mask(losses, mask)
            loss = tf.reduce_mean(losses)


        self.loss = loss
        self.predictions = tf.cast(tf.argmax(scores, axis=-1),tf.int32)
