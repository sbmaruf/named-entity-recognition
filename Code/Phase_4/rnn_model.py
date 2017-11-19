import tensorflow as tf
import numpy as np

class rnn_model(object):
    def __init__(self,
                 num_classes,
                 embeddings,
                 lstm_hidden_size,
                 char_hidden_size,
                 nchars,
                 dim_char,
                 crf,
                 lstm_crf_hid):

        with tf.name_scope("inputs"):
            # (i,j) = j'th word id of i'th sentence
            self.word_ids = tf.placeholder(tf.int32, shape=[None, None],name="word_ids")

            # (i) = length of i'th sentence
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],name="sequence_lengths")

            # (i,j,k) = id value of k'th letter if of j'th word of i'th sentence
            self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],name="char_ids")

            # (i,j) = number of letter of j'th word of i'th sentence)
            self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],name="word_lengths")

            # dropout and decaying learning rate
            self.dropout = tf.placeholder(dtype=tf.float32, shape=[],name="dropout")

            # placeholder for learning rate. for decaying learning
            self.lr = tf.placeholder(dtype=tf.float32, shape=[],name="lr")


        with tf.name_scope("outputs"):
            # (i,j) = label of j'th word of i'th sentence
            self.labels = tf.placeholder(tf.int32, shape=[None, None],name="labels")


        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(
                        embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=True)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,self.word_ids, name="word_embeddings")


        with tf.variable_scope("chars"):
            if dim_char != 0 :
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[nchars, dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(char_hidden_size,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(char_hidden_size,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*char_hidden_size])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(lstm_hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(lstm_hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)


        with tf.variable_scope("proj"):

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*lstm_hidden_size])
            if (lstm_crf_hid == 0):
                W = tf.get_variable("W", dtype=tf.float32,
                                    shape=[2 * lstm_hidden_size, num_classes])
                b = tf.get_variable("b", shape=[num_classes],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
                pred = tf.matmul(output, W) + b
                self.logits = tf.reshape(pred, [-1, nsteps, num_classes])
            else:
                W = tf.get_variable("W", dtype=tf.float32,
                                    shape=[2 * lstm_hidden_size, lstm_crf_hid])
                b = tf.get_variable("b", shape=[lstm_crf_hid],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
                pred = tf.matmul(output, W) + b
                pred1 = tf.nn.relu(pred)

                W1 = tf.get_variable("W1", dtype=tf.float32,
                                     shape=[lstm_crf_hid, num_classes])
                b1 = tf.get_variable("b1", shape=[num_classes],
                                     dtype=tf.float32, initializer=tf.zeros_initializer())
                pred2 = tf.matmul(pred1, W1) + b1

                self.logits = tf.reshape(pred2, [-1, nsteps, num_classes])

        with tf.variable_scope("loss-op"):

            if crf != 0:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
                self.trans_params = trans_params  # need to evaluate it for decoding
                self.loss = tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)
                self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),tf.int32)