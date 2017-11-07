import tensorflow as tf
import numpy as np
from prepare_io import *

eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "run")
eval_script = os.path.join(eval_path, "conlleval")

def max_idx_softmax(pred_out):
    ids = []
    for i, word in enumerate(pred_out):
        one_hot_vec = pred_out[i]
        mx = one_hot_vec[0]
        mx_idx = 0
        for j, val in enumerate(one_hot_vec):
            if (val > mx):
                mx = val
                mx_idx = j
        ids.append(mx_idx)
    return ids

def evaluate_CNN(param,
             cnn,
             sess,
             dev_sentences,
             dev_data,
             id_to_tag,
             dico_tags,
             vocabulary_size,word_to_id,
             batch,eval_id,itr):

    no_of_class = len(id_to_tag)
    predictions = []
    count = np.zeros((no_of_class, no_of_class), dtype=np.int32)
    dropout_val = 1
    dropout_val = np.asarray(float(dropout_val))
    for dev_sentence, data in zip(dev_sentences, dev_data):

        cur_X, cur_Y, cur_cap = prepare_input_CNN(data, vocabulary_size, no_of_class, param['case_sense'],param['win_size'],word_to_id[''])
        # rint("Cur_X : ", len(cur_X))
        # assert(len(cur_X) == len(cur_Y))

        feed_dict = {
            cnn.input_x: cur_X,
            cnn.input_y: cur_Y,
            cnn.dropout_keep_prob: np.asarray(1)
        }
        pred_out = sess.run( [cnn.predictions] , feed_dict = feed_dict )
        # print(pred_out)
        # print(len(pred_out))
        # print(type(pred_out))
        #y_preds = max_idx_softmax(pred_out)
        y_preds = np.asarray(pred_out[0])
        y_reals = np.array(data['tags']).astype(np.int32)


        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]

        if param['label_scheme'] == 2:  # 'iobes'
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(dev_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")


    output_path = os.path.join(param['folder'], "eval.%i.output" % eval_id)
    scores_path = os.path.join(param['folder'], "eval.%i.scores" % eval_id)

    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
        os.system("%s < '%s' > '%s'" % (eval_script, output_path, scores_path))

    #result print
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    for line in eval_lines:
        print(line)

    #confusion matrix print
    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * no_of_class)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(no_of_class)] + ["Percent"])
    ))
    for i in range(no_of_class):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * no_of_class)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in range(no_of_class)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        ))

    # Global accuracy
    print("%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    ))
    return float(eval_lines[1].strip().split()[-1])





def evaluate_RNN(param,
             rnn,
             sess,
             dev_sentences,
             dev_data,
             id_to_tag,
             dico_tags,
             vocabulary_size,word_to_id,
             batch,eval_id,itr):

    no_of_class = len(id_to_tag)
    predictions = []
    count = np.zeros((no_of_class, no_of_class), dtype=np.int32)
    dropout_val = 1
    dropout_val = np.asarray(float(dropout_val))
    for dev_sentence, data in zip(dev_sentences, dev_data):

        cur_X, cur_Y, seq_len, cur_cap = prepare_input_rnn(data, vocabulary_size, no_of_class, param['case_sense'])

        feed_dict = {
            rnn.input_x: cur_X,
            rnn.input_y: cur_Y,
            rnn.rnn_seq_len: seq_len,
            rnn.dropout_keep_prob: np.asarray(1)
        }

        pred_out = sess.run( [rnn.predictions] , feed_dict = feed_dict )
        # print(pred_out)
        y_preds = np.asarray(pred_out[0][0])
        # print(y_preds)
        y_reals = np.array(data['tags']).astype(np.int32)


        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]

        if param['label_scheme'] == 2:  # 'iobes'
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(dev_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")


    output_path = os.path.join(param['folder'], "eval.%i.output" % eval_id)
    scores_path = os.path.join(param['folder'], "eval.%i.scores" % eval_id)

    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
        os.system("%s < '%s' > '%s'" % (eval_script, output_path, scores_path))

    #result print
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    for line in eval_lines:
        print(line)

    #confusion matrix print
    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * no_of_class)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(no_of_class)] + ["Percent"])
    ))
    for i in range(no_of_class):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * no_of_class)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in range(no_of_class)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        ))

    # Global accuracy
    print("%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    ))
    return float(eval_lines[1].strip().split()[-1])
