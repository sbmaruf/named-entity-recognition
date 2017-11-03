import tensorflow as tf
import numpy as np
from prepare_io import *

eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
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

def evaluate(param,
             prediction,
             sess,
             dev_sentences,
             dev_data,
             id_to_tag,
             dico_tags,
             vocabulary_size,
             x,y_goal,cap,keep_prob,
             batch,eval_id,itr):

    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)
    dropout_val = 1
    dropout_val = np.asarray(float(dropout_val))
    for dev_sentence, data in zip(dev_sentences, dev_data):

        cur_X, cur_Y, cur_cap = prepare_input(data, vocabulary_size, n_tags, param['case_sense'])

        pred_out = sess.run(prediction, feed_dict={x: cur_X, y_goal: cur_Y, cap: cur_cap, keep_prob: dropout_val})

        y_preds = max_idx_softmax(pred_out)
        y_reals = np.array(data['tags']).astype(np.int32)

        assert (len(y_preds) == len(y_reals))

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


    #output_path = os.path.join(param['folder'])
    #scores_path = os.path.join(param['folder'])
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
    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(n_tags)] + ["Percent"])
    ))
    for i in range(n_tags):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in range(n_tags)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        ))

    # Global accuracy
    print("%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    ))

    return float(eval_lines[1].strip().split()[-1])
