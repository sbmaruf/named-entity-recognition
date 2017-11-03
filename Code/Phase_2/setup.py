import os
import tensorflow as tf
import numpy as np
import time
import optparse
from collections import OrderedDict

from train import *
from support import *
from prepare_io import *
import random

currDir = os.getcwd()
prevDir = os.path.dirname(currDir)
prevDir = os.path.dirname(prevDir)
#1000 1500 1000 500 100 50
"""
    parameter list
    ---------------------
    --train -T       --vocab -v         --dev -D          --test -t
    --embed_dim -e   --init_emb -i      --lr_method -L    --lr_rate -l
    --reload -r      --epoch -E         --clp_grad -c     --zero_replace -z
    --dropout -d       --label_scheme -s  --case_sense -u   --hid_archi -H
"""
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="./../../Data/eng.train",
    help="Train set location. Multiple location supported, separated by space. (default: './../../Data/eng.train') (type : 'str')"
)
optparser.add_option(
    "-v", "--vocab", default="./../../Data/eng.train",
    help="Vocabulary set location. Multiple location supported, separated by space. (default: './../../Data/eng.train') (type : 'str')"
)
optparser.add_option(
    "-D", "--dev", default="./../../Data/eng.testa",
    help="Dev set location. Multiple location supported, separated by space. (default: './../../Data/eng.testa') (type : 'str')"
)
optparser.add_option(
    "-t", "--test", default="./../../Data/eng.testb",
    help="Test set location. Multiple location supported, separated by space (Default: './../../Data/eng.testb') (type : 'str')"
)
optparser.add_option(
    "-e", "--embed_dim", default=100,
    type='int', help="Token embedding dimension (Default: 100 'int') (type : 'int')"
)
optparser.add_option(
    "-i", "--init_emb", default="./../../Data/sskip.100.vectors",
    type='str', help="Location of pretrained embeddings (Default: './../../Data/sskip.100.vectors') (type: 'str')"
)
optparser.add_option(
    "-L", "--lr_method", default=0,
    type='int', help="Learning method (SGD(0), Adadelta(1), Adagrad(2), AdagradDA(3), Momentum(4), Adam(5), ftrl(6), ProximalSGD(7), ProximalAdagrad(8), RMSProp(9) (Default: 0) (type: 'int')"
)
optparser.add_option(
    "-l", "--lr_rate", default=.005,
    type='float', help="Learning rate (Default: .005) (type: 'float')"
)
optparser.add_option(
    "-r", "--reload", default=0,
    type='int', help="Reload the last saved model (Default: 0) (type: 'int')"
)
optparser.add_option(
    "-H", "--hid_archi", default="100",
    help="Number of hidden layer with number of hidden neurons. Ex : '100 200 300' for 3 layer of 100,200,300 neuron. Currently supported upto 1 layer. (Default: '100') (type: 'str')"
)
optparser.add_option(
    "-E", "--epoch", default=100,
    type='int', help="Number of training epoch. (Default: 100) (type: 'int')"
)
optparser.add_option(
    "-c", "--clp_grad", default=5,
    type='float', help="The threshold of clipping gradient (Default: 1) (type: 'float')"
)
optparser.add_option(
    "-z", "--zero_replace", default="1",
    type = 'int', help="Replace all digits by zero (Default: 1) (type: 'int')"
)
optparser.add_option(
    "-d", "--dropout", default=.75,
    type='float', help="dropout rate [<float> (0-1)] (Default: .5) (type: 'float')"
)
optparser.add_option(
    "-s", "--label_scheme", default=2,
    type= 'int', help="label_scheme - IOB(1) or IOBES(2) (Default: 2) (type: 'int')"
)
optparser.add_option(
    "-u", "--case_sense", default=0,
    type='int', help="Case Sensitivity feature, NO(0), YES(1) (Default: 0) (type: 'int')"
)
optparser.add_option(
    "-x", "--early_stop", default=100,
    type='int', help="How many batch further we will wait for dev score update? (Default: 100) (type: 'int')"
)


opts = optparser.parse_args()[0]


def parse_parameter():
    """
    parameter list
    ---------------------
    --train -T       --vocab -v         --dev -D          --test -t
    --embed_dim -e   --init_emb -i      --lr_method -L    --lr_rate -l
    --reload -r      --epoch -E         --clp_grad -c     --zero_replace -z
    --dropout -d     --label_scheme -s  --case_sense -u   --hid_archi -H
    --early_stop -x

    Parse the input parameter of the setup.py file
    :return: <dictionary>  a dictionary consists with parameter
    """

    param = OrderedDict()
    param['embed_dim'] = opts.embed_dim
    param['lr_method'] = opts.lr_method
    param['lr_rate'] = opts.lr_rate
    param['reload'] = opts.reload
    param['epoch'] = opts.epoch
    param['clp_grad'] = opts.clp_grad
    param['zero_replace'] = opts.zero_replace
    param['dropout'] = opts.dropout
    param['label_scheme'] = opts.label_scheme
    param['case_sense'] = opts.case_sense
    param['early_stop'] = opts.early_stop

    address = ""
    for k,v in param.items():
        key = k
        key = str(k)
        val = str(v)
        address += key + " " + val + " "
    address += "hid_archi "
    address += str(opts.hid_archi) +" "

    address = "./evaluation/temp/%s" % address
    address += time.strftime("%c")
    command = "mkdir '%s'" % address

    action = os.system(command)

    param['init_emb'] = opts.init_emb
    param['hid_archi'] = splitNlist(opts.hid_archi)
    param['train'] = splitNlist(opts.train)
    param['vocab'] = splitNlist(opts.vocab)
    param['dev'] = splitNlist(opts.dev)
    param['test'] = splitNlist(opts.test)
    param['folder'] = address

    for i,v in enumerate(param['hid_archi']):
        param['hid_archi'][i] = int(v)

    assert( param['embed_dim'] > 0 )
    assert( param['lr_method'] >= 0 and
            param['lr_method'] <= 9 and
            type(param['lr_method'])==type(int(1)))
    assert( param['reload'] == 0 or
            param['reload'] == 1 )
    assert( param['epoch'] > 0 )
    assert( param['zero_replace'] == 0 or
            param['zero_replace'] == 1 )
    assert( param['dropout'] > 0 and
            param['dropout'] <= 1 )
    assert( param['label_scheme'] == 1 or
            param['label_scheme'] == 2 )
    assert( param['case_sense'] == 0 or
            param['case_sense'] == 1 )


    for i in param['train']:
        if not os.path.isfile(i):
            raise Exception("Train file {0} doesn't exists.".format(i))

    for i in param['vocab']:
        if not os.path.isfile(i):
            raise Exception("vocabulary file {0} doesn't exists.".format(i))

    for i in param['dev']:
        if not os.path.isfile(i):
            raise Exception("dev file {0} doesn't exists.".format(i))

    for i in param['dev']:
        if not os.path.isfile(i):
            raise Exception("test file {0} doesn't exists.".format(i))

    for i in param['hid_archi']:
        if not type(i) == type(int(1)):
            raise Exception("Number of neuron(s) must be integer")

    print("--- Printing the parameters ---")
    for k , v in param.items():
        print(k,"-",v)
    print("--- x ---")
    print("\n")



    return param


def preprocess( param ):

    train_sentences, vocab_sentences, dev_sentences, test_sentences = read_datsets(param)

    if( param['zero_replace']==1 ):
        replace_digits(train_sentences, vocab_sentences, dev_sentences, test_sentences)

    update_tag_scheme(train_sentences, param['label_scheme'])
    update_tag_scheme(vocab_sentences, param['label_scheme'])
    update_tag_scheme(dev_sentences, param['label_scheme'])
    update_tag_scheme(test_sentences, param['label_scheme'])

    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, param['case_sense'])
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    raw_data = [train_sentences, vocab_sentences, dev_sentences, test_sentences]
    words_info = [dico_words, word_to_id, id_to_word]
    tags_info = [dico_tags, tag_to_id, id_to_tag]

    return raw_data, words_info, tags_info


def main():

    """
    parameter list
    ---------------------
    --train -T       --vocab -v         --dev -D          --test -t
    -1--embed_dim -e   --init_emb -i      --lr_method -L    --lr_rate -l
    --reload -r      --epoch -E         --clp_grad -c     --zero_replace -z
    --dropout -d       --label_scheme -s  --case_sense -u   --hid_archi -H

    """

    print("--- experiment taken on : ", time.strftime("%c")," ---\n\n")
    start_time = time.time()

    param = parse_parameter()
    raw_data, words_info, tags_info = preprocess(param)

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

    train_data = prepare_dataset( train_sentences, word_to_id, tag_to_id, param['case_sense'] )
    vocab_data = prepare_dataset( vocab_sentences, word_to_id, tag_to_id, param['case_sense'] )
    dev_data = prepare_dataset( dev_sentences, word_to_id, tag_to_id, param['case_sense'] )
    test_data = prepare_dataset( test_sentences, word_to_id, tag_to_id, param['case_sense'] )

    print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

    train(param,
          train_data,
          vocab_data,
          dev_data,
          test_data,
          raw_data,
          words_info,
          tags_info
    )
    end_time = time.time()
    print("--- total execution time %s seconds ---" % end_time)



if __name__ == '__main__':
    main()
