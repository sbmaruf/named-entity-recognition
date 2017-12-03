import os
import tensorflow as tf
import numpy as np
import time
import optparse
from collections import OrderedDict
from tensorflow.contrib import learn
import random
import itertools
from support import *
from train import*
currDir = os.getcwd()
prevDir = os.path.dirname(currDir)
prevDir = os.path.dirname(prevDir)
"""
        total number of options = 25.
        parameter list
        ---------------------
        --train -T          --vocab -v          --dev -D            --test -t
        --embed_dim -e      --init_emb -i       --lr_method -L      --lr_rate -l
        --lr_decay  -j      --reload -r         --hid_archi -H      --epoch -E         
        --clp_grad -c       --zero_replace -z   --dropout -d        --label_scheme -s  
        --case_sense -u     --early_stop -x     --filter_size -f    --no_of_filter -n   
        --l2_reg_lambda -R  --batch_size -b     --win_size -w       --char_dim
        --crf               --lstm_crf_hid
"""
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="./../../Data/esp.train",
    help="Train set location. Multiple location supported, separated by space. (default: './../../Data/eng.train') (type : 'str')"
)
optparser.add_option(
    "-v", "--vocab", default="./../../Data/esp.train ./../../Data/esp.testa ./../../Data/esp.testb",
    help="Vocabulary set location. Multiple location supported, separated by space. (default: './../../Data/eng.train') (type : 'str')"
)
optparser.add_option(
    "-D", "--dev", default="./../../Data/esp.testa",
    help="Dev set location. Multiple location supported, separated by space. (default: './../../Data/eng.testa') (type : 'str')"
)
optparser.add_option(
    "-t", "--test", default="./../../Data/esp.testb",
    help="Test set location. Multiple location supported, separated by space (Default: './../../Data/eng.testb') (type : 'str')"
)
optparser.add_option(
    "-e", "--embed_dim", default=100,
    type='int', help="Token embedding dimension (Default: 100 'int') (type : 'int')"
)
optparser.add_option(
    "-i", "--init_emb", default="./../../Data/ES64",
    type='str', help="Location of pretrained embeddings. For random initializer use empty string. (Default: './../../Data/sskip.100.vectors') (type: 'str')"
)
optparser.add_option(
    "-L", "--lr_method", default=0,
    type='int', help="Learning method (SGD(0), Adadelta(1), Adagrad(2), AdagradDA(3), Momentum(4), Adam(5), ftrl(6), ProximalSGD(7), ProximalAdagrad(8), RMSProp(9) (Default: 0) (type: 'int')"
)
optparser.add_option(
    "-l", "--lr_rate", default=.05,
    type='float', help="Learning rate (Default: .005) (type: 'float')"
)
optparser.add_option(
    "-j", "--lr_decay", default=".9",
    type='float', help="The way learning rate reduces (Default: 0) (type: 'int')"
)
optparser.add_option(
    "-r", "--reload", default=0,
    type='int', help="Reload the last best dev model. (0 to stop the feature) (Default: 0) (type: 'int')"
)
optparser.add_option(
    "-H", "--hid_archi", default="100 300",
    help="Number of hidden layer with number of hidden neurons. Ex : '100 200 300' for 3 layer of 100,200,300 neuron. for lstm first integer is char level lstm size, second integer is word level lstm size."
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
    "-z", "--zero_replace", default="0",
    type = 'int', help="Replace all digits by zero (Default: 0) (type: 'int')"
)
optparser.add_option(
    "-d", "--dropout", default=.5,
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
    "-x", "--early_stop", default=15,
    type='int', help="How many epoch further we will wait for dev score update? (Default: 5) (type: 'int')"
)
optparser.add_option(
    "-f", "--filter_size", default='3 4 5',
     help="Size of the each filter separated by space. Also indicates number of filter (Default : '3 4 5') (type: 'str')"
)
optparser.add_option(
    "-n", "--no_of_filter", default='100 100 100',
    help="No of filter for each size separated by space. Also indicates number of filter (Default: '25 100') (type: 'str')"
)
optparser.add_option(
    "-R", "--l2_reg_lambda", default='0.0',
     type = 'float', help="L2 regularization lambda. (Default: 0.0) (type: float)"
)
optparser.add_option(
    "-b", "--batch_size", default='20',
    type='int', help="Batch size (Default: 20) (type: 'int')"
)
optparser.add_option(
    "-w", "--win_size", default='5',
    type='int', help="Size of the sliding window (Default: 5) (type: 'int')"
)
optparser.add_option(
    "-p", "--char_dim", default="100",
    type='int', help="Character level embedding (Default: 25) (type: 'int')"
)
optparser.add_option(
    "-g", "--crf", default="1",
    type='int', help="crf layes is on(1) or off(0) (Default: 0) (type: 'int')"
)
optparser.add_option(
    "-C", "--lstm_crf_hid", default="300",
    type='int', help="size of the hidden layer between lstm and crf. 0 to deactivate. (Default : 0) (type: 'int')"
)
opts = optparser.parse_args()[0]


def parse_parameter():
    """
        total number of options = 25.
        parameter list
        ---------------------
        --train -T          --vocab -v          --dev -D            --test -t
        --embed_dim -e      --init_emb -i       --lr_method -L      --lr_rate -l
        --lr_decay  -j      --reload -r         --hid_archi -H      --epoch -E
        --clp_grad -c       --zero_replace -z   --dropout -d        --label_scheme -s
        --case_sense -u     --early_stop -x     --filter_size -f    --no_of_filter -n
        --l2_reg_lambda -R  --batch_size -b     --win_size -w       --char_dim
        --crf               --lstm_crf_hid

    Parse the input parameter of the setup.py file
    :return: <dictionary>  a dictionary consists with parameter
    """

    param = OrderedDict()
    param['embed_dim'] = opts.embed_dim
    param['lr_method'] = opts.lr_method
    param['lr_rate'] = opts.lr_rate
    param['lr_decay'] = opts.lr_decay
    param['reload'] = opts.reload
    param['epoch'] = opts.epoch
    param['clp_grad'] = opts.clp_grad
    param['zero_replace'] = opts.zero_replace
    param['dropout'] = opts.dropout
    param['label_scheme'] = opts.label_scheme
    param['case_sense'] = opts.case_sense
    param['early_stop'] = opts.early_stop
    param['l2_reg_lambda'] = opts.l2_reg_lambda
    param['batch_size'] = opts.batch_size
    param['win_size'] = opts.win_size
    param['char_dim'] = opts.char_dim
    param['crf'] = opts.crf
    param['lstm_crf_hid'] = opts.lstm_crf_hid

    address = ""

    address = os.path.join(os.getcwd(), 'evaluation')
    if(not os.path.isdir(address)):
        os.makedirs(address)

    address = os.path.join(address, 'run')
    if(not os.path.isdir(address)):
        os.makedirs(address)

    address = os.path.join(address, time.strftime("%d-%m-%y_%I:%M:%S"))
    if(not os.path.isfile(address)):
        os.makedirs(address)

    print( address )

    param['init_emb'] = opts.init_emb
    param['hid_archi'] = splitNlist(opts.hid_archi,int)
    param['train'] = splitNlist(opts.train,str)
    param['vocab'] = splitNlist(opts.vocab,str)
    param['dev'] = splitNlist(opts.dev,str)
    param['test'] = splitNlist(opts.test,str)
    param['filter_size'] = splitNlist(opts.filter_size,int)
    param['no_of_filter'] = splitNlist(opts.no_of_filter,int)


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


    param['folder'] = address
    print_param(param)

    result_report = os.path.join( param['folder'], "results" )
    with codecs.open(result_report, 'w', 'utf8') as f:
        f.write("\n--- Printing the parameters ---")
        for k , v in param.items():
            f.write("\n"+str(k)+" - "+str(v))
        f.write("\n--- x ---\n\n")


    return param


def preprocess( param ):

    train_sentences, vocab_sentences, dev_sentences, test_sentences = read_datsets(param)

    if( param['zero_replace']==1 ):
        replace_digits(train_sentences, vocab_sentences, dev_sentences, test_sentences)

    update_tag_scheme(train_sentences, param['label_scheme'])
    update_tag_scheme(vocab_sentences, param['label_scheme'])
    update_tag_scheme(dev_sentences, param['label_scheme'])
    update_tag_scheme(test_sentences, param['label_scheme'])
    lower = param['case_sense']

    dico_words, word_to_id, id_to_word = word_mapping(vocab_sentences, lower)
    dico_words_train = dico_words

    # Create a dictionary and a mapping for words / POS tags / tags
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    raw_data = [train_sentences, vocab_sentences, dev_sentences, test_sentences]
    words_info = [dico_words, word_to_id, id_to_word]
    tags_info = [dico_tags, tag_to_id, id_to_tag]
    char_info = [dico_chars, char_to_id, id_to_char]

    return raw_data, words_info, tags_info, char_info


def main():

    """
        total number of options = 25.
        parameter list
        ---------------------
        --train -T          --vocab -v          --dev -D            --test -t
        --embed_dim -e      --init_emb -i       --lr_method -L      --lr_rate -l
        --lr_decay  -j      --reload -r         --hid_archi -H      --epoch -E
        --clp_grad -c       --zero_replace -z   --dropout -d        --label_scheme -s
        --case_sense -u     --early_stop -x     --filter_size -f    --no_of_filter -n
        --l2_reg_lambda -R  --batch_size -b     --win_size -w       --char_dim
        --crf
    """

    print("--- experiment taken on : ", time.strftime("%c")," ---\n\n")
    start_time = time.time()

    param = parse_parameter()
    raw_data, words_info, tags_info, char_info = preprocess(param)

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

    dico_char = char_info[0]
    char_to_id = char_info[1]
    id_to_char = char_info[2]

    train_data = prepare_dataset( train_sentences, word_to_id, char_to_id, tag_to_id )
    vocab_data = prepare_dataset( vocab_sentences, word_to_id, char_to_id, tag_to_id )
    dev_data = prepare_dataset( dev_sentences, word_to_id, char_to_id, tag_to_id )
    test_data = prepare_dataset( test_sentences, word_to_id, char_to_id, tag_to_id )

    print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

    train(param,
          train_data,
          vocab_data,
          dev_data,
          test_data,
          raw_data,
          words_info,
          tags_info,
          char_info)

    end_time = time.time()
    print("--- total execution time %s seconds ---" % end_time)



if __name__ == '__main__':
    main()
