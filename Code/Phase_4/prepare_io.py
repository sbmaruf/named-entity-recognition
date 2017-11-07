import codecs
import re
import time
from support import *
from collections import deque

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def update_tag_scheme1(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def load_sentences(path, lower=1, zeros=1):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    temp = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def read_path(address):
    """
    Read the data from the source address and return the sentences
    :param address: a path to the source file
    :return: <list <list>> each row of the list(a sentence) consists of another list ( word(s) of that sentence )
    """
    sentences = []
    sentence = []
    for line in codecs.open(address, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            sentence.append(word)
            assert len(word) >= 2
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    return sentences


def read_datsets(param):
    """
    read the train, vocab, deb, test dataset from addressed
    :param param: <dictionary> the dictionary of the parameter
    :return: <list> train_sentences, <list> vocab_sentences, <list> dev_sentences, <list> test_sentences
    """
    train_sentences = []
    for address in param['train']:
        train_sentences = train_sentences + read_path(address)

    vocab_sentences = []
    for address in param['vocab']:
        vocab_sentences = vocab_sentences + read_path(address)

    dev_sentences = []
    for address in param['dev']:
        dev_sentences = dev_sentences + read_path(address)

    test_sentences = []
    for address in param['test']:
        test_sentences = test_sentences + read_path(address)

    return  train_sentences, vocab_sentences, dev_sentences, test_sentences


def replace_digits(train_sentences, vocab_sentences, dev_sentences, test_sentences):
    for i, sentence in enumerate(train_sentences):
        for j, word in enumerate(sentence):
            sentence[j][0] = re.sub('\d', '0', sentence[j][0])
        train_sentences[i] = sentence

    for i, sentence in enumerate(vocab_sentences):
        for j, word in enumerate(sentence):
            sentence[j][0] = re.sub('\d', '0', sentence[j][0])
        vocab_sentences[i] = sentence

    for i, sentence in enumerate(dev_sentences):
        for j, word in enumerate(sentence):
            sentence[j][0] = re.sub('\d', '0', sentence[j][0])
        dev_sentences[i] = sentence

    for i, sentence in enumerate(test_sentences):
        for j, word in enumerate(sentence):
            sentence[j][0] = re.sub('\d', '0', sentence[j][0])
        test_sentences[i] = sentence


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 1: #'iob'
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 2: #'iobes'
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def prepare_dataset(sentences, word_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]

        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'caps': caps,
            'tags': tags,
        })
    return data


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all ca    os.system("mkdir %s" % param['folder'])ps
    2 = first letter caps
    3 = one capital (not first letter)
    """

    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

def loademb(emb_param):
    tic = time.time()
    Word2Vec={}
    itr = 0 ;
    address = emb_param
    for line in codecs.open(address, 'r', 'utf8'):
        line = line.rstrip()
        line = line.split()
        itr = itr+1
        if( itr == 1 ):
            continue
        Word2Vec[line[0]] = []
        for i in range(100):
            Word2Vec[line[0]].append(float(line[i+1]))
    toc = time.time()
    print("skip-gram vector loading time ", toc-tic , " (s)")
    return Word2Vec

def prepare_input(data,vocabulary_size,no_of_class,isCaseSense):

    cur_X = data['words']
    cap_in = data['caps']
    cur_Y = data['tags']

    # cur_X = one_hot_embedding(cur_X, vocabulary_size)
    cur_Y = one_hot_embedding(cur_Y, no_of_class)

    cur_X = np.asarray(cur_X)
    cur_Y = np.asarray(cur_Y)
    cur_cap = []
    for j in cap_in:
        if (isCaseSense == 0):
            j = 0
        cur_cap.append([j])
    cur_cap = np.asarray(cur_cap)
    return cur_X,cur_Y,cur_cap

def prepare_input_rnn(data,vocabulary_size,no_of_class,isCaseSense):

    cur_X = data['words']
    cap_in = data['caps']
    cur_Y = data['tags']

    # cur_X = one_hot_embedding(cur_X, vocabulary_size)
    cur_X = [cur_X]
    cur_Y = [cur_Y]

    cur_X = np.asarray(cur_X)
    cur_Y = np.asarray(cur_Y)

    cur_cap = []
    for j in cap_in:
        if (isCaseSense == 0):
            j = 0
        cur_cap.append([j])
    cur_cap = np.asarray(cur_cap)

    seq_len = []
    for idx,val in enumerate(cur_X):
        seq_len.append(val.size)
    seq_len = np.asarray(seq_len)

    return cur_X,cur_Y,seq_len,cur_cap

def prepare_input_CNN(data,vocabulary_size,no_of_class,isCaseSense,win_size,padding_idx):

    cur_X = data['words']
    cap_in = data['caps']
    cur_Y = data['tags']
    #making window
    #   for a 5 size window
    #   <UNK> <UNK> 1 2 3
    l = int(win_size/2)
    sin_input = deque()
    for i in range(l):
        sin_input.append(padding_idx)

    for idx, val in enumerate(cur_X):
        if(l == win_size):
            break
        sin_input.append(val)
        l += 1
    while( l != win_size ):
        sin_input.append(padding_idx)
        l += 1
    # sliding the window
    l = int(win_size/2)
    tot = len(cur_X)
    X = []
    for idx, val in enumerate(cur_X):
        try:
            X.append(np.asarray(sin_input))
        except:
            print("Cur X" , cur_X)
            print("\n\n")
            print("singular input" , sin_input)
        sin_input.popleft()
        nxt = idx+l+1
        if( nxt < tot ):
            sin_input.append(cur_X[nxt])
        else:
            sin_input.append(padding_idx)

    cur_Y = one_hot_embedding(cur_Y, no_of_class)

    X = np.asarray(X)
    cur_Y = np.asarray(cur_Y)
    cur_cap = []
    for j in cap_in:
        if (isCaseSense == 0):
            j = 0
        cur_cap.append([j])
    cur_cap = np.asarray(cur_cap)

    return X,cur_Y,cur_cap
