import codecs
import re
import time
from support import *
from collections import deque
import numpy as np
np.set_printoptions(threshold=np.nan)
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


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    itr = 0 ;
    for s in sentences:
        str_words = [w[0] for w in s]
        k = []
        for w in str_words:
            if w in word_to_id:
                k += [word_to_id[w]]
            else:
                k += [word_to_id['<UNK>']]
        words = k
        j = []
        for w in str_words:
            k = []
            for c in w:
                # print(c,"",end="")
                #
                if c in char_to_id:       
                        k += [char_to_id[c]]
                else:
                    k += [char_to_id['<UNK>']]
            j += [k]
        chars = j
        assert(len(words)==len(chars))
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags,
        })

    return data

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

def loademb(emb_param,embed_dim):
    print("Loading pretrained model ... .. .")
    tic = time.time()
    Word2Vec={}
    itr = 0 ;
    address = emb_param
    for line in codecs.open(address, 'r', 'utf8'):
        line = line.rstrip()
        line = line.split()
        if( len(line) !=  embed_dim+1 ):
            continue
        k = []
        for i in range(embed_dim):
            k.append(float(line[i+1]))
        Word2Vec[line[0]] = k
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

def prepare_input_rnn(data):
    # data :==: batch , array of dictionary
    # each dictionary is for one sentence
    # each dictionary contains
    #   1. word_ids
    #   2. sequence_lengths
    #   3. char_ids
    #   4. word_lengths
    #   5. labels
    word_ids = [ i['words'] for i in data ]
    sequence_lengths = []
    char_ids = [i['chars'] for i in data]
    labels = [ i['tags'] for i in data]

    # calculate maximum length of the sentences and word.
    # populate the sequence_lengths and word_lengths
    max_sent_len = max_word_len = 0
    for i, j in enumerate(word_ids):
        # j is an list of word id, index i represents a sentence
        temp = len(j)
        max_sent_len = max( max_sent_len , temp )
        sequence_lengths += [temp]

        for k in char_ids[i]:
            # k is an list of character id
            max_word_len = max(max_word_len,len(k))

    char_ids_ret = np.zeros((len(word_ids), max_sent_len,max_word_len))
    word_lengths = np.zeros((len(word_ids), max_sent_len))

    # padding word_ids and each element of char_ids
    for i, j in enumerate(word_ids):
        # j is an list of word id, index i represents a sentence

        temp = word_ids[i][:max_sent_len] + [0]* (max_sent_len-len(j))
        word_ids[i] = temp

        temp = labels[i][:max_sent_len] + [0] * (max_sent_len - len(j))
        labels[i] = temp

        word_to_char = char_ids[i]

        for k,_ in enumerate(word_to_char):
            # _ is an list of character id representing a word
            temp = word_to_char[k][:max_word_len] + [0] *(max_word_len - len(_))
            word_lengths[i][k] = len(_)
            char_ids_ret[i][k] = temp


    word_ids = np.asarray(word_ids)
    char_ids_ret = np.asarray(char_ids_ret)
    sequence_lengths = np.asarray(sequence_lengths)
    word_lengths = np.asarray(word_lengths)
    labels = np.asarray(labels)

    return word_ids, sequence_lengths, char_ids_ret, word_lengths, labels


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


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...',ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word




def get_embeddings(param, vocabulary_size, no_of_class, dico_words, word_to_id):
    cnt1 = cnt2 = cnt3 = cnt4 = 0
    if( len(param['init_emb']) <= 2 ):
        #vector initialize by xavier initializer
        Word2Vec = tf.get_variable(shape=[vocabulary_size, param['embed_dim']], initializer=tf.contrib.layers.xavier_initializer())
        # embedding = tf.random_uniform([vocabulary_size, param['embed_dim']], -1.0, 1.0)
        print("embedding vector size : ",Word2Vec.get_shape())
    else:
        #vector initialized by pretrained embedding
        Word2Vec = np.zeros((vocabulary_size,param['embed_dim']))
        Word2Vec_ = loademb( param['init_emb'],param['embed_dim'] )
        for k,v in dico_words.items():
            if( k in Word2Vec_):
                vector = Word2Vec_[k]
                Word2Vec[ word_to_id[k] ] = vector
                cnt1 += 1 ;
            elif( k.lower() in Word2Vec_ ):
                vector = Word2Vec_[ k.lower() ]
                Word2Vec[ word_to_id[ k ] ] = vector
                cnt2 += 1 ;
            elif( re.sub('\d', '0', k.lower()) in Word2Vec_ ):
                vector = Word2Vec_[ re.sub('\d', '0', k.lower()) ]
                Word2Vec[ word_to_id[ k ] ] = vector
                cnt3 += 1
            else:
                # vector = np.random.normal(size=param['embed_dim'])
                vector = np.zeros(param['embed_dim'])
                Word2Vec[ word_to_id[k] ] = vector
                cnt4 += 1

        print("embedding vector size : ",Word2Vec.shape)
    print("vector initialized by preemb", cnt1)
    print("vector initialized lowering cnaracter", cnt2)
    print("vector initialized substituting digits", cnt3)
    print("vector initialized by zero replacing", cnt4)
    print("Percent of vector initialized ", (cnt1+cnt2+cnt3)*100/(cnt1+cnt2+cnt3+cnt4))
    return Word2Vec


def get_minibatch(data,batch_size):
    batch = []
    for k in data:
        if( len(batch) == batch_size ):
            yield batch
            batch = []
        batch.append(k)
    if( len(batch) > 0 ):
        yield batch
