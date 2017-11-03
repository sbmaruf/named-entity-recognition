import os
import re
import codecs
import numpy as np
import tensorflow as tf
eval_path = "./evaluation"
eval_script = os.path.join(eval_path, "conlleval")


def num2optimizer(i):
    if(i == 0):
        return "SGD"
    elif(i == 1):
        return "Adadelta"
    elif (i == 2):
        return "Adagrad"
    elif (i == 3):
        return "AdagradDA"
    elif (i == 4):
        return "Momentum"
    elif (i == 5):
        return "Adam"
    elif (i == 6):
        return "ftrl"
    elif (i == 7):
        return "ProximalSGD"
    elif (i == 8):
        return "ProximalAdagrad"
    elif (i == 9):
        return "RMSProp"
    else:
        return None


def get_batch_input(batch_no,train_X,train_Y,batch_size=100):
    row_st = batch_no*batch_size
    if( row_st > len(train_X) ):
        return [],[]
    return train_X[row_st:min(row_st+batch_size,len(train_X)+1)],train_Y[row_st:min(row_st+batch_size,len(train_Y)+1)]



# def one_hot_embedding(vector,one_hot_hash):
#     retVec = []
#     for idx,val in enumerate(vector):
#         tmp_one_hot = np.zeros(len(one_hot_hash))
#         if val in one_hot_hash:
#             tmp_one_hot[ one_hot_hash[val] ] = 1
#         else:
#             tmp_one_hot[ -1 ] = 1
#         retVec.append(tmp_one_hot)
#     return retVec

def one_hot_embedding( vec , vz):
    retVec = []
    for idx,val in enumerate(vec):
        tmp_one_hot = np.zeros(vz)
        tmp_one_hot[ val ] = 1
        retVec.append(tmp_one_hot)
    return retVec


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags



def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    dico[''] = 10000001
    word_to_id, id_to_word = create_mapping(dico)
    print ("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def splitNlist(str,f):
    temp = str.split()
    ret = []
    for i in temp:
        ret.append(f(i))
    return ret


def print_param(param):
    print("--- Printing the parameters ---")
    for k , v in param.items():
        print(k,"-",v)
    print("--- x ---\n")


def save_model(sess, saver, param_folder, saved_ckpt):
    print("\nNew best score on dev.")
    print("Saving model to disk...")
    address = os.path.join(param_folder, 'model')
    if(not os.path.isdir(address)):
        os.makedirs(address)
    address = os.path.join(address, saved_ckpt )
    save_path = saver.save(sess, address)
    print("Model saved in file: %s" % save_path)
