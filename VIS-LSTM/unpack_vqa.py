from random import shuffle, seed
import numpy as np
import scipy.io
import h5py
from nltk.tokenize import word_tokenize
import json
import re

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def get_tokens(imgs,token_method,print_proc=False):
    # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['question']
        if token_method == 'nltk':
            txt = word_tokenize(str(s).lower())
        else:
            txt = tokenize(s)

        img['processed_tokens'] = txt
        #prints only first 10 tokenized questions
        if i < 10: print txt
        if print_proc:
            if i % 1000 == 0:
                print "processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs))
    return imgs

def build_vocab_question(imgs, count_thr):
    # build vocabulary for question and answers.
    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str,cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print 'number of words in vocab would be %d' % (len(vocab), )
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')
  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs, vocab

def apply_vocab_question(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if w in wtoi else 'UNK' for w in txt] # Replace unknown words by 'UNK'
        img['final_question'] = question

    return imgs

def get_top_answers(imgs, params):
    counts = {}
    for img in imgs:
        ans = img['ans'] 
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top answer and their counts:'    
    print '\n'.join(map(str,cw[:20]))
    
    vocab = []
    for i in range(params['num_ans']):
        vocab.append(cw[i][1])

    return vocab[:params['num_ans']]

def encode_question(imgs, max_length, wtoi):

    N = len(imgs) # no of questions

    label_arrays = np.zeros((N, max_length), dtype='uint32') # 2D Array of size (N,max_length)
    label_length = np.zeros(N, dtype='uint32') # 1D Array of size N
    question_id = np.zeros(N, dtype='uint32') # 1D Array of size N that would contain all questions ids
    question_counter = 0
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['ques_id']
        label_length[question_counter] = min(max_length, len(img['final_question'])) # no of words in question
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w] # each word has an id, it creates an array with id of word
    
    return label_arrays, label_length, question_id


def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi.get(img['ans'], -1) # -1 means wrong answer.

    return ans_arrays

def encode_mc_answer(imgs, atoi):
    N = len(imgs)
    mc_ans_arrays = np.zeros((N, 18), dtype='uint32')

    for i, img in enumerate(imgs):
        for j, ans in enumerate(img['MC_ans']):
            mc_ans_arrays[i,j] = atoi.get(ans, 0)
    return mc_ans_arrays

def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if img['ans'] in atoi:
            new_imgs.append(img)

    print 'question number reduce from %d to %d '%(len(imgs), len(new_imgs))
    return new_imgs

def get_unqiue_img(imgs):
    count_img = {}
    N = len(imgs) # no of questions
    img_pos = np.zeros(N, dtype='uint32') # 1D Array of size N
    ques_pos_tmp = {}
    for img in imgs:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1 # no of instances of an image

    unique_img = [w for w,n in count_img.iteritems()]
    imgtoi = {w:i for i,w in enumerate(unique_img)} # add one for torch, since torch start from 1.

    for i, img in enumerate(imgs):
        idx = imgtoi.get(img['img_path'])
        img_pos[i] = idx

        if idx-1 not in ques_pos_tmp:
            ques_pos_tmp[idx-1] = []

        ques_pos_tmp[idx-1].append(i+1)
    
    img_N = len(ques_pos_tmp)
    ques_pos = np.zeros((img_N,3), dtype='uint32')
    ques_pos_len = np.zeros(img_N, dtype='uint32')

    for idx, ques_list in ques_pos_tmp.iteritems():
        ques_pos_len[idx] = len(ques_list)
        for j in range(len(ques_list)):
            ques_pos[idx][j] = ques_list[j]
    return unique_img, img_pos, ques_pos, ques_pos_len


############# DEFINE PATHS AND PARAMATERS ##########################
imdir='%s/COCO_%s_%012d.jpg'
q='D:/workspace/aditya_akshita/vqa/VQA_Keras/data/'
imgs_train_path = q+'vqa_raw_train.json'
imgs_test_path = q+'vqa_raw_test.json'
token_method = 'other'
word_count_threshold = 0
max_length=26
output_json=q+'vqa_data_prepro.json'
output_h5=q+'vqa_data_prepro.h5'
prep='dev'


######### PREPARE DATA #############################################
if prep=='dev':
    train_anno = json.load(open(q+'annotations/mscoco_train2014_annotations.json', 'r'))
    val_anno = json.load(open(q+'annotations/mscoco_val2014_annotations.json', 'r'))
    
    train_ques = json.load(open(q+'annotations/MultipleChoice_mscoco_train2014_questions.json', 'r'))
    val_ques = json.load(open(q+'annotations/MultipleChoice_mscoco_val2014_questions.json', 'r'))
    subtype = 'train2014'
    train,test=[],[]
    for i in range(len(train_anno['annotations'])):
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        question_id = train_anno['annotations'][i]['question_id']
        image_path = imdir%(subtype, subtype, train_anno['annotations'][i]['image_id'])
        
        question = train_ques['questions'][i]['question']
        mc_ans = train_ques['questions'][i]['multiple_choices']
        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
    
    subtype = 'val2014'
    for i in range(len(val_anno['annotations'])):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        image_path = imdir%(subtype, subtype, val_anno['annotations'][i]['image_id'])
    
        question = val_ques['questions'][i]['question']
        mc_ans = val_ques['questions'][i]['multiple_choices']
    
        test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
elif prep=='eval':
    train_anno = json.load(open(q+'annotations/mscoco_train2014_annotations.json', 'r'))
    val_anno = json.load(open(q+'annotations/mscoco_val2014_annotations.json', 'r'))

    train_ques = json.load(open(q+'annotations/MultipleChoice_mscoco_train2014_questions.json', 'r'))
    val_ques = json.load(open(q+'annotations/MultipleChoice_mscoco_val2014_questions.json', 'r'))
    test_ques = json.load(open(q+'annotations/MultipleChoice_mscoco_test2015_questions.json', 'r'))
    
    subtype = 'train2014'
    for i in range(len(train_anno['annotations'])):
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        question_id = train_anno['annotations'][i]['question_id']
        image_path = imdir%(subtype, subtype, train_anno['annotations'][i]['image_id'])

        question = train_ques['questions'][i]['question']
        mc_ans = train_ques['questions'][i]['multiple_choices']

        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})

    subtype = 'val2014'
    for i in range(len(val_anno['annotations'])):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        image_path = imdir%(subtype, subtype, val_anno['annotations'][i]['image_id'])

        question = val_ques['questions'][i]['question']
        mc_ans = val_ques['questions'][i]['multiple_choices']

        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
    
    subtype = 'test2015'
    for i in range(len(test_ques['questions'])):
        question_id = test_ques['questions'][i]['question_id']
        image_path = imdir%(subtype, subtype, test_ques['questions'][i]['image_id'])

        question = test_ques['questions'][i]['question']
        mc_ans = test_ques['questions'][i]['multiple_choices']

        test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans})

print 'Training sample %d, Testing sample %d...' %(len(train), len(test))

json.dump(train, open(imgs_train_path, 'w'))
json.dump(test, open(imgs_test_path, 'w'))



###########    PREPROCESS DATA   #################
imgs_train = json.load(open(imgs_train_path, 'r'))
imgs_test = json.load(open(imgs_test_path, 'r'))
num_ans=1000

counts = {}
x=0
for img in imgs_train:
    ans = img['ans'] 
    counts[ans] = counts.get(ans, 0) + 1 # Counts no of occurences of each ans

#Sort in descending order(Maximum occurence words at the top)
cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
print 'top answer and their counts:'    
print '\n'.join(map(str,cw[:20]))

vocab = [] # All unique answers are put in vocab
for i in range(num_ans):
    vocab.append(cw[i][1])
    
top_ans = vocab[:num_ans] #Slice to get n=num_ans top answers

#Make dict with id from 1 to n=num_ans
atoi = {w:i for i,w in enumerate(top_ans)}
itoa = {i:w for i,w in enumerate(top_ans)}

# filter question, which isn't in the top answers.
new_imgs = []
for i, img in enumerate(imgs_train):
    if img['ans'] in atoi:
        new_imgs.append(img)

print 'question number reduce from %d to %d '%(len(imgs_train), len(new_imgs))
imgs_train=new_imgs
########################################################################
# tokenization and preprocessing training question
imgs_train = get_tokens(imgs_train, token_method, print_proc=True)
# tokenization and preprocessing testing question
imgs_test = get_tokens(imgs_test, token_method, print_proc=True)

#######################################################################
imgs_train, vocab = build_vocab_question(imgs_train, word_count_threshold)
#######################################################################

itow = {i:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
wtoi = {w:i for i,w in enumerate(vocab)} # inverse table

ques_train, ques_length_train, question_id_train = encode_question(imgs_train, max_length, wtoi)

imgs_test = apply_vocab_question(imgs_test, wtoi)
ques_test, ques_length_test, question_id_test = encode_question(imgs_test, max_length, wtoi)
###################################################################

# get the unique image for train and test
unique_img_train, img_pos_train, ques_pos_train, ques_pos_len_train = get_unqiue_img(imgs_train)
unique_img_test, img_pos_test, ques_pos_test, ques_pos_len_test = get_unqiue_img(imgs_test)


# get the answer encoding.
ans_train = encode_answer(imgs_train, atoi)

ans_test = encode_answer(imgs_test, atoi)
MC_ans_test = encode_mc_answer(imgs_test, atoi)

# get the split
N_train = len(imgs_train)
N_test = len(imgs_test)
# since the train image is already suffled, we just use the last val_num image as validation
# train = 0, val = 1, test = 2
split_train = np.zeros(N_train)
#split_train[N_train - params['val_num']: N_train] = 1

split_test = np.zeros(N_test)
split_test[:] = 2

# create output h5 file for training set.
f = h5py.File(output_h5, "w")
f.create_dataset("ques_train", dtype='uint32', data=ques_train)
f.create_dataset("ques_test", dtype='uint32', data=ques_test)

f.create_dataset("answers", dtype='uint32', data=ans_train)
f.create_dataset("ans_test", dtype='uint32', data=ans_test)

f.create_dataset("ques_id_train", dtype='uint32', data=question_id_train)
f.create_dataset("ques_id_test", dtype='uint32', data=question_id_test)

f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)


f.create_dataset("ques_pos_train", dtype='uint32', data=ques_pos_train)
f.create_dataset("ques_pos_test", dtype='uint32', data=ques_pos_test)

f.create_dataset("ques_pos_len_train", dtype='uint32', data=ques_pos_len_train)
f.create_dataset("ques_pos_len_test", dtype='uint32', data=ques_pos_len_test)

f.create_dataset("split_train", dtype='uint32', data=split_train)
f.create_dataset("split_test", dtype='uint32', data=split_test)

f.create_dataset("ques_len_train", dtype='uint32', data=ques_length_train)
f.create_dataset("ques_len_test", dtype='uint32', data=ques_length_test)
f.create_dataset("MC_ans_test", dtype='uint32', data=MC_ans_test)

f.close()
print 'wrote ', output_h5


# create output json file
out = {}
out['ix_to_word'] = itow # encode the (1-indexed) vocab
out['ix_to_ans'] = itoa
out['unique_img_train'] = unique_img_train
out['uniuqe_img_test'] = unique_img_test
json.dump(out, open(output_json, 'w'))
print 'wrote ', output_json
