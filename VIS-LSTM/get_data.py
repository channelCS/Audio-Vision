from __future__ import print_function
import numpy as np
import h5py
import json
from keras.utils import to_categorical

def most_common(lst):
    return max(set(lst), key=lst.count)

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
    return v

def anat_data(input_img_h5, input_ques_h5, data_limit=215359):
    img_data = h5py.File(input_img_h5)
    ques_data = h5py.File(input_ques_h5)
  
    img_data = np.array(img_data['images_train'])
    img_pos_train = ques_data['img_pos_train'][:data_limit]
    train_img_data = np.array([img_data[_-1,:] for _ in img_pos_train])
    # Normalizing images
    tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
    train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem,(4096,1))))

    #shifting padding to left side
    ques_train = np.array(ques_data['ques_train'])[:data_limit, :]
    ques_length_train = np.array(ques_data['ques_length_train'])[:data_limit]
    ques_train = right_align(ques_train, ques_length_train)

    train_X = [train_img_data, ques_train]
    # NOTE should've consturcted one-hots using exhausitve list of answers, cause some answers may not be in dataset
    # To temporarily rectify this, all those answer indices is set to 1 in validation set
    train_y = to_categorical(ques_data['answers'])[:data_limit, :]

    return train_X, train_y

def get_val_data(input_img_h5, input_ques_h5,metadata,val_annotations_path):
    img_data = h5py.File(input_img_h5)
    ques_data = h5py.File(input_ques_h5)
    with open(val_annotations_path, 'r') as an_file:
        annotations = json.loads(an_file.read())

    img_data = np.array(img_data['images_test'])
    img_pos_train = ques_data['img_pos_test']
    train_img_data = np.array([img_data[_-1,:] for _ in img_pos_train])
    tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
    train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem,(4096,1))))

    ques_train = np.array(ques_data['ques_test'])
    ques_length_train = np.array(ques_data['ques_length_test'])
    ques_train = right_align(ques_train, ques_length_train)

    # Convert all last index to 0, coz embeddings were made that way :/
    for _ in ques_train:
        if 12602 in _:
            _[_==12602] = 0

    val_X = [train_img_data, ques_train]

    ans_to_ix = {str(ans):int(i) for i,ans in metadata['ix_to_ans'].items()}
    ques_annotations = {}
    for _ in annotations['annotations']:
        idx = ans_to_ix.get(_['multiple_choice_answer'].lower())
        _['multiple_choice_answer_idx'] = 1 if idx in [None, 1000] else idx
        ques_annotations[_['question_id']] = _

    abs_val_y = [ques_annotations[ques_id]['multiple_choice_answer_idx'] for ques_id in ques_data['question_id_test']]
    abs_val_y = to_categorical(np.array(abs_val_y))

    multi_val_y = [list(set([ans_to_ix.get(_['answer'].lower()) for _ in ques_annotations[ques_id]['answers']])) for ques_id in ques_data['question_id_test']]
    for i,_ in enumerate(multi_val_y):
        multi_val_y[i] = [1 if ans in [None, 1000] else ans for ans in _]

    return val_X, abs_val_y

def get_train_data(input_json, input_img_h5, input_ques_h5, img_vec_dim,img_norm):

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        img_feature = np.array(tem)
    
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        #-----1~82460-----
        tem = hf.get('img_pos_train')
        # convert into 0~82459
        train_data['img_list'] = np.array(tem)[:215359]
        
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1
        
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
        img_feature = np.divide(img_feature, np.tile(tem,(1,4096)))

    return dataset, img_feature, train_data



def get_test_data(input_json,input_img_h5,input_ques_h5,img_vec_dim,ans_file,img_norm):
    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_test')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        # -----1~82460-----
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)[:215359]-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
        # MC_answer_test
        tem = hf.get('MC_ans_test')
        test_data['MC_ans_test'] = np.array(tem)

        tem =  np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
        img_feature = np.divide(img_feature, np.tile(tem,(1,img_vec_dim)))


    # Added by Adi, make sure the ans_file is provided
    nb_data_test = len(test_data[u'question'])
    val_all_answers_dict = json.load(open(ans_file))
    val_answers = np.zeros(nb_data_test, dtype=np.int32)

    ans_to_ix = {v: k for k, v in dataset[u'ix_to_ans'].items()}
    count_of_not_found = 0
    for i in xrange(nb_data_test):
        qid = test_data[u'ques_id'][i]
        try : 
            val_ans_ix =int(ans_to_ix[most_common(val_all_answers_dict[str(qid)])]) -1
        except KeyError:
            count_of_not_found += 1
            val_ans_ix = 480
        val_answers[i] = val_ans_ix
    print("Beware: " + str(count_of_not_found) + " number of val answers are not really correct")

    return dataset, img_feature, test_data, val_answers