import torch
import jsonlines
import json
from nltk.corpus import stopwords
import string
import numpy as np
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer

# Wed Jan 07 14:04:29 +0000 2015
def time_stamp(time):
    # given a tweet dict
    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', "Aug", 'Sep', 'Oct', 'Nov', 'Dec']
    #time = dic['created_at']
    month = time[4:7]
    day = time[8:10]
    h = time[11:13]
    m = time[14:16]
    s = time[17:19]
    return month_list.index(month)*86400*30 + int(day)*86400 + int(h)*3600 + int(m)*60 + int(s)*1



# Load training data as train_data， write into train_file, test_file and label_file
# Load PHEME dataset.


path = 'project-data/'
train_data_path = 'train.data.jsonl'
dev_data_path = 'dev.data.jsonl'
test_data_path = 'test.data.jsonl'

def load_sort_data(path, train_data_path, dev_data_path, test_data_path):
    """
    Load and sort data, return PHEME and PHEME_label
    """
    train_data = []
    with jsonlines.open(path + train_data_path) as reader:
        for obj in reader:
            train_data.append(obj)
    print('length of traininig data:', len(train_data))

    # load the development data as dev_data(used as test_data in RvNN)
    dev_data = []
    with jsonlines.open(path + dev_data_path) as reader1:
        for obj in reader1:
            dev_data.append(obj)
    print('length of devolop data: ', len(dev_data))

    test_data = []
    with jsonlines.open(path + test_data_path) as reader2:
        for obj in reader2:
            test_data.append(obj)
    print('length of test data: ', len(test_data))


    # Sort the three dataset in time_stamp order


    for idx in range(len(train_data)):
        train_data[idx] = sorted(train_data[idx], key=lambda tree_node: time_stamp(tree_node['created_at']))

    for idx in range(len(test_data)):
        test_data[idx] = sorted(test_data[idx], key=lambda tree_node: time_stamp(tree_node['created_at']))

    for idx in range(len(dev_data)):
        dev_data[idx] = sorted(dev_data[idx], key=lambda tree_node: time_stamp(tree_node['created_at']))

    PHEME = train_data + dev_data + test_data
    print('length of whole data: ', len(PHEME))


    label_path = 'PHEME_label.json'
    with open(path + label_path) as f:
        PHEME_label = json.load(f)

    return PHEME, PHEME_label


def large_diffsuion_filter(PHEME, PHEME_label, diffuse_size):
    # The diffusions in PHEME can be very small, lets select those with larger diffsuion size.
    # Filter dataset with diffusion size larger than diffuse_size.
    mini_PHEME = []
    mini_PHEME_label = []

    for sample in PHEME:
        if len(sample) >= diffuse_size:
            mini_PHEME.append(sample)
            if PHEME_label[sample[0]['id_str']] == 'rumour':
                mini_PHEME_label.append(0)
            else:
                mini_PHEME_label.append(1)
    assert len(mini_PHEME) == len(mini_PHEME_label)
    print('number of rumors: ', len([i for i in mini_PHEME_label if i == 0]), 'over all samples: ', len(mini_PHEME))
    return mini_PHEME, mini_PHEME_label


def extract_usr_attributes(post_dict, source_post_time):
    user_dict = post_dict['user']
    reply_post_time = time_stamp(post_dict['created_at'])
    # Extract User Attributes from user dictionary.
    rep = []
    # 1.
    rep.append(1 if user_dict['profile_use_background_image'] else 0)
    # 2.
    rep.append(1 if user_dict['verified'] else 0)
    # 3.
    rep.append(user_dict['followers_count'])
    # 4.
    rep.append(user_dict['listed_count'])
    # 5.
    rep.append(user_dict['statuses_count'])
    # 6. Holy some users have no descriptions (NoneType)
    if user_dict['description'] is not None:
        rep.append(len(user_dict['description']))
    else:
        rep.append(0)
    # 7.
    rep.append(user_dict['friends_count'])
    # 8.
    rep.append(1 if user_dict['geo_enabled'] else 0)
    # 9.
    rep.append(1 if user_dict['profile_background_tile'] else 0)
    # 10.
    rep.append(user_dict['favourites_count'])
    # 11.
    rep.append(1 if user_dict['contributors_enabled'] else 0)
    # 12. Reply time over the source post
    rep.append(reply_post_time - source_post_time)
    return rep

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)

def collect_dataset(mini_PHEME, mini_PHEME_label):
    mini_data = []
    for idx in range(len(mini_PHEME)):
        sample = mini_PHEME[idx]
        cur_dict = {}
        cur_dict['source_text'] = sample[0]['text']
        user_info = []
        source_post_time = time_stamp(sample[0]['created_at'])
        for post in sample:
            user_info.append(extract_usr_attributes(post, source_post_time))
        cur_dict['user_info'] = user_info
        cur_dict['y'] = 0 if not mini_PHEME_label[idx] else 1
        mini_data.append(cur_dict)

    assert len(mini_data) == len(mini_PHEME)
    for sam_dict in mini_data:
        sam_dict['source_text'] = clean_doc(sam_dict['source_text'])

    all_texts = [i['source_text'] for i in mini_data]
    vectorizer = CountVectorizer()
    # tokenize and build vocab
    vectorizer.fit(all_texts)

    return mini_data, vectorizer


class PHEME_Dataset(Dataset):
    def __init__(self, pheme_data, Count_Vectorizer, user_length=25, source_length=40):
        self.raw_data = pheme_data

        self.user_length = user_length
        self.source_length = source_length
        self.vectorizer = Count_Vectorizer

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        # get source_text representation and user representation to same length.
        source_text = self.raw_data[index]['source_text']
        source_list = source_text.split()
        source_rep = self.vectorizer.transform(source_list).toarray()
        if len(source_rep) >= self.source_length:
            source_rep = source_rep[:self.source_length]
        else:  # source_rep is too short
            # print(self.source_length, len(source_rep), len(source_rep[0]))
            zero_pad = np.zeros([self.source_length - len(source_rep), len(source_rep[0])])
            source_rep = np.concatenate((source_rep, zero_pad), axis=0)

        user_rep = self.raw_data[index]['user_info']
        if len(user_rep) >= self.user_length:
            new_user_rep = user_rep[:self.user_length]
        else:
            idxs = np.random.choice(len(user_rep), self.user_length, replace=True)
            idxs = sorted(idxs)
            new_user_rep = [user_rep[i] for i in idxs]

        label = [self.raw_data[index]['y']]

        return torch.Tensor(source_rep), torch.Tensor(new_user_rep), torch.LongTensor(label)




