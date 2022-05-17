from model.GCAN import GCAN
from data.Preprocess import *
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
"""
Experiment on PHEME dataset, 0-rumor, 1-non-rumor
"""
device = 'cpu'
if __name__ == "__main__":
    # 1. Data Preprocessing
    path = 'project-data/'
    train_data_path = 'train.data.jsonl'
    dev_data_path = 'dev.data.jsonl'
    test_data_path = 'test.data.jsonl'

    raw_PHEME, raw_PHEME_label = load_sort_data(path, train_data_path, dev_data_path, test_data_path)
    mini_PHEME, mini_PHEME_label = large_diffsuion_filter(PHEME=raw_PHEME, PHEME_label=raw_PHEME_label, diffuse_size=0) # TODO: you can set size filter here.

    mini_data, vectorizer = collect_dataset(mini_PHEME, mini_PHEME_label)
    dataset = PHEME_Dataset(pheme_data=mini_data, Count_Vectorizer=vectorizer, user_length=25, source_length=40)

    x_train, x_test, y_train, y_test = train_test_split(dataset, mini_PHEME_label, test_size=0.25)

    # 2. x_train is imbalanced. So resample
    count = Counter(y_train)
    class_count = np.array([count[0], count[1]])
    weight = 1. / class_count

    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


    trainLoader = torch.utils.data.DataLoader(dataset=x_train, batch_size=4)
    testLoader = torch.utils.data.DataLoader(dataset=x_test, batch_size=4)

    # 3. train model
    word_embedding_dim = dataset[0][0].shape[1]
    gcan = GCAN(gcn_in_dim=12,
                gcn_hid_dim=64,
                gcn_out_dim=256,
                source_gru_in_dim=word_embedding_dim,
                source_gru_mid_dim=512,
                source_gru_hid_dim=256,
                cnn_filter_size=3,
                cnn_in_dim=12,
                cnn_kernel_size=128,
                propagation_gru_in_dim=12,
                propagation_gru_hid_dim=128,
                source_gcn_coattn_dim=256,
                source_cnn_coattn_dim=256,
                fc_out_dim=2
                ).to(device)


    loss = torch.nn.CrossEntropyLoss()
    '''
    optimizer = torch.optim.Adam([
        {'params':base_params},
        {'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    '''

    optimizer = torch.optim.Adam(gcan.parameters(), lr=0.05, weight_decay=1e-4)
    for epoch in range(200):
        loss_sum = 0.0
        batch_num_trained = 0
        for batch in tqdm(trainLoader):
            # s, u, y
            s = batch[0].to(device)
            u = batch[1].to(device)
            y = batch[2].to(device)

            if y.shape[0] < 4:
                continue
            model_output = gcan(s, u)
            # print(y.shape,model_output.shape)
            # l = loss(model_output, y)

            l = loss(model_output, y.squeeze(dim=1))

            l.backward()
            optimizer.step()
            loss_sum += l.detach()
            batch_num_trained += 1
        print('training epoch: ', epoch, 'Loss: ', loss_sum / batch_num_trained)

        # 4. Test: We test model at every
        test_loss_sum = 0
        batch_num_tested = 0
        pred_list = []
        true_list = []
        for batch in testLoader:
            s = batch[0].to(device)
            u = batch[1].to(device)
            y = batch[2].to(device)

            if y.shape[0] < 4:
                continue
            model_output = gcan(s, u)
            test_loss_sum += loss(model_output, y.squeeze(dim=1)).detach()
            batch_num_tested += 1

            _, test_pred = model_output.max(dim=1)
            test_pred = test_pred.to('cpu').detach().tolist()
            test_true = y.squeeze(dim=1).to('cpu').detach().tolist()

            pred_list.extend(test_pred)
            true_list.extend(test_true)
            #print(test_pred, test_true)

        a = accuracy_score(true_list, pred_list)
        #p = precision_score(true_list, pred_list)
        r = recall_score(true_list, pred_list)
        f = f1_score(true_list, pred_list)
        print('a, r, f1: ', a, r, f, 'avg_loss: ', test_loss_sum/batch_num_tested)








