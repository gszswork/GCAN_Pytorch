from model.GCAN import GCAN
from data.Preprocess import *
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
"""
Experiment on PHEME dataset, 0-rumor, 1-non-rumor
"""

if __name__ == "__main__":
    # 1. Data Preprocessing
    path = 'project-data/'
    train_data_path = 'train.data.jsonl'
    dev_data_path = 'dev.data.jsonl'
    test_data_path = 'test.data.jsonl'

    raw_PHEME, raw_PHEME_label = load_sort_data(path, train_data_path, dev_data_path, test_data_path)
    mini_PHEME, mini_PHEME_label = large_diffsuion_filter(raw_PHEME, raw_PHEME_label, 25) # mini in size but larger diffusion.

    mini_data, vectorizer = collect_dataset(mini_PHEME, mini_PHEME_label)
    dataset = PHEME_Dataset(mini_data, vectorizer, 25, 40)

    x_train, x_test, y_train, y_test = train_test_split(dataset, mini_PHEME_label, test_size=0.25)

    # 2. x_train is imbalanced. So resample
    count = Counter(y_train)
    class_count = np.array([count[0], count[1]])
    weight = 1. / class_count

    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


    trainLoader = torch.utils.data.DataLoader(dataset=x_train, batch_size=4, sampler=sampler)
    testLoader = torch.utils.data.DataLoader(dataset=x_test, batch_size=4)

    # 3. train model
    gcan = GCAN(gcn_in_dim=12,
                gcn_hid_dim=64,
                gcn_out_dim=128,
                source_gru_in_dim=3347,
                source_gru_hid_dim=32,
                cnn_filter_size=3,
                cnn_in_dim=12,
                cnn_kernel_size=32,
                propagation_gru_in_dim=12,
                propagation_gru_hid_dim=32,
                source_gcn_coattn_dim=64,
                source_cnn_coattn_dim=64,
                fc_out_dim=2
                )

    device = 'cpu'
    loss = torch.nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(gcan.parameters(), 0.001)
    for epoch in range(200):
        loss_sum = 0.0
        batch_num_trained = 0
        for batch in tqdm(trainLoader):
            # s, u, y
            s = batch[0].to(device)
            u = batch[1].to(device)
            y = batch[2].to(device)

            model_output = gcan(s, u)
            # print(y.shape,model_output.shape)
            # l = loss(model_output, y)

            l = loss(model_output, y.squeeze(dim=1))

            l.backward()
            optimizer.step()
            loss_sum += l.detach()
            batch_num_trained += 1
        print(loss_sum / batch_num_trained)

        # 4. Test: We test model at every
        test_loss_sum = 0
        batch_num_tested = 0
        pred_list = []
        true_list = []
        for batch in tqdm(testLoader):
            s = batch[0].to(device)
            u = batch[1].to(device)
            y = batch[2].to(device)


            model_output = gcan(s, u)
            test_loss_sum += loss(model_output, y.squeeze(dim=1)).detach()
            batch_num_tested += 1

            _, test_pred = model_output.max(dim=1)
            test_pred = test_pred.to('cpu').detach().tolist()
            test_true = y.squeeze(dim=1).to('cpu').detach().tolist()

            pred_list.extend(test_pred)
            true_list.extend(test_true)
            #print(test_pred, test_true)

        print(true_list[:10], pred_list[:10])
        a = accuracy_score(true_list, pred_list)
        #p = precision_score(true_list, pred_list)
        r = recall_score(true_list, pred_list)
        f = f1_score(true_list, pred_list)
        print('a, r, f1: ', a, r, f, 'avg_loss: ', test_loss_sum/batch_num_tested)








