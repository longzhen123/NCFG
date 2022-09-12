import time
import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from src.evaluate import get_all_metrics
from src.load_base import load_data, get_records


class NCFG(nn.Module):

    def __init__(self, dim, n_entity, n_relation, L, n_user, n_item):

        super(NCFG, self).__init__()
        self.dim = dim
        rec_user_embedding_mat = t.randn(n_user, dim)
        entity_embedding_mat = t.randn(n_entity, dim)
        relation_embedding_mat = t.randn(n_relation, dim)
        rec_item_embedding_mat = t.randn(n_item, dim)
        nn.init.xavier_uniform_(entity_embedding_mat)
        nn.init.xavier_uniform_(relation_embedding_mat)
        nn.init.xavier_uniform_(rec_item_embedding_mat)
        nn.init.xavier_uniform_(rec_user_embedding_mat)
        self.entity_embedding_mat = nn.Parameter(entity_embedding_mat)
        self.rec_user_embedding_mat = nn.Parameter(rec_user_embedding_mat)
        self.relation_embedding_mat = nn.Parameter(relation_embedding_mat)
        self.rec_item_embedding_mat = nn.Parameter(rec_item_embedding_mat)
        self.L = L
        self.rnn = nn.RNN(2*dim, dim, num_layers=1, nonlinearity='tanh')

    def forward(self, pairs, ripple_sets):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]

        heads_list, relations_list, tails_list = self.get_head_relation_and_tail(users, ripple_sets)
        items_list = self.get_items_list(users, ripple_sets)
        user_embeddings = self.get_vector(users, heads_list, relations_list, tails_list, items_list) + self.rec_user_embedding_mat[users]
        item_embeddings = self.entity_embedding_mat[items] + self.rec_item_embedding_mat[items]

        predict = (user_embeddings * item_embeddings).sum(dim=1)
        return t.sigmoid(predict)

    def get_items_list(self, users, ripple_sets):
        items_list = []

        for user in users:
            items_list.extend(ripple_sets[user][0])

        return items_list

    def get_head_relation_and_tail(self, users, ripple_sets):

        heads_list = []
        relations_list = []
        tails_list = []
        for l in range(1, self.L+1):
            l_head_list = []
            l_relation_list = []
            l_tail_list = []

            for user in users:

                l_head_list.extend(ripple_sets[user][l][0])
                l_relation_list.extend(ripple_sets[user][l][1])
                l_tail_list.extend(ripple_sets[user][l][2])

            heads_list.append(l_head_list)
            relations_list.append(l_relation_list)
            tails_list.append(l_tail_list)

        return heads_list, relations_list, tails_list

    def get_vector(self, users, heads_list, relations_list, tails_list, items_list):

        o_list = [self.entity_embedding_mat[items_list].reshape(len(users), -1, self.dim).sum(dim=1)]

        for l in range(self.L):
            head_embeddings = self.entity_embedding_mat[heads_list[l]].reshape(len(users), -1, self.dim)
            relation_embeddings = self.relation_embedding_mat[relations_list[l]].reshape(len(users), -1, self.dim)
            tail_embeddings = self.entity_embedding_mat[tails_list[l]].reshape(len(users), -1, self.dim)

            hr = t.cat([head_embeddings, relation_embeddings], dim=-1)  # (batch_size, -1, 2 * dim)
            tr = t.cat([tail_embeddings, relation_embeddings], dim=-1)  # (batch_size, -1, 2 * dim)

            pi = (hr * tr).sum(dim=-1).reshape(len(users), -1, 1)  # (batch_size, -1, 1)
            pi = t.softmax(pi, dim=1)

            ht = t.cat([hr.reshape(1, -1, 2*self.dim), tr.reshape(1, -1, 2*self.dim)], dim=0)
            triple_embeddings = self.rnn(ht)[0][-1].reshape(len(users), -1, self.dim)
            # print(pi.shape, triple_embeddings.shape)
            o_embeddings = (pi * triple_embeddings).sum(dim=1)

            o_list.append(o_embeddings)

        return sum(o_list)


def get_scores(model, rec, ripple_sets):
    scores = {}
    model.eval()
    for user in (rec):
        items = list(rec[user])
        pairs = [[user, item] for item in items]
        predict = []
        for i in range(0, len(pairs), 1024):
            predict.extend(model.forward(pairs[i: i+1024], ripple_sets).cpu().detach().view(-1).numpy().tolist())
        # predict = self.forward(pairs, ripple_sets).cpu().detach().view(-1).numpy().tolist()
        n = len(pairs)
        user_scores = {items[i]: predict[i] for i in range(n)}
        user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        scores[user] = user_list
    model.train()
    return scores


def eval_ctr(model, pairs, ripple_sets, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model(pairs[i: i+batch_size], ripple_sets).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return round(auc, 3), round(acc, 3)


def get_ripple_set(train_dict, kg_dict, H, size):

    ripple_set_dict = {user: [] for user in train_dict}

    for u in train_dict:
        if len(train_dict[u]) >= size:
            indices = np.random.choice(len(train_dict[u]), size, replace=False)
        else:
            indices = np.random.choice(len(train_dict[u]), size, replace=True)
        next_e_list = [train_dict[u][i] for i in indices]
        ripple_set_dict[u].append(next_e_list)
        for h in range(H):
            h_head_list = []
            h_relation_list = []
            h_tail_list = []
            for head in next_e_list:
                if head not in kg_dict:
                    continue
                for rt in kg_dict[head]:
                    relation = rt[0]
                    tail = rt[1]
                    h_head_list.append(head)
                    h_relation_list.append(relation)
                    h_tail_list.append(tail)

            if len(h_head_list) == 0:
                h_head_list = ripple_set_dict[u][-1][0]
                h_relation_list = ripple_set_dict[u][-1][1]
                h_tail_list = ripple_set_dict[u][-1][0]
            else:
                replace = len(h_head_list) < size
                indices = np.random.choice(len(h_head_list), size, replace=replace)
                h_head_list = [h_head_list[i] for i in indices]
                h_relation_list = [h_relation_list[i] for i in indices]
                h_tail_list = [h_tail_list[i] for i in indices]

            ripple_set_dict[u].append((h_head_list, h_relation_list, h_tail_list))

            next_e_list = ripple_set_dict[u][-1][2]

    return ripple_set_dict


def train(args, is_topk=False):
    np.random.seed(555)
    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]
    test_records = get_records(test_set)
    train_records = get_records(train_set)
    ripple_sets = get_ripple_set(train_records, kg_dict, args.L, args.K_l)
    model = NCFG(args.dim, n_entity, n_relation, args.L, n_user, n_item)
    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCELoss()

    print(args.dataset + '-----------------------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('L: %d' % args.L, end='\t')
    print('K_l: %d' % args.K_l, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)

    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    all_precision_list = []
    for epoch in (range(args.epochs)):

        start = time.clock()
        loss_sum = 0
        np.random.shuffle(train_set)
        for i in range(0, len(train_set), args.batch_size):

            batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([int(uvl[2]) for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)

            predicts = model(pairs, ripple_sets)

            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        train_auc, train_acc = eval_ctr(model, train_set, ripple_sets, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, ripple_sets, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, ripple_sets, args.batch_size)

        print('epoch: %d \t train_auc: %.3f \t train_acc: %.3f \t '
              'eval_auc: %.3f \t eval_acc: %.3f \t test_auc: %.3f \t test_acc: %.3f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            scores = get_scores(model, rec, ripple_sets)
            precision_list = get_all_metrics(scores, test_records)[0]
            print(precision_list, end='\t')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        all_precision_list.append(precision_list)
        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.3f \t train_acc: %.3f \t eval_auc: %.3f \t eval_acc: %.3f \t '
          'test_auc: %.3f \t test_acc: %.3f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')

    print(all_precision_list[indices])

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]