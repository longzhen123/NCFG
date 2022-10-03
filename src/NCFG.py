import time
import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from src.evaluate import get_hit, get_ndcg
from src.load_base import load_data, get_records


class NCFG(nn.Module):

    def __init__(self, dim, n_entity, n_relation, L, n_user, n_item):

        super(NCFG, self).__init__()
        self.dim = dim
        entity_embedding_mat = t.randn(n_entity, dim)
        relation_embedding_mat = t.randn(n_relation, dim)

        nn.init.xavier_uniform_(entity_embedding_mat)
        nn.init.xavier_uniform_(relation_embedding_mat)

        self.entity_embedding_mat = nn.Parameter(entity_embedding_mat)
        self.relation_embedding_mat = nn.Parameter(relation_embedding_mat)
        self.L = L
        self.rnn = nn.RNN(2*dim, dim, num_layers=1, nonlinearity='relu')

    def forward(self, pairs, history_dict, ripple_sets):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]

        heads_list, relations_list, tails_list = self.get_head_relation_and_tail(items, ripple_sets)

        user_embeddings = self.get_user_kg_embedding(users, history_dict)
        item_embeddings = self.get_item_kg_embedding(items, heads_list, relations_list, tails_list)

        predict = (user_embeddings * item_embeddings).sum(dim=1)
        return t.sigmoid(predict)

    def get_head_relation_and_tail(self, items, ripple_sets):

        heads_list = []
        relations_list = []
        tails_list = []

        for i in range(self.L):
            l_head_list = []
            l_relation_list = []
            l_tail_list = []

            for item in items:

                l_head_list.extend(ripple_sets[item][i][0])
                l_relation_list.extend(ripple_sets[item][i][1])
                l_tail_list.extend(ripple_sets[item][i][2])

            heads_list.append(l_head_list)
            relations_list.append(l_relation_list)
            tails_list.append(l_tail_list)

        return heads_list, relations_list, tails_list

    def get_user_kg_embedding(self, users, history_dict):
        embedding_list = []

        for user in users:
            embedding_list.append(self.entity_embedding_mat[history_dict[user]].sum(dim=0).reshape(1, self.dim))

        return t.cat(embedding_list, dim=0)

    def get_item_kg_embedding(self, items, heads_list, relations_list, tails_list):

        o_list = [self.entity_embedding_mat[items]]

        for i in range(self.L):
            head_embeddings = self.entity_embedding_mat[heads_list[i]].reshape(len(items), -1, self.dim)
            relation_embeddings = self.relation_embedding_mat[relations_list[i]].reshape(len(items), -1, self.dim)
            tail_embeddings = self.entity_embedding_mat[tails_list[i]].reshape(len(items), -1, self.dim)

            hr = t.cat([head_embeddings, relation_embeddings], dim=-1)  # (batch_size, -1, 2 * dim)
            tr = t.cat([tail_embeddings, relation_embeddings], dim=-1)  # (batch_size, -1, 2 * dim)

            pi = (hr * tr).sum(dim=-1).reshape(len(items), -1, 1)  # (batch_size, -1, 1)
            pi = t.softmax(pi, dim=1)

            ht = t.cat([hr.reshape(1, -1, 2*self.dim), tr.reshape(1, -1, 2*self.dim)], dim=0)
            triple_embeddings = self.rnn(ht)[0][-1].reshape(len(items), -1, self.dim)

            o_embeddings = (pi * triple_embeddings).sum(dim=1)

            o_list.append(o_embeddings)

        return sum(o_list)


def eval_topk(model, rec, history_dict, ripple_sets, topk):
    HR, NDCG = [], []

    model.eval()
    for user in rec:
        items = list(rec[user])
        pairs = [[user, item] for item in items]
        predict = []

        predict.extend(model.forward(pairs, history_dict, ripple_sets).cpu().detach().view(-1).numpy().tolist())
        # predict = self.forward(pairs, ripple_sets).cpu().detach().view(-1).numpy().tolist()
        n = len(pairs)
        item_scores = {items[i]: predict[i] for i in range(n)}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())[: topk]
        HR.append(get_hit(items[-1], item_list))
        NDCG.append(get_ndcg(items[-1], item_list))

    model.train()
    return np.mean(HR), np.mean(NDCG)


def eval_ctr(model, pairs, ripple_sets, history_dict, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model(pairs[i: i+batch_size], history_dict, ripple_sets).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return auc, acc


def get_ripple_set(items, kg_dict, H, size):

    ripple_set_dict = {item: [] for item in items}

    for item in items:

        next_e_list = [item]

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
                h_head_list = ripple_set_dict[item][-1][0]
                h_relation_list = ripple_set_dict[item][-1][1]
                h_tail_list = ripple_set_dict[item][-1][0]
            else:
                replace = len(h_head_list) < size
                indices = np.random.choice(len(h_head_list), size, replace=replace)
                h_head_list = [h_head_list[i] for i in indices]
                h_relation_list = [h_relation_list[i] for i in indices]
                h_tail_list = [h_tail_list[i] for i in indices]

            ripple_set_dict[item].append((h_head_list, h_relation_list, h_tail_list))

            next_e_list = ripple_set_dict[item][-1][2]

    return ripple_set_dict


def get_history(train_records, size):

    history_dict = dict()

    for user, items in train_records.items():
        n = len(items)

        if n >= size:
            indices = np.random.choice(n, size, replace=False)
        else:
            indices = np.random.choice(n, size, replace=True)

        history_dict[user] = [items[i] for i in indices]

    return history_dict


def train(args, is_topk=False):
    np.random.seed(123)
    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]

    train_records = get_records(train_set)
    ripple_sets = get_ripple_set(range(n_item), kg_dict, args.L, args.K_v)
    history_dict = get_history(train_records, args.K_u)
    model = NCFG(args.dim, n_entity, n_relation, args.L, n_user, n_item)
    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCELoss()

    print(args.dataset + '-----------------------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('L: %d' % args.L, end='\t')
    print('K_u: %d' % args.K_u, end='\t')
    print('K_v: %d' % args.K_v, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)

    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    HR_list = []
    NDCG_list = []

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

            predicts = model(pairs, history_dict, ripple_sets)

            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        train_auc, train_acc = eval_ctr(model, train_set, ripple_sets, history_dict, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, ripple_sets, history_dict, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, ripple_sets, history_dict, args.batch_size)

        print('epoch: %d \t train_auc: %.4f \t train_acc: %.4f \t '
              'eval_auc: %.4f \t eval_acc: %.4f \t test_auc: %.4f \t test_acc: %.4f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        HR, NDCG = 0, 0
        if is_topk:
            HR, NDCG = eval_topk(model, rec, history_dict, ripple_sets, args.topk)
            print('HR: %.4f NDCG: %.4f' % (HR, NDCG), end='\t')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        HR_list.append(HR)
        NDCG_list.append(NDCG)

        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.4f \t train_acc: %.4f \t eval_auc: %.4f \t eval_acc: %.4f \t '
          'test_auc: %.4f \t test_acc: %.4f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')

    print('HR: %.4f \t NDCG: %.4f' % (HR_list[indices], NDCG_list[indices]))

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]