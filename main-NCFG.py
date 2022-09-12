import numpy as np
from src.NCFG import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=30, help='epochs')
    # parser.add_argument("--device", type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=40, help='embedding size')
    # parser.add_argument('--L', type=int, default=1, help='L')
    # parser.add_argument('--K_l', type=int, default=50, help='The size of ripple set')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=30, help='epochs')
    # parser.add_argument("--device", type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=5, help='embedding size')
    # parser.add_argument('--L', type=int, default=1, help='L')
    # parser.add_argument('--K_l', type=int, default=50, help='The size of ripple set')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=10, help='epochs')
    # parser.add_argument("--device", type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=10, help='embedding size')
    # parser.add_argument('--L', type=int, default=3, help='L')
    # parser.add_argument('--K_l', type=int, default=30, help='The size of ripple set')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument("--device", type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=10, help='embedding size')
    parser.add_argument('--L', type=int, default=3, help='L')
    parser.add_argument('--K_l', type=int, default=20, help='The size of ripple set')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()
    train(args, True)

'''
music	train_auc: 0.998 	 train_acc: 0.987 	 eval_auc: 0.838 	 eval_acc: 0.769 	 test_auc: 0.838 	 test_acc: 0.763 		[0.26, 0.37, 0.5, 0.52, 0.52, 0.57, 0.61, 0.64]
book	train_auc: 0.897 	 train_acc: 0.818 	 eval_auc: 0.739 	 eval_acc: 0.674 	 test_auc: 0.740 	 test_acc: 0.676 		[0.08, 0.15, 0.3, 0.35, 0.35, 0.41, 0.44, 0.46]
ml	train_auc: 0.948 	 train_acc: 0.875 	 eval_auc: 0.902 	 eval_acc: 0.823 	 test_auc: 0.905 	 test_acc: 0.823 		[0.22, 0.32, 0.54, 0.58, 0.58, 0.64, 0.66, 0.69]
yelp	train_auc: 0.942 	 train_acc: 0.864 	 eval_auc: 0.861 	 eval_acc: 0.790 	 test_auc: 0.861 	 test_acc: 0.788 		[0.14, 0.24, 0.45, 0.46, 0.46, 0.51, 0.53, 0.56]

'''