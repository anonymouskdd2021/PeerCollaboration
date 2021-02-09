import torch
import torch.nn as nn
import utils
import shutil
import time
import math
import numpy as np
import argparse
import Data_loader
import os
import random
import collections
from SASRec import SASRec


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def INFO_LOG(info):
    print("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def getBatch(data, batch_size):
	start_inx = 0
	end_inx = batch_size

	while end_inx < len(data):
		batch = data[start_inx:end_inx]
		start_inx = end_inx
		end_inx += batch_size
		yield batch

	if end_inx >= len(data):
		batch = data[start_inx:]
		yield batch


parser = argparse.ArgumentParser()
parser.add_argument('--top_k', type=int, default=5,
                    help='Sample from top k predictions')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='hyperpara-Adam')
parser.add_argument('--batch_size', default=128, type=int)
# diginetica_removecold5_seq
# ml20m_removecold5_seq
# qqSearch
parser.add_argument('--datapath', type=str, default='../Data/Session/diginetica_removecold5_seq.csv',
                    help='data path')
parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--savedir', default='checkpoint', type=str)
parser.add_argument('--tt_percentage', type=float, default=0.2,
                    help='0.2 means 80% training 20% testing')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--L2', default=0, type=float)
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden factors, i.e., embedding size.')
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--dropout', default=0, type=float)

# cooperation setting
parser.add_argument('--a', default=1, type=int)
parser.add_argument('--b', default=1, type=int)
parser.add_argument('--num', default=100, type=int)
parser.add_argument('--i', default=1, type=int)
# Increase models diversity
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--cos', action="store_true", default=False)
parser.add_argument('--difflr', action="store_true", default=False)
parser.add_argument('--fixlr', action="store_true", default=False)
parser.add_argument('--percent', default=50, type=float)

parser.add_argument('--shuffle', type=str2bool, default='true')

args = parser.parse_args()
print(args)

dl = Data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath, 'max_len': args.max_len})
train_set, valid_set, test_set = dl.train_set, dl.valid_set, dl.test_set
items_voc = dl.item2id

# Randomly shuffle data
np.random.seed(args.seed)
shuffle_indices = np.random.permutation(np.arange(len(train_set)))
train_set = train_set[shuffle_indices]

print("train_set shape: ", np.shape(train_set))
print("valid_set shape: ", np.shape(valid_set))
print("test_set shape: ", np.shape(test_set))


if args.is_generatesubsession:
    x_train = generatesubsequence(train_set)

model_para = {
    'item_size': len(items_voc)+1,
    'embed_dim': args.hidden,
    'hidden_factor': args.hidden,
    'num_blocks': args.num_blocks,
    'num_heads': args.num_heads,
    'dropout': args.dropout,
    'batch_size': args.batch_size,
    'iterations': 200,
    'max_len': args.max_len,
    'pad': dl.padid,
}
print(model_para)

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SASRec(hidden_size=args.hidden, item_num=model_para['item_size'],
                   max_len=model_para['max_len'], device=args.device, num_blocks=model_para['num_blocks'],
                   num_heads=model_para['num_heads'], dropout_rate=model_para['dropout'],
                   padid=model_para['pad']).to(args.device)
criterion = nn.CrossEntropyLoss()

if args.fixlr:
    loc = 1
else:
    if args.difflr:
        loc = (1 + np.cos(np.pi * ((args.num - args.i) / args.num))) / 2
    else:
        loc = 1
print('The initial learning rate is:', args.lr * loc)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * loc, weight_decay=0)
if args.cos == True:
    steps = train_set.shape[0] // model_para['batch_size']
    if train_set.shape[0] % model_para['batch_size'] == 0:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  args.epochs * steps)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  args.epochs * (steps + 1))
else:
    lr_scheduler = None



best_acc = 0

def cooperation_singleweight(model, epoch):
    while True:
        try:
            checkpoint = torch.load('%s/ckpt%d_%d.t7' % (args.savedir, args.i - 1, epoch))['net']
            break
        except:
            time.sleep(10)
    model_cooperation = collections.OrderedDict()
    print('*'*25, ' start: ', epoch, ' ', '*'*25)
    for i, (key, u) in enumerate(model.state_dict().items()):
        if len(u.size()) != 1:
            threshold = np.percentile(np.array(list(u.cpu().data.abs().numpy().flatten())), args.percent)
            invalid_inds = (u.data.abs() > threshold).float()
            model_cooperation[key] = u * invalid_inds + checkpoint[key] * (1 - invalid_inds)
            continue
        model_cooperation[key] = u

    while(True):
        try:
            model.load_state_dict(model_cooperation)
            break
        except:
            time.sleep(5)
    print('*' * 25, ' end: ', epoch, ' ', '*' * 25)


best_mrr_5, best_mrr_20, best_hit_5, best_hit_20, best_ndcg_5, best_ndcg_20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def test(epoch):
    global best_acc
    global best_mrr_5, best_mrr_20, best_hit_5, best_hit_20, best_ndcg_5, best_ndcg_20

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_size = model_para['batch_size']
    batch_num = test_set.shape[0] / batch_size
    INFO_LOG("-------------------------------------------------------test")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_sam in enumerate(getBatch(test_set, batch_size)):
            inputs, targets = torch.LongTensor(batch_sam[:,:-1]).to(args.device), torch.LongTensor(batch_sam[:,-1]).to(args.device).view([-1])
            outputs = model(inputs, onecall=True) # [batch_size, item_size] only predict the last position

            _, sort_idx_20 = torch.topk(outputs, k=args.top_k + 15, sorted=True)  # [batch_size, 20]
            _, sort_idx_5 = torch.topk(outputs, k=args.top_k, sorted=True)  # [batch_size, 5]
            accuracy(sort_idx_5.data.cpu().numpy(), sort_idx_20.data.cpu().numpy(), targets.data.cpu().numpy(),
                     batch_idx, batch_num, epoch)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        end = time.time()
        INFO_LOG("TIME FOR EPOCH During Testing: {}".format(end - start))
        INFO_LOG("TIME FOR BATCH (mins): {}".format((end - start) / batch_num))
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
    print('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))

    INFO_LOG("epoch: {}\t total_epoch:{}\t total_batches:{}".format(
        epoch, args.epochs, batch_num))
    epoch_mrr_5 = sum(curr_preds_5) / float(len(curr_preds_5))
    epoch_mrr_20 = sum(curr_preds_20) / float(len(curr_preds_20))
    epoch_hit_5 = sum(rec_preds_5) / float(len(rec_preds_5))
    epoch_hit_20 = sum(rec_preds_20) / float(len(rec_preds_20))
    epoch_ndcg_5 = sum(ndcg_preds_5) / float(len(ndcg_preds_5))
    epoch_ndcg_20 = sum(ndcg_preds_20) / float(len(ndcg_preds_20))

    INFO_LOG("Accuracy mrr_5: {}".format(epoch_mrr_5))
    INFO_LOG("Accuracy mrr_20: {}".format(epoch_mrr_20))
    INFO_LOG("Accuracy hit_5: {}".format(epoch_hit_5))
    INFO_LOG("Accuracy hit_20: {}".format(epoch_hit_20))
    INFO_LOG("Accuracy ndcg_5: {}".format(epoch_ndcg_5))
    INFO_LOG("Accuracy ndcg_20: {}".format(epoch_ndcg_20))

    if epoch_mrr_5 > best_mrr_5:
        best_mrr_5 = epoch_mrr_5
    if epoch_mrr_20 > best_mrr_20:
        best_mrr_20 = epoch_mrr_20
    if epoch_hit_5 > best_hit_5:
        best_hit_5 = epoch_hit_5
    if epoch_hit_20 > best_hit_20:
        best_hit_20 = epoch_hit_20
    if epoch_ndcg_5 > best_ndcg_5:
        best_ndcg_5 = epoch_ndcg_5
    if epoch_ndcg_20 > best_ndcg_20:
        best_ndcg_20 = epoch_ndcg_20
    print('Best MRR_5: %.4f Best MRR_20: %.4f '
          'Best Hit_5: %.4f Best Hit_20: %.4f '
          'Best NDCG_5: %.4f Best NDCG_20: %.4f'
          % (best_mrr_5, best_mrr_20, best_hit_5, best_hit_20, best_ndcg_5, best_ndcg_20))


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_size = model_para['batch_size']
    batch_num = train_set.shape[0] / batch_size
    start = time.time()
    INFO_LOG("-------------------------------------------------------train")
    for batch_idx, batch_sam in enumerate(getBatch(train_set, batch_size)):
        inputs, targets = torch.LongTensor(batch_sam[:, :-1]).to(args.device), torch.LongTensor(batch_sam[:, 1:]).to(
            args.device).view([-1])
        optimizer.zero_grad()
        outputs = model(inputs, onecall=False) # [batch_size*seq_len, item_size]
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % max(10, batch_num//10) == 0:
            INFO_LOG("epoch: {}\t {}/{}".format(epoch, batch_idx, batch_num))
            print('Loss: %.3f | Acc(hit@1): %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if args.cos == True:
            lr_scheduler.step()
    end = time.time()
    INFO_LOG("TIME FOR EPOCH During Training: {}".format(end - start))
    INFO_LOG("TIME FOR BATCH (mins): {}".format((end - start) / batch_num))


def accuracy(pred_items_5, pred_items_20, target, batch_idx, batch_num, epoch): # output: [batch_size, 20] target: [batch_size]
    """Computes the accuracy over the k top predictions for the specified values of k"""
    for bi in range(pred_items_5.shape[0]):

        true_item=target[bi]
        predictmap_5={ch : i for i, ch in enumerate(pred_items_5[bi])}
        predictmap_20 = {ch: i for i, ch in enumerate(pred_items_20[bi])}

        rank_5 = predictmap_5.get(true_item)
        rank_20 = predictmap_20.get(true_item)
        if rank_5 == None:
            curr_preds_5.append(0.0)
            rec_preds_5.append(0.0)
            ndcg_preds_5.append(0.0)
        else:
            MRR_5 = 1.0/(rank_5+1)
            Rec_5 = 1.0#3
            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
            curr_preds_5.append(MRR_5)
            rec_preds_5.append(Rec_5)#4
            ndcg_preds_5.append(ndcg_5)  # 4
        if rank_20 == None:
            curr_preds_20.append(0.0)
            rec_preds_20.append(0.0)#2
            ndcg_preds_20.append(0.0)#2
        else:
            MRR_20 = 1.0/(rank_20+1)
            Rec_20 = 1.0#3
            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
            curr_preds_20.append(MRR_20)
            rec_preds_20.append(Rec_20)#4
            ndcg_preds_20.append(ndcg_20)  # 4

    if batch_idx % max(10, batch_num//10) == 0:
        INFO_LOG("epoch/total_epoch: {}/{}\t batch/total_batches: {}/{}".format(
            epoch, args.epochs, batch_idx, batch_num))
        INFO_LOG("Accuracy hit_5: {}".format(sum(rec_preds_5) / float(len(rec_preds_5))))  # 5
        INFO_LOG("Accuracy hit_20: {}".format(sum(rec_preds_20) / float(len(rec_preds_20))))  # 5


if __name__ == '__main__':
    for epoch in range(args.epochs):
        train(epoch)

        curr_preds_5 = []
        rec_preds_5 = []
        ndcg_preds_5 = []
        curr_preds_20 = []
        rec_preds_20 = []
        ndcg_preds_20 = []
        test(epoch)
        state = {
            'net': model.state_dict(),
            'opt': optimizer.state_dict(),
        }
        if args.cos == True:
            state['sdl'] = lr_scheduler.state_dict()

        while (True):
            try:
                torch.save(state, '%s/ckpt%d_%d.t7' % (args.savedir, args.i % args.num, epoch))
                break
            except:
                time.sleep(5)

        cooperation_singleweight(model, epoch)
