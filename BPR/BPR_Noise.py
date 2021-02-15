import torch
import torch.nn as nn
from bpr import BPR
import utils
import shutil
import time
import math
import numpy as np
import argparse
from Data_loader import Data_loader
import os
import random
import collections


def INFO_LOG(info):
    print("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



def negativesample(batch):
    sample = []
    for pair in batch:
        u = pair[0]
        i = pair[1]
        j = np.random.randint(item_size)
        while j in train_seq[u]:
            j = np.random.randint(item_size)
        sample.append([u, i, j])
    return sample


def getBatch(data, batch_size):
    start_inx = 0
    end_inx = batch_size

    while end_inx < len(data):
        batch = data[start_inx:end_inx]
        start_inx = end_inx
        end_inx += batch_size
        sample = negativesample(batch)
        yield np.array(sample)

    if end_inx >= len(data):
        batch = data[start_inx:]
        sample = negativesample(batch)
        yield np.array(sample)


parser = argparse.ArgumentParser()
parser.add_argument('--top_k', type=int, default=5,
                    help='Sample from top k predictions')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='hyperpara-Adam')
parser.add_argument('--batch_size', default=2048, type=int)
parser.add_argument('--datapath', type=str, default='ml20m_removecold5_seq.csv',
                    help='data path')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--savedir', default='checkpoint', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--layer', type=int, default=0)
parser.add_argument('--L2', type=float, default=0)
parser.add_argument('--reg_type', type=str, default='part')

parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--percent', default=50, type=float)
parser.add_argument('--T', default=50, type=int)
parser.add_argument('--rate', default=0.02, type=float)

parser.add_argument('--shuffle', type=str2bool, default='true')
args = parser.parse_args()
print(args)

# Load preprocess data
dl = Data_loader({'model_type': 'generator', 'dir_name': args.datapath})
train_set, valid_set, test_set = dl.train_set, dl.valid_set, dl.test_set
train_seq = dl.train_seq
items_voc = dl.item2id
user_size = dl.user_size
item_size = dl.item_size
print('Load complete')

# Randomly shuffle data
np.random.seed(args.seed)
shuffle_indices = np.random.permutation(np.arange(len(train_set)))
train_set = train_set[shuffle_indices]

print("train_set shape: ", np.shape(train_set))
print("valid_set shape: ", np.shape(valid_set))
print("test_set shape: ", np.shape(test_set))

model_para = {
    'user_size': user_size,
    'item_size': item_size,
    'embed_dim': 256,
    'layer': args.layer,
    'batch_size':args.batch_size,
    'iterations':200,
}
print(model_para)

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BPR(model_para["user_size"], model_para["item_size"], model_para["embed_dim"], args.L2,
            model_para["layer"], args.reg_type).to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_acc = 0


def cooperation_noise(model, epoch):
    model_cooperation = collections.OrderedDict()
    print('*' * 25, ' start: ', epoch, ' ', '*' * 25)
    for i, (key, u) in enumerate(model.state_dict().items()):
        if 'weight' in key:
            threshold = np.percentile(np.array(list(u.cpu().data.abs().numpy().flatten())), args.percent)
            invalid_inds = (u.data.abs() < threshold).float()
            noise_information = torch.Tensor(u.size()).normal_(0, math.pow(args.rate, epoch // args.T + 1)).cuda()
            model_cooperation[key] = u + noise_information * invalid_inds
    model.load_state_dict(model_cooperation)
    while(True):
        try:
            model.load_state_dict(model_cooperation)
            break
        except:
            time.sleep(5)
    print('*' * 25, ' end: ', epoch, ' ', '*' * 25)


best_mrr_5, best_mrr_20, best_hit_5, best_hit_20, best_ndcg_5, best_ndcg_20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def pad_train_seq(data):
    max_len = max([len(s) for s in data])
    pad_samples = []
    for idx, s in enumerate(data):
        if len(s) < 1:
            print(s, idx)
        pad_samples.append([s[0]]*(max_len-len(s))+s)
    return np.array(pad_samples)

mask_train_seq = pad_train_seq(train_seq)

def test(epoch):
    global best_mrr_5, best_mrr_20, best_hit_5, best_hit_20, best_ndcg_5, best_ndcg_20

    model.eval()
    correct = 0
    total = 0
    batch_size = model_para['batch_size']
    batch_num = test_set.shape[0] / batch_size
    INFO_LOG("-------------------------------------------------------test")

    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_sam in enumerate(getBatch(test_set, batch_size)):
            u, targets = torch.LongTensor(batch_sam[:, 0]).to(args.device), \
                   torch.LongTensor(batch_sam[:, 1]).to(args.device)
            outputs = model.predict(u) # [batch_size, item_size] only predict the last position
            mask = torch.ones([min([batch_size, len(batch_sam)]), item_size]).cuda()
            mask.scatter_(dim=1, index=torch.LongTensor(mask_train_seq[batch_sam[:, 0]]).cuda(),
                          value=torch.tensor(0).cuda())

            # Calculate prediction value
            outputs = torch.sigmoid(outputs)
            outputs = torch.mul(mask, outputs)

            _, sort_idx_20 = torch.topk(outputs, k=args.top_k+15, sorted=True)  # [batch_size, 20]
            _, sort_idx_5 = torch.topk(outputs, k=args.top_k, sorted=True)  # [batch_size, 5]

            accuracy(sort_idx_5.data.cpu().numpy(), sort_idx_20.data.cpu().numpy(), targets.data.cpu().numpy(), batch_idx, batch_num, epoch)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        end = time.time()
        print('Acc(hit@1): %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
        INFO_LOG("TIME FOR EPOCH During Testing: {}".format(end - start))
        INFO_LOG("TIME FOR BATCH (mins): {}".format((end - start) / batch_num))
    # acc = 100. * correct / total
    # if acc > best_acc:
    #     best_acc = acc
    #     state = {
    #         'net': model.state_dict(),
    #         'acc(hit@1)': acc
    #     }
    #     torch.save(state, '%s/best_model.t7' % (args.savedir))
    # print('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))

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
        u, i, j = torch.LongTensor(batch_sam[:, 0]).to(args.device), \
                  torch.LongTensor(batch_sam[:, 1]).to(args.device), \
                  torch.LongTensor(batch_sam[:, 2]).to(args.device)

        optimizer.zero_grad()
        loss = model(u, i, j)  # [batch_size]
        if args.reg_type == 'all':
            L2_loss = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    L2_loss += torch.norm(param, 2)
            loss += args.L2 * L2_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % max(10, batch_num // 10) == 0:
            INFO_LOG("epoch: {}\t {}/{} \t train_loss: {}".format(epoch, batch_idx, batch_num,
                                                                  train_loss / (batch_idx + 1) * batch_size))
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
    for i, (key, u) in enumerate(model.state_dict().items()):
        print(key, u.size())
    for epoch in range(args.epochs):
        train(epoch)
        curr_preds_5 = []
        rec_preds_5 = []
        ndcg_preds_5 = []
        curr_preds_20 = []
        rec_preds_20 = []
        ndcg_preds_20 = []
        # if epoch > 15:
        test(epoch)
        state = {
            'net': model.state_dict(),
            'opt': optimizer.state_dict(),
        }
        while(True):
            try:
                torch.save(state, '%s/ckpt.t7' % (args.savedir))
                break
            except:
                time.sleep(5)
        del state
        cooperation_noise(model, epoch)


