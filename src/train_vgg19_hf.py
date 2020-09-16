from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
from distutils.version import LooseVersion

from wt_utils_new import wt, wt_successive, wt_hf, wt_lf, iwt, create_filters, create_inv_filters

import wandb

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

# Wavelet options
parser.add_argument('--wt-filter-type', type=str, default='bior2.2',
                    help='type of wavelet filter')
parser.add_argument('--num-wt-levels', type=int, default=1,
                    help='number of wavelet transforms applied')
parser.add_argument('--hf', action='store_true', default=False,
                    help='Only train on high frequency patches (IWTed)')
parser.add_argument('--bn', action='store_true', default=False,
                    help='Batch normalization in VGG19 model')
               

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

### Wandb settings
parser.add_argument('--wandb_project_name', type=str)
parser.add_argument('--wandb-entity', type=str)
parser.add_argument('--wandb-api-key', type=str)

### Storage and retrieval
parser.add_argument('--resume-from-epoch', action='store_true', default=False,
                    help='Restore model from path')
parser.add_argument('--model-load-path', type=str,
                    help='Restore model from directory')
parser.add_argument('--save-models', action='store_true', default=True,
                    help='Store checkpoints')
parser.add_argument('--save-path', type=str)

### Validate settings
parser.add_argument('--validate-only', action='store_true', default=False,
                    help='Only validate without training')


args = parser.parse_args()

################# Wandb ##################

wandb_api_key_path = args.wandb_api_key
wandb_path = os.path.join(wandb_api_key_path)
    
with open(wandb_path, 'r') as f:
    key = f.read().split('\n')[0]
os.environ['WANDB_API_KEY'] = key

wandb.init(project=args.wandb_project_name, entity=args.wandb_entity)
wandb.config.update(args)
# wandb_run_id = wandb.run.get_url().split('/')[-1]
############################################

############# DDL settings ####################
args.cuda = not args.no_cuda and torch.cuda.is_available()
allreduce_batch_size = args.batch_size * args.batches_per_allreduce

hvd.init()
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

################# Resume training ##################
results_dir = os.path.join(args.save_path, args.wandb_project_name) #Directory to store runs
try:
    os.makedirs(results_dir)
except OSError:
    pass

# If set > 0, will resume training from a given checkpoint.
if args.resume_from_epoch:
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(os.path.join(results_dir, args.checkpoint_format.format(epoch=try_epoch))):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()
else:
    resume_from_epoch = 0
############################################

print('Resume from epoch: {}'.format(resume_from_epoch))

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0



# # Horovod: write TensorBoard logs on first worker.
# try:
#     if LooseVersion(torch.__version__) >= LooseVersion('1.2.0'):
#         from torch.utils.tensorboard import SummaryWriter
#     else:
#         from tensorboardX import SummaryWriter
#     log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
# except ImportError:
#     log_writer = None

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(4)
 
########################## WT #########################
filters = create_filters('cpu', args.wt_filter_type)
if args.cuda:
    filters = filters.to(device='cuda')
    print('Wavelet filter moved to GPU')

if args.hf:
    print('Running VGG19 model with high frequencies IWTed as input')
    wt_transform = lambda vimg: wt_hf(vimg, filters, levels=args.num_wt_levels)
######################################################

# # Setting up for collecting intermediate "texture" features from pretrained model
# if args.hier_model:
#     features_l4_1 = [[] for i in range(4)]
#     features_l4_2 = [[] for i in range(4)]
#     features_l4_3 = [[] for i in range(4)]

#     def l4_1_2_hook(self, inputs, outputs):
#         global features_l4_1
#         global features_l4_2
#         features_l4_1[hvd.rank()].append(inputs[0])
#         features_l4_2[hvd.rank()].append(outputs)
    
#     def l4_3_hook(self, inputs, outputs):
#         global features_l4_3
#         features_l4_3[hvd.rank()].append(outputs)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(256),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                            #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #                       std=[0.229, 0.224, 0.225])
                         ]))
# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size,
    sampler=train_sampler, **kwargs)

val_dataset = \
    datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.ToTensor(),
                            #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #                       std=[0.229, 0.224, 0.225])
                         ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)


# Set up standard VGG19-BN model.
if args.bn:
    model = models.vgg19_bn()
else:
    model = models.vgg19()
print(model)

# By default, Adasum doesn't need scaling up learning rate.
# For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()

    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = args.batches_per_allreduce * hvd.local_size()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          lr_scaler),
                      momentum=args.momentum, weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
    compression=compression,
    backward_passes_per_step=args.batches_per_allreduce,
    op=hvd.Adasum if args.use_adasum else hvd.Average)

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if args.resume_from_epoch:
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = os.path.join(results_dir, args.checkpoint_format.format(epoch=resume_from_epoch))
        checkpoint = torch.load(filepath)
        print('Loading from checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = wt_transform(data[i:i + args.batch_size])
                if (i+epoch)==0: print('Ran WT without error')
                target_batch = target[i:i + args.batch_size]
                
                output = model(data_batch)

                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'train loss': train_loss.avg.item(),
                           'train accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

    metrics = {'Step' : epoch}
    metrics["Train Loss"] = train_loss.avg.item()
    metrics["Train Acc"] = train_accuracy.avg.item()
    wandb.log(metrics)

    # if log_writer:
    #     log_writer.add_scalar('train/loss', train_loss.avg, epoch)
    #     log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                
                data_tf = wt_transform(data)

                output = model(data_tf)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'val loss': val_loss.avg.item(),
                               'val accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    metrics = {'Val Step' : epoch}
    metrics["Val Loss"] = val_loss.avg.item()
    metrics["Val Acc"] = val_accuracy.avg.item()
    wandb.log(metrics)

    return val_accuracy.avg.item()

# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    elif epoch < 100:
        lr_adj = 1e-3
    else:
        lr_adj = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch, is_best=False):
    if hvd.rank() == 0:
        filepath = os.path.join(results_dir, args.checkpoint_format.format(epoch=epoch + 1))
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)

        if is_best:
            best_filepath = os.path.join(results_dir, args.checkpoint_format.format(epoch='best'))
            torch.save(state, best_filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


for epoch in range(resume_from_epoch, args.epochs):
    best_val_acc = float('-inf')
    is_best = False

    if not args.validate_only:
        train(epoch)
        val_acc = validate(epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            is_best = True

        save_checkpoint(epoch, is_best)
    else:
        validate(epoch)
