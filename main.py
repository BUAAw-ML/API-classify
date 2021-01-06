import argparse
from engine import *
from models import *
from util import *
from dataLoader import *
from transformers import BertModel

from keras_utils.dataset_convert import convert
from keras_model_train import train_keras_model

import datetime

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Training Super-parameters')

parser.add_argument('-seed', default=0, type=int, metavar='N',
                    help='random seed')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[40], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--G-lr', '--Generator-learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--D-lr', '--Discriminator-learning-rate', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--B-lr', '--Bert-learning-rate', default=0.001, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=400, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_model_path', default='./checkpoint', type=str,
                    help='path to save checkpoint (default: none)')
# parser.add_argument('--log_dir', default='./logs', type=str,
#                     help='path to save log (default: none)')
parser.add_argument('--data_type', default='All', type=str,
                    help='The type of data')
parser.add_argument('--data_path', default='../datasets/AAPD/aapd2.csv', type=str,
                    help='path of data')
parser.add_argument('--bert_trainable', default=True, type=bool,
                    help='bert_trainable')
# parser.add_argument('--utilize_unlabeled_data', default=1, type=int,
#                     help='utilize_unlabeled_data')
parser.add_argument('--use_previousData', default=0, type=int,
                    help='use_previousData')
parser.add_argument('--method', default='MultiLabelMAP', type=str,
                    help='Method')
parser.add_argument('--overlength_handle', default='truncation', type=str,
                    help='overlength_handle')
parser.add_argument('--min_tagFrequence', default=0, type=int,
                    help='min_tagFrequence')
parser.add_argument('--max_tagFrequence', default=100000, type=int,
                    help='max_tagFrequence')
parser.add_argument('--intanceNum_limit', default=2000, type=int,
                    help='max_tagFrequence')
parser.add_argument('--data_split', default=0.5, type=float,
                    help='data_split')
parser.add_argument('--experiment_no', default='01', type=str,
                    help='experiment_no')
parser.add_argument('--test_description', default='01', type=str,
                    help='test_description')

parser.add_argument("--model_backend", default="pytorch", type=str,
                    help="model backend (pytorch | keras)")
parser.add_argument("--pretrained_word_emb_path", default="glove.6B.200d.txt", type=str,
                    help="pretrained word embedding path")
parser.add_argument("--keras_model_type", default="ServerNet", type=str,
                    help="Keras model type")

global args, use_gpu
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

result_path = os.path.join('result', datetime.date.today().strftime('%Y%m%d'))
log_dir = os.path.join(result_path, 'logs')
if not os.path.exists(result_path):
    os.makedirs(result_path)
method_str = args.experiment_no + '_' + args.data_path.split("/")[-1]
fo = open(os.path.join(result_path, method_str + '.txt'), "a+")
print('#' * 100 + '\n')
fo.write('#' * 100 + '\n')
setting_str = 'Setting: \t batch-size: {} \t epoch_step: {} \t G_LR: {} \t D_LR: {} \t B_LR: {}'\
              '\ndevice_ids: {} \t data_path: {} \t bert_trainable: {}' \
              '\nuse_previousData: {} \t method: {} \t overlength_handle: {} \t data_split: {} \n' \
              'experiment_no: {} \t test_description: {} \n'.format(
                args.batch_size, args.epoch_step, args.G_lr, args.D_lr, args.B_lr,
                args.device_ids, args.data_path, args.bert_trainable,
                args.use_previousData, args.method, args.overlength_handle, args.data_split,
                args.experiment_no, args.test_description)

print(setting_str)
fo.write(setting_str)

data_config = {'overlength_handle': args.overlength_handle, 'intanceNum_limit': args.intanceNum_limit,
               'min_tagFrequence': args.min_tagFrequence, 'max_tagFrequence': args.max_tagFrequence,
               'data_split': args.data_split}

dataset, encoded_tag, tag_mask = load_data(data_config=data_config,
                                           data_path=args.data_path,
                                           data_type=args.data_type,
                                           use_previousData=args.use_previousData)

data_size = "train_data_size: {} \nunlabeled_train_data: {} \nval_data_size: {} \ntag_num: {} \n".format(
    len(dataset.train_data), len(dataset.unlabeled_train_data), len(dataset.test_data), len(dataset.tag2id))
print(data_size)
fo.write(data_size)

if args.model_backend == "pytorch":
    bert = BertModel.from_pretrained('bert-base-uncased')

    model = {}
    model['Discriminator'] = Discriminator(num_classes=len(dataset.tag2id))
    model['Encoder'] = Bert_Encoder(bert, bert_trainable=args.bert_trainable)
    model['Generator'] = Generator(bert, num_classes=len(dataset.tag2id))
    model['MABert'] = MABert(bert, num_classes=len(dataset.tag2id), bert_trainable=args.bert_trainable, device=args.device_ids[0])

    # define loss function (criterion)
    criterion = nn.BCELoss() #nn.MultiLabelSoftMarginLoss()#


    # define optimizer
    optimizer = {}
    optimizer['Generator'] = torch.optim.SGD([{'params': model['Generator'].parameters(), 'lr': args.G_lr}],
                                            momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer['enc'] = torch.optim.SGD(model['MABert'].get_config_optim(args.D_lr, args.B_lr),
                                momentum=args.momentum, weight_decay=args.weight_decay)

    # optimizer['enc'] = torch.optim.Adam(model['MABert'].get_config_optim(args.D_lr, args.B_lr))

    state = {'batch_size': args.batch_size, 'max_epochs': args.epochs, 'evaluate': args.evaluate,
            'resume': args.resume, 'num_classes': dataset.get_tags_num(), 'difficult_examples': False,
            'save_model_path': args.save_model_path, 'log_dir': log_dir, 'workers': args.workers,
            'epoch_step': args.epoch_step, 'lr': args.D_lr, 'encoded_tag': encoded_tag, 'tag_mask': tag_mask,
            'device_ids': args.device_ids, 'print_freq': args.print_freq, 'id2tag': dataset.id2tag,
            'result_file': fo, 'method': args.method}

    if args.evaluate:
        state['evaluate'] = True

    if args.method == 'MultiLabelMAP':
        engine = MultiLabelMAPEngine(state)
    elif args.method == 'semiGAN_MultiLabelMAP':
        engine = semiGAN_MultiLabelMAPEngine(state)

    engine.learning(model, criterion, dataset, optimizer)

    fo.close()

elif args.model_backend == "keras":
    X, Y, test_X, test_Y = convert(dataset, args.pretrained_word_emb_path)
    
    train_keras_model(args.keras_model_type, data={"train": (X, Y), "test": (test_X, test_Y)}, glove_path=args.pretrained_word_emb_path, save_model_path=args.save_model_path)