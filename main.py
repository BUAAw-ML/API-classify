import argparse
from engine import *
from models import *
from util import *
from dataLoader import *
from transformers.optimization import AdamW

parser = argparse.ArgumentParser(description='Training Super-parameters')
#
# parser.add_argument('data_path', default='data/ProgrammerWeb/', type=str,
#                     help='path to dataset (e.g. data/')

parser.add_argument('-seed', default=0, type=int, metavar='N',
                    help='random seed')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[13], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=6, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.01, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_model_path', default='./checkpoint', type=str,
                    help='path to save checkpoint (default: none)')
parser.add_argument('--log_dir', default='./logs', type=str,
                    help='path to save log (default: none)')


def multiLabel_text_classify():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    print("device_id: {}".format(args.device_ids))

    use_gpu = torch.cuda.is_available()
    train_dataset, val_dataset, encoded_tag, tag_mask, tag_embedding_file, tag_weight = \
        load_dataset('../datasets/ProgrammerWeb/programweb-data.csv') #Mashups

    model = gcn_bert(num_classes=len(train_dataset.tag2id), t=0.4, co_occur_mat=train_dataset.co_occur_mat)

    # define loss function (criterion)
    # criterion = nn.BCELoss()
    criterion = nn.MultiLabelSoftMarginLoss() #weight=torch.from_numpy(np.array(tag_weight)).float().cuda(0)

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = AdamW(model.get_config_optim(args.lr, args.lrp),
    #                      lr=args.lr, correct_bias=False)

    state = {'batch_size': args.batch_size, 'max_epochs': args.epochs, 'evaluate': args.evaluate, 'resume': args.resume,
             'num_classes': train_dataset.get_tags_num(), 'difficult_examples': False,
             'save_model_path': args.save_model_path, 'log_dir': args.log_dir, 'workers': args.workers,
             'epoch_step': args.epoch_step, 'lr': args.lr, 'encoded_tag': encoded_tag, 'tag_mask': tag_mask,
             'device_ids': args.device_ids, 'print_freq': args.print_freq, 'id2tag': train_dataset.id2tag,
             'tfidf_result': train_dataset.tfidf_dict, 'tag_embedding_file': tag_embedding_file}

    if args.evaluate:
        state['evaluate'] = True

    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    multiLabel_text_classify()
