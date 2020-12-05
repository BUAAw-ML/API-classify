import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet as tnt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sampler import MultilabelBalancedRandomSampler

import tensorflow as tf

from util import *
import json

tqdm.monitor_interval = 0

class Engine(object):
    def __init__(self, state={}):
        self.writer = SummaryWriter(state['log_dir'])
        os.makedirs(state['log_dir'], exist_ok=True)
        self.result_file = state['result_file']

        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = False
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0
        # best score
        self.state['best_score'] = 0.
        self.state['train_iters'] = 0
        self.state['eval_iters'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        input_var = self.state['input']
        target_var = self.state['target']

        # compute output
        self.state['output'] = model(input_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            self.state['train_iters'] += 1
        else:
            self.state['eval_iters'] += 1

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def learning(self, model, criterion, dataset, optimizer=None):
        # data loading code

        # train_sampler = MultilabelBalancedRandomSampler(dataset.train_data)

        train_loader = torch.utils.data.DataLoader(dataset.train_data,
                                                   # sampler=train_sampler,
                                                   batch_size=self.state['batch_size'], shuffle=False,
                                                   num_workers=self.state['workers'],
                                                   collate_fn=dataset.collate_fn)

        unlabeled_train_loader = torch.utils.data.DataLoader(dataset.unlabeled_train_data,
                                                   # sampler=train_sampler,
                                                   batch_size=self.state['batch_size'], shuffle=False,
                                                   num_workers=self.state['workers'],
                                                   collate_fn=dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(dataset.test_data,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'], collate_fn=dataset.collate_fn)

        if self.state['use_gpu']:
            # train_loader.pin_memory = True
            # val_loader.pin_memory = True
            # cudnn.benchmark = True

            model['Discriminator'] = model['Discriminator'].cuda(self.state['device_ids'][0])
            model['Generator'] = model['Generator'].cuda(self.state['device_ids'][0])
            model['Encoder'] = model['Encoder'].cuda(self.state['device_ids'][0])
            model['MABert'] = model['MABert'].cuda(self.state['device_ids'][0])


            # model = torch.nn.DataParallel(model, device_ids=self.state['device_ids'])
            if 'encoded_tag' in self.state:
                self.state['encoded_tag'] = self.state['encoded_tag'].cuda(self.state['device_ids'][0])
            if 'tag_mask' in self.state:
                self.state['tag_mask'] = self.state['tag_mask'].cuda(self.state['device_ids'][0])
            criterion = criterion.cuda(self.state['device_ids'][0])

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:', lr)

            if self.state['method'] == 'semiGAN_MultiLabelMAP':
                # train for one epoch
                print("Train with unlabeled data:")
                self.train(unlabeled_train_loader, model, criterion, optimizer, epoch, True)

            # train for one epoch
            print("Train with labeled data:")
            self.train(train_loader, model, criterion, optimizer, epoch, False)

            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])

            best_str = ' *** best={best:.3f}'.format(best=self.state['best_score'])
            print(best_str)
            self.result_file.write(best_str + '\n')

        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch, semi_supervised):

        # switch to train mode
        model['Discriminator'].train()
        model['Generator'].train()
        model['Encoder'].train()
        model['MABert'].train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target, _) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(self.state['device_ids'][0])

            self.on_forward(True, model, criterion, data_loader, optimizer, semi_supervised=semi_supervised)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)
    
    @torch.no_grad()
    def validate(self, data_loader, model, criterion, epoch):
        # switch to evaluate mode

        model['Discriminator'].eval()
        model['Generator'].eval()
        model['Encoder'].eval()
        model['MABert'].eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (input, target, self.state['dscp']) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(self.state['device_ids'][0])

            output = self.on_forward(False, model, criterion, data_loader)

            #record the detials of the result:
            # if epoch == self.state['max_epochs'] - 1:
            #     self.recordResult(target, output)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)

        return score

    def recordResult(self, target, output):
        result = []
        for i in range(len(target)):
            buf = [self.state['dscp'][i],
                   [self.state['id2tag'][index] for (index, value) in enumerate(target[i]) if value == 1],
                   [self.state['id2tag'][index] for index in
                    sorted(range(len(output[i])), key=lambda k: output[i][k], reverse=True)[:10]]]
            if buf[2][0] not in buf[1]:
                result.append(buf)

        with open('testResult.json', 'a') as f:
            json.dump(result, f)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'],
                                             'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for opt in optimizer.values():
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * decay
                lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True, semi_supervised=False):
        target_var = self.state['target']
        ids, token_type_ids, attention_mask = self.state['input']
        ids = ids.cuda(self.state['device_ids'][0])
        token_type_ids = token_type_ids.cuda(self.state['device_ids'][0])
        attention_mask = attention_mask.cuda(self.state['device_ids'][0])

        if training:
            self.state['train_iters'] += 1
        else:
            self.state['eval_iters'] += 1

        z = torch.rand(ids.shape[0], 768).type(torch.FloatTensor).cuda(self.state['device_ids'][0])
        x_g = model['Generator'](z)

        _, logits, prob = model['MABert'](ids, token_type_ids, attention_mask,
                                                                      self.state['encoded_tag'],
                                                                      self.state['tag_mask'], x_g.detach())#
        logits = logits[:, 1:]
        self.state['output'] = F.softmax(logits, dim=-1)

        log_probs = F.log_softmax(logits, dim=-1)
        per_example_loss = -1 * torch.sum(target_var * log_probs, dim=-1) / target_var.shape[-1]
        D_L_Supervised = torch.mean(per_example_loss)
        self.state['loss'] = D_L_Supervised

        if training:
            optimizer['enc'].zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm_(optimizer['enc'].param_groups[0]["params"], max_norm=10.0)
            optimizer['enc'].step()
        else:
            return self.state['output']

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                reselt_str = 'Epoch: [{0}]\t Loss {loss:.4f}\t mAP {map:.3f} \n ' \
                'OP: {OP:.4f}\t OR: {OR:.4f}\t OF1: {OF1:.4f}\t CP: {CP:.4f}\t CR: {CR:.4f}\t CF1: {CF1:.4f}'.format(
                self.state['epoch'], loss=loss, map=map, OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1)

            else:
                reselt_str = 'Test: \t Loss {loss:.4f}\t mAP {map:.3f} \n' \
                'OP: {OP:.4f}\t OR: {OR:.4f}\t OF1: {OF1:.4f}\t CP: {CP:.4f}\t CR: {CR:.4f}\t CF1: {CF1:.4f} \n' \
                'OP_3: {OP_3:.4f}\t OR_3: {OR_3:.4f}\t OF1_3: {OF1_3:.4f}\t CP_3: {CP_3:.4f}\t CR_3: {CR_3:.4f}\t CF1_3: {CF1_3:.4f}'.format(
                    loss=loss, map=map, OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1,
                    OP_3=OP_k, OR_3=OR_k, OF1_3=OF1_k, CP_3=CP_k, CR_3=CR_k, CF1_3=CF1_k)

            print(reselt_str)
            self.result_file.write(reselt_str + '\n')

        if training:
            self.writer.add_scalar('loss/train_epoch_loss', loss, self.state['epoch'])
            self.writer.add_scalar('mAP/train_mAP', map, self.state['epoch'])
            self.writer.add_scalar('OF1/train_OF1', OF1, self.state['epoch'])
            self.writer.add_scalar('CF1/train_CF1', CF1, self.state['epoch'])
        else:
            self.writer.add_scalar('loss/eval_epoch_loss', loss, self.state['epoch'])
            self.writer.add_scalar('mAP/eval_mAP', map, self.state['epoch'])
            self.writer.add_scalar('OF1/eval_OF1', OF1, self.state['epoch'])
            self.writer.add_scalar('CF1/eval_CF1', CF1, self.state['epoch'])
        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])
        if training:
            self.writer.add_scalar('loss/train_batch_loss', self.state['loss_batch'], self.state['train_iters'] - 1)
        else:
            self.writer.add_scalar('loss/eval_batch_loss', self.state['loss_batch'], self.state['eval_iters'] - 1)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data.cpu(), self.state['target_gt'].cpu())

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))


class semiGAN_MultiLabelMAPEngine(MultiLabelMAPEngine):

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True, semi_supervised=False):
        target_var = self.state['target']
        ids, token_type_ids, attention_mask = self.state['input']
        ids = ids.cuda(self.state['device_ids'][0])
        token_type_ids = token_type_ids.cuda(self.state['device_ids'][0])
        attention_mask = attention_mask.cuda(self.state['device_ids'][0])

        if training:
            self.state['train_iters'] += 1
        else:
            self.state['eval_iters'] += 1

        epsilon = 1e-8

        z = torch.rand(ids.shape[0], 768).type(torch.FloatTensor).cuda(self.state['device_ids'][0])
        x_g = model['Generator'](z)

        #-----------train enc-----------
        _, logits, prob = model['MABert'](ids, token_type_ids, attention_mask,
                                                                      self.state['encoded_tag'],
                                                                      self.state['tag_mask'], x_g.detach())#

        print(prob[:, 0])

        logits = logits[:, 1:]
        self.state['output'] = F.softmax(logits, dim=-1)

        # prob = prob[:, 0]
        # epsion2 = torch.zeros((4, 1)).cuda(self.state['device_ids'][0])
        # epsion2[prob == 1] = 1e-8
        # D_L_unsupervised = -1 * torch.mean(torch.log(1 - prob + epsion2))
        D_L_unsupervised = -1 * torch.mean(torch.log(prob[:, 0] + epsilon))

        if semi_supervised == False: #train with labeled data
            log_probs = F.log_softmax(logits, dim=-1)
            per_example_loss = -1 * torch.sum(target_var * log_probs, dim=-1) / target_var.shape[-1]
            D_L_Supervised = torch.mean(per_example_loss)
            d_loss = D_L_Supervised + D_L_unsupervised
        else:
            # pseudo_label = torch.max(self.state['output'], -1, keepdim=True)[0]
            # pseudo_label = self.state['output'] - pseudo_label
            #
            # pseudo_label[pseudo_label >= 0] = 1.
            # pseudo_label[pseudo_label < 0] = 0.
            #
            # log_probs = F.log_softmax(logits, dim=-1)
            # per_example_loss = -1 * torch.sum(pseudo_label * log_probs, dim=-1) / pseudo_label.shape[-1]
            # D_L_Supervised = torch.mean(per_example_loss)

            d_loss = D_L_unsupervised

        if training:
            optimizer['enc'].zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(optimizer['enc'].param_groups[0]["params"], max_norm=10.0)
            optimizer['enc'].step()

        #-----------train Generator-----------

        features, _, prob = model['MABert'](ids, token_type_ids, attention_mask,
                                                                      self.state['encoded_tag'],
                                                                      self.state['tag_mask'], x_g)
        # prob = prob[:, 0]
        # epsion = torch.zeros((4, 1)).cuda(self.state['device_ids'][0])
        # epsion[prob == 0] = 1e-8
        # g_loss = -1 * torch.mean(torch.log(prob + epsion))
        g_loss = -1 * torch.mean(torch.log(1 - prob[:, 0] + epsilon))
        # feature_error = torch.mean(torch.mean(features.detach(), dim=0) - torch.mean(x_g[:,:features.shape[1],:], dim=0), dim=0)
        feature_error = torch.mean(torch.mean(features.detach(), dim=0) - torch.mean(x_g, dim=0), dim=0)
        G_feat_match = torch.mean(feature_error * feature_error)
        g_loss = g_loss #+G_feat_match#

        if training:
            optimizer['Generator'].zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(model['Generator'].parameters(), max_norm=10.0)
            optimizer['Generator'].step()

        self.state['loss'] = [d_loss, g_loss]

        if not training:
            return self.state['output']

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'][0].item() + self.state['loss'][1].item()
        self.state['meter_loss'].add(self.state['loss_batch'])
        if training:
            self.writer.add_scalar('loss/train_batch_loss', self.state['loss_batch'], self.state['train_iters'] - 1)
        else:
            self.writer.add_scalar('loss/eval_batch_loss', self.state['loss_batch'], self.state['eval_iters'] - 1)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data.cpu(), self.state['target_gt'].cpu())

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'd_Loss {d_loss_current:.4f}\t'
                      'g_Loss {g_loss_current:.4f}'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time,
                    d_loss_current=self.state['loss'][0].item(),
                    g_loss_current=self.state['loss'][1].item()
                ))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'd_Loss {d_loss_current:.4f}\t'
                      'g_Loss {g_loss_current:.4f}'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time,
                    d_loss_current=self.state['loss'][0].item(),
                    g_loss_current=self.state['loss'][1].item()
                ))




