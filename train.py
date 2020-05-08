import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import duts_train_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from torch.backends import cudnn
from model import DPNet

cudnn.benchmark = True

torch.manual_seed(2018)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(0)

ckpt_path = './ckpt'

args = {
    'iter_num': 30000,
    'train_batch_size': 10,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'crop_size': 380
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(args['crop_size']),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10),
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

train_set = ImageFolder(duts_train_path, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True, drop_last=True)

criterionBCE = nn.BCELoss().cuda()


def main():
    exp_name = 'dpnet'
    train(exp_name)


def train(exp_name):
    log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
    net = DPNet().cuda().train()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    print 'start to train'

    curr_iter = args['last_iter']
    while True:
        total_loss_record = AvgMeter()
        loss1_record, loss2_record, loss3_record, loss4_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_DPM1_record = AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            predict1, predict2, predict3, predict4, predict_DPM1 = net(inputs)

            loss1 = criterionBCE(predict1, labels)
            loss2 = criterionBCE(predict2, labels)
            loss3 = criterionBCE(predict3, labels)
            loss4 = criterionBCE(predict4, labels)
            loss_DPM1 = criterionBCE(predict_DPM1, labels)

            total_loss = loss1 + loss2 + loss3 + loss4 + loss_DPM1
            total_loss.backward()

            optimizer.step()

            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)
            loss3_record.update(loss3.item(), batch_size)
            loss4_record.update(loss4.item(), batch_size)
            loss_DPM1_record.update(loss_DPM1.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [total loss %.5f], [loss1 %.5f], [loss2 %.5f], [loss3 %.5f], ' \
                  '[loss4 %.5f], [loss_DPM1 %.5f], [lr %.13f]' \
                  % (curr_iter, total_loss_record.avg, loss1_record.avg, loss2_record.avg, loss3_record.avg,
                     loss4_record.avg, loss_DPM1_record.avg, optimizer.param_groups[1]['lr'])

            print log
            open(log_path, 'a').write(log + '\n')

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
