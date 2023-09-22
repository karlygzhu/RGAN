import datetime
import os
import time
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from option import opt
from datasets import Train_Vimeo
from torch.utils.data import DataLoader
from build_models import VSR
from torch.autograd import Variable
from test import test

systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
def main():
    torch.manual_seed(opt.seed)

    if opt.train_datasets == 'Vimeo-90K':
        train_data = Train_Vimeo()
    else:
        raise Exception('No training set, please choose a training set')

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.threads, pin_memory=False, drop_last=False)

    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        print('--------------------Exist cuda--------------------')
        use_gpu = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True

    model = VSR()
    criterion = nn.L1Loss()

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    print(model)
    print("Model_add size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) * 4 / 1048576))
    args = sum(p.numel() for p in model.parameters()) / 1000000
    print('args=', args)
    with open('./results/parameter.txt', 'a+') as f:
        f.write('Params = {:.6f}M'.format(args) + '\n')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    start_epoch = 0

    for epochs in range(start_epoch + 1, opt.num_epochs + 1):
        loss_all = []
        with open('./results/lr_results.txt', 'a+') as f:
            f.write('第%d个epoch的学习率：%f' % (epochs, optimizer.param_groups[0]['lr']) + '\n')
        for steps, data in enumerate(train_dataloader):
            start_time = time.time()
            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()

            outputs = model(inputs)
            loss_mse = criterion(labels, outputs)

            loss_all.append(loss_mse.item())

            optimizer.zero_grad()
            loss_mse.backward()

            nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            end_time = time.time()
            cost_time = end_time - start_time
            if steps % 30 == 0:
                print('===> Epochs[{}]({}/{}) || Time = {:.3f}s,'.format(epochs, steps + 1, len(train_dataloader), cost_time),
                      'loss_mse = {:.8f}'.format(loss_mse))

        scheduler.step()
        with open('./results/loss_mse.txt', 'a+') as f:
            f.write('===> Epochs[{}] || Loss_MSE = {:.6f}'
                    .format(epochs, func_sum(loss_all)) + '\n')

        if epochs % 5 == 0:
            save_models(model, epochs)


def save_models(model, epochs):
    save_model_path = os.path.join(opt.save_model_path, systime)
    if not os.path.exists(save_model_path):
        os.makedirs(os.path.join(save_model_path))
    save_name = 'X' + str(opt.scale) + '_epoch_{}.pth'.format(epochs)
    checkpoint = {"net": model.state_dict(), "epoch": epochs}
    torch.save(checkpoint, os.path.join(save_model_path, save_name))
    print('Checkpoints save to {}'.format(save_model_path))

def func_sum(loss):
    outputs = sum(loss)/len(loss)
    return outputs

if __name__ == '__main__':
    main()
