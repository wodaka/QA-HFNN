import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from data.scene15.data_loader import load_dataset_scene15
from data.dirtyMnist.data_loader import load_dataset_dmnist
from data.cifar10.data_reader import load_dataset_cifar10
from utils.evaluation import cal_ms,cal_merics
from models.QAHFNN import MyNetwork_dmnist

import torch.optim as optim

import datetime
import time

import logging
import sys


def init_logger(filename, logger_name):
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')

    logging.basicConfig(
        level=logging.INFO,
        # format='[%(asctime)s] %(name)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        format='%(message)s',
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Test
    logger = logging.getLogger(logger_name)
    logger.info('### Init. Logger {} ###'.format(logger_name))
    return logger

# Initialize
my_logger = init_logger("./run_result.log", "ml_logger")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def trainModel(config):

    checkpoints_path = config.ckpt_path
    # 判断文件夹是否存在
    if not os.path.exists(checkpoints_path):
        # 如果不存在则创建文件夹
        os.mkdir(checkpoints_path)
        print(f"{checkpoints_path} is created")

    batchsize = config.batch_size
    epochs = config.epochs
    LR = config.lr  # 初始学习率

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # data_train, data_test = load_dataset_scene15()
    # data_train, data_test = load_dataset_cifar10()
    data_train, data_test = load_dataset_dmnist()

    train_data_size = len(data_train)
    valid_data_size = len(data_test)
    print('train_size: {:4d}  valid_size:{:4d}'.format(train_data_size, valid_data_size))

    train_loader = DataLoader(
        data_train,
        batch_size=batchsize,
        shuffle=True,
        # pin_memory=True,
    )
    test_loader = DataLoader(
        data_test,
        batch_size=batchsize,
        shuffle=False,
        # pin_memory=True,
    )

    history = []
    best_acc = 0.0
    best_epoch = 0

    num_classes = 10
    input_feature_num = 3*32*32
    classical_output_feature_num = 32

    # print(f'The model has {count_parameters(model):,} trainable parameters')
    best_acc_list = []
    best_prec_list = []
    best_rec_list = []
    best_f1_list = []

    #跑几次，比如10次或者15次，计算均值和方差
    for ti in range(2):
        ti_start = time.time()

        model = MyNetwork_dmnist().to(device)

        # model = MyNetwork_scene15().to(device)
        # model = MyNetwork_cifar10n().to(device)

        criterion = nn.CrossEntropyLoss()


        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-3)
        # 学习率调整策略 MultiStep：
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                   milestones=[int(epochs * 0.56), int(epochs * 0.78)],
                                                   # milestones=[int(epochs * 1)],
                                                   gamma=0.1, last_epoch=-1)

        best_acc = 0
        precision_t = []
        recall_t = []
        f1_t = []
        for epoch in range(epochs):
            epoch_start = time.time()

            # print("Epoch: {}/{}".format(epoch + 1, epochs))
            my_logger.info("Epoch: {}/{}".format(epoch + 1, epochs))
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0

            for i,(inputs, labels) in enumerate(train_loader):

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                train_acc += acc.item() * inputs.size(0)

            with torch.no_grad():
                model.eval()

                y_true=[]
                y_pred=[]
                for j, (inputs, labels) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    valid_loss += loss.item() * inputs.size(0)

                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    valid_acc += acc.item() * inputs.size(0)

                    y_true.extend(labels.cpu().numpy().tolist())
                    y_pred.extend(predictions.cpu().numpy().tolist())

            scheduler.step()
            # print('\t last_lr:', scheduler.get_last_lr())
            my_logger.info('\t last_lr: ' + str(scheduler.get_last_lr()[0]))

            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / train_data_size

            avg_valid_loss = valid_loss / valid_data_size
            avg_valid_acc = valid_acc / valid_data_size

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            if best_acc < avg_valid_acc:
                best_acc = avg_valid_acc
                best_epoch = epoch + 1

            epoch_end = time.time()

            precision, recall, f1 = cal_merics(y_true,y_pred)
            precision_t.append(precision)
            recall_t.append(recall)
            f1_t.append(f1)
            # print(
            #     "\t Training: Loss: {:.4f}, Accuracy: {:.4f}%, "
            #     "\n\t Validation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.3f}s".format(
            #         avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
            #                         epoch_end - epoch_start
            #     ))
            # print("\t Precision : {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision,recall,f1))
            # print("\t Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
            # # wandb.log({'epoch': epoch, 'Training_Loss': avg_train_loss,'Validation_Loss': avg_valid_loss,
            # #            'train_acc': avg_train_acc,'val_acc': avg_valid_acc, 'best_val_acc': best_acc})

            my_logger.info(
                "\t Training: Loss: {:.4f}, Accuracy: {:.4f}%, "
                "\n\t Validation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.3f}s".format(
                    avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                                    epoch_end - epoch_start
                ))
            my_logger.info("\t Precision : {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision, recall, f1))
            my_logger.info("\t Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

            torch.save(model, '%s/' % 'checkpoints' + '%02d' % (epoch + 1) + '.pt')  # 保存模型

        best_prec = precision_t[best_epoch-1]
        best_rec = recall_t[best_epoch-1]
        best_f1 = f1_t[best_epoch-1]

        best_prec_list.append(best_prec*100)
        best_rec_list.append(best_rec*100)
        best_f1_list.append(best_f1*100)
        best_acc_list.append(best_acc*100)
        ti_end = time.time()
        # print("The: {}/{}, Best Accuracy: {:.4f}, Best Recall: {:.4f},Best Precision:"
        #       " {:.4f},Best F1: {:.4f},Time: {:.3f}s".format(ti + 1, 20,best_acc,best_rec,best_prec,best_f1,ti_end - ti_start))
        my_logger.info("The: {}/{}, Best Accuracy: {:.4f}, Best Recall: {:.4f},Best Precision:"
                       " {:.4f},Best F1: {:.4f},Time: {:.3f}s".format(ti + 1, 20, best_acc, best_rec, best_prec,
                                                                      best_f1,
                                                                      ti_end - ti_start))

    # print(best_acc_list)
    # print(cal_ms(best_acc_list))
    # print(best_rec_list)
    # print(cal_ms(best_rec_list))
    # print(best_prec_list)
    # print(cal_ms(best_prec_list))
    # print(best_f1_list)
    # print(cal_ms(best_f1_list))

    my_logger.info(best_acc_list)
    my_logger.info(cal_ms(best_acc_list))
    my_logger.info(best_rec_list)
    my_logger.info(cal_ms(best_rec_list))
    my_logger.info(best_prec_list)
    my_logger.info(cal_ms(best_prec_list))
    my_logger.info(best_f1_list)
    my_logger.info(cal_ms(best_f1_list))

    return model, history

def PredictModel(config):

    batchsize = config.batch_size

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    #loaddata
    data_train, data_test = load_dataset_scene15()
    valid_data_size = len(data_test)

    test_loader = DataLoader(
        data_test,
        batch_size=batchsize,
        shuffle=False,
        # pin_memory=True,
    )


    model = torch.load('./checkpoints/183.pt')

    qnn_params = model.parameters()

    valid_acc = 0
    with torch.no_grad():
        model.eval()

        y_true = []
        y_pred = []
        for j, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            valid_acc += acc.item() * inputs.size(0)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
    avg_valid_acc = valid_acc / valid_data_size
    print("Accuracy: {:.4f}%".format(avg_valid_acc))
    return y_true,y_pred,qnn_params

import matplotlib.pyplot as plt
if __name__ == '__main__':

    train_flag = True

    from argparse import Namespace

    config = Namespace(
        project_name='wandb_fuzzy',
        batch_size=128,
        # hidden_layer_width=64,
        # dropout_p=0.1,
        lr=1e-2,
        optim_type='SGD',
        epochs=100,
        ckpt_path='./checkpoints'
    )

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if train_flag:
        # logger = get_logger()
        all_start = time.time()
        model, history = trainModel(config)
        history = np.array(history)

        num_epochs = config.epochs
        # Loss曲线
        plt.figure(figsize=(10, 10))
        plt.plot(history[:, 0:2])
        plt.legend(['Tr Loss', 'Val Loss'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        # 设置坐标轴刻度
        plt.xticks(np.arange(0, num_epochs + 1, step=10))
        plt.yticks(np.arange(0, 2.05, 0.1))
        plt.grid()  # 画出网格
        plt.savefig('cifar10_shuffle_' + '_loss_curve1.png')

        # 精度曲线
        plt.figure(figsize=(10, 10))
        plt.plot(history[:, 2:4])
        plt.legend(['Tr Accuracy', 'Val Accuracy'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        # 设置坐标轴刻度
        plt.xticks(np.arange(0, num_epochs + 1, step=10))
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.grid()  # 画出网格
        plt.savefig('cifar10_shuffle_' + '_accuracy_curve1.png')

        all_end = time.time()
        all_time = round(all_end - all_start)
        print('all time: ', all_time, ' 秒')
        print("All Time: {:d} 分 {:d} 秒".format(all_time // 60, all_time % 60))
    else:
        y_true,y_pred,qnn_params = PredictModel(config)
        print(len(y_true),len(y_pred))
        print(y_true)
        print(y_pred)
