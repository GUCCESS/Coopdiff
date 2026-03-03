
import argparse
import os
import statistics

import torch
import torch.nn as nn
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset, LateFusionDataset
from opencood.tools import train_utils
import torch.nn.functional as F

import torch.multiprocessing as mp

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, default='',
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt

use_late = False

def main():

    #mp.set_start_method('spawn')

    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)
    
    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)
        
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=16,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=16,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=16,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True, #True  是否打乱
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=16,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)


    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    # # 加载预训练权重
    # pretrain_path = '/home/my/桌面/chengong/DSRC/opencood/logs/A_opv2v_diffuser_withlocal/net_epoch1.pth'
    # pre_weights = torch.load(pretrain_path)
    #
    # # 提取 pillar_vfe 权重，并调整键名
    # pillar_vfe_weights = {
    #     key.replace("pillar_vfe.", ""): value
    #     for key, value in pre_weights.items()
    #     if key.startswith("pillar_vfe.")
    # }
    #
    # # 提取 backbone 权重，并调整键名
    # backbone_weights = {
    #     key.replace("backbone.", ""): value
    #     for key, value in pre_weights.items()
    #     if key.startswith("backbone.")
    # }
    #
    # # 提取 backbone 权重，并调整键名
    # cls_head_weights = {
    #     key.replace("cls_head.", ""): value
    #     for key, value in pre_weights.items()
    #     if key.startswith("cls_head.")
    # }
    #
    # reg_head_weights = {
    #     key.replace("reg_head.", ""): value
    #     for key, value in pre_weights.items()
    #     if key.startswith("reg_head.")
    # }
    #
    # shrink_conv_weights = {
    #     key.replace("shrink_conv.", ""): value
    #     for key, value in pre_weights.items()
    #     if key.startswith("shrink_conv.")
    # }
    #
    # fuse_moudles_weights = {
    #     key.replace("fuse_modules.", ""): value
    #     for key, value in pre_weights.items()
    #     if key.startswith("fuse_modules.")
    # }
    #
    #
    # # 加载权重到模型
    # model.pillar_vfe.load_state_dict(pillar_vfe_weights)
    # model.backbone.load_state_dict(backbone_weights)
    # model.cls_head.load_state_dict(cls_head_weights)
    # model.reg_head.load_state_dict(reg_head_weights)
    # model.shrink_conv.load_state_dict(shrink_conv_weights)
    #

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hasattr(model_without_ddp, 'update_epoch'):
            model_without_ddp.update_epoch(epoch)

        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)

        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):

            # if (batch_data['ego']['record_len'] < 2).any():
            #     continue


            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # batch_data_list = train_utils.to_device(batch_data_list, device)

            # batch_data = batch_data_list[0]
            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well

            # from torchinfo import summary
            # results = summary(
            #     model,
            #     input_data=(batch_data['ego'],),
                
            # )
            # total_params = results.total_params  # 总参数量（int）
            # print(f"Total parameters: {total_params / 1e6:.2f} M")

            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])

                if  'late_loss' in ouput_dict:
                    final_loss = final_loss + 10*ouput_dict['late_loss']

                    print('final_loss|late_loss ', final_loss.item(), 10*ouput_dict['late_loss'].item())

                if 'diff_grad_loss' in ouput_dict:
                    final_loss = final_loss + ouput_dict['diff_grad_loss']

                if 'psm_singe' in ouput_dict:
                        ouput_dict['psm'] = ouput_dict['psm_singe']
                        ouput_dict['rm'] = ouput_dict['rm_singe']
                        print('loss before', final_loss.item())
                        final_loss += criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                        print('loss after', final_loss.item())
                
                if 'reg_loss' in ouput_dict:
                    final_loss = final_loss + ouput_dict['reg_loss']
                    print('final_loss|reg_loss ', final_loss.item(), ouput_dict['reg_loss'].item())

            else:
                with torch.cuda.amp.autocast():
                    # # ouput_dict = model(batch_data_list)
                    # ouput_dict = model(batch_data['ego'])
                    # final_loss = criterion(ouput_dict,
                    #                    batch_data['ego']['label_dict'])
                    # rec_loss = F.mse_loss(ouput_dict['x_rec'],ouput_dict['x_idea'])
                    # final_loss = final_loss+rec_loss+ouput_dict['mask_loss']
                    # # # final_loss = final_loss+rec_loss
                    # print("[epoch %d][%d/%d], || Loss: %.4f || Rec Loss: %.4f || Mask Loss: %.4f" % (
                    #         epoch, i + 1, len(train_loader),
                    #         final_loss.item(),rec_loss.item(), ouput_dict['mask_loss'].item(), ouput_dict['offset_loss'].item()))
                    ouput_dict = model(batch_data['ego'])
                    # ouput_dict = model(batch_data['ego'])
                    # first argument is always your output dictionary,
                    # second argument is always your label dictionary.

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])

                    if 'late_loss' in ouput_dict:
                        final_loss = final_loss +  10*ouput_dict['late_loss']
                        print('final_loss|late_loss ', final_loss.item(), 10*ouput_dict['late_loss'].item())


                    if 'psm_singe' in ouput_dict:
                        ouput_dict['psm'] = ouput_dict['psm_singe']
                        ouput_dict['rm'] = ouput_dict['rm_singe']
                        print('loss before', final_loss.item())
                        final_loss += criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                        print('loss after', final_loss.item())


            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
                # torch.cuda.empty_cache()

            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])
                    # ouput_dict = model(batch_data_list)

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    
                    if 'psm_singe' in ouput_dict:
                        ouput_dict['psm'] = ouput_dict['psm_singe']
                        ouput_dict['rm'] = ouput_dict['rm_singe']
                        print('loss before', final_loss.item())
                        final_loss += criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                        print('loss after', final_loss.item())
                    
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            with open(os.path.join(saved_path, 'valid_ave_loss.txt'), 'a+') as f:
                msg = 'Epoch: {} | valid_ave_loss: {:.04f} \n'.format(epoch, valid_ave_loss)
                f.write(msg)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
