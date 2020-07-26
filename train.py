"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex

from torch.autograd import Variable

# import sys
# sys.path.append('/content/gdrive/My Drive/Research/U-2-Net')

# from model import U2NETP

# normalize the predicted SOD probability map
# def normPRED(d):
#     ma = torch.max(d)
#     mi = torch.min(d)

#     dn = (d-mi)/(ma-mi)

#     # out = (dn>0.5).float()
#     return dn

# SMOOTH = 1e-6

# def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     labels = labels.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
#     outputs = outputs.int()
#     labels = labels.int()
#     intersection = (outputs & labels).sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
#     return iou

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    model.train()

    # Using Saliency
    # u2_model_dir = '/content/gdrive/My Drive/Research/U-2-Net/saved_models/'+ 'u2netp' + '/' + 'u2netp.pth'
    # u2_net = U2NETP(3,1)
    # u2_net.load_state_dict(torch.load(u2_model_dir))
    
    # if torch.cuda.is_available():
    #     u2_net.cuda()

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    print(_config['mode'])
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0, 'dist_loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
        # Forward and Backward
        optimizer.zero_grad()
        
        # with torch.no_grad():
        #   # u2net
        #   inputs = query_images[0].type(torch.FloatTensor)
        #   labels = query_labels.type(torch.FloatTensor)
        #   if torch.cuda.is_available():
        #       inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
        #                                                                                   requires_grad=False)
        #   else:
        #       inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        #   d1,d2,d3,d4,d5,d6,d7= u2_net(inputs_v)

        #   # normalization
        #   pred = d1[:,0,:,:]
        #   pred = normPRED(pred)
        pred = []

        query_pred, align_loss, dist_loss = model(support_images, support_fg_mask, support_bg_mask,
                                       query_images, pred)
        query_loss = criterion(query_pred, query_labels)
        loss = query_loss + dist_loss + align_loss * 0.2 #_config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        dist_loss = dist_loss.detach().data.cpu().numpy() if dist_loss != 0 else 0
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        _run.log_scalar('dist_loss', dist_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss
        log_loss['dist_loss'] += dist_loss



        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, dist_loss: {dist_loss}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
