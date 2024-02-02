import torch
from PIL import Image
from data.dataloader import *
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import os
from utils_my.CBLoss import *
from utils_my.utils import *
import argparse
# from Resnet_model.models_224x224.resnet224x224_DropFcBlocks import *
import time
from clip_customize.learn_prompt_contrasive_friendly import*
import yaml
from utils_my.munch import DefaultMunch
import ast
from Resnet_model_my.models_32x32.resnet_cifar import resnet32_w_adpator
from Resnet_model_my.models_224x224.resnet224x224 import resnet152_adaptor, scratch_resnet50_adaptor

'''
This script is the baseline for our prompt distribution learning
(1) It only updates the prompt parameters
(2) both text and image encoder are fixed at all time
(3) This cript is for baseline CoOp + prp + logit-adj loss
'''

def main(args):
    #==========================================================================
    torch.backends.cudnn.benchmark = True
    seed_all(args.seed)
    writer = SummaryWriter(args.tensorboard)
    log_path = os.path.join(args.tensorboard,'clip.log')
    logfile = open(log_path, 'w')
    # flush out the arguments
    argdict = vars(args)
    print(argdict)
    for k, v in argdict.items():
        logfile.write(k + ': ' + str(v) + '\n')
    logfile.write('\n')
    logfile.close()
    #==========================================================================

    seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = {'ImageNet_LT': '/home/walterl/ml20_scratch/walterl/data/ImageNet2012',
                 'iNaturalist18': '/home/walterl/ml20_scratch/walterl/data/iNaturalist2018',
                 # 'Places_LT': '/home/walterl/ml20_scratch/walterl/data/places365',
                 'Places_LT': '/home/walterl/oy30/walterl/data/Places365',
                 # 'CIFAR100_LT': '/media/wlia0021/hdd/public_dataset/cifar-100-python',
                 # 'CIFAR10_LT': '/media/wlia0021/hdd/public_dataset/cifar-10-python'}
                 'CIFAR100_LT': '/home/walterl/ml20_scratch/walterl/data/cifar100',
                 'CIFAR10_LT': '/home/walterl/ml20_scratch/walterl/data/cifar10'}


    # generated sub-datasets all have test split
    splits = ['train', 'val']
    if args.dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')

    if args.cifar_imb_ratio != 'None':
        cifar_imb_ratio = float(args.cifar_imb_ratio)
    else:
        cifar_imb_ratio = None

    data = {x: load_data_resolution(data_root=data_root[args.dataset],
                         dataset=args.dataset,
                         phase=x,
                         batch_size=args.batch_size,
                         batch_size_tst_val=args.batch_size,
                         num_workers=args.workers,
                         top_k_class=None,
                         reverse=False,
                         cifar_imb_ratio=cifar_imb_ratio,
                         resolution=args.resolution,
                         )
            for x in splits}

    loaders = {key: item[1] for key, item in data.items()}
    imb_num_per_cls = data['train'][2]

    # Decide which label belongs to head/mid/low category
    data_groups = {'head_lbs': [], 'mid_lbs': [], 'low_lbs': [], 'head_smpls': 0, 'mid_smpls': 0, 'low_smpls': 0}
    for lb in range(len(imb_num_per_cls)):
        if imb_num_per_cls[lb] >= 100:
            data_groups['head_lbs'].append(lb)
            data_groups['head_smpls'] += imb_num_per_cls[lb]
        elif imb_num_per_cls[lb] <= 20:
            data_groups['low_lbs'].append(lb)
            data_groups['low_smpls'] += imb_num_per_cls[lb]
        else:
            data_groups['mid_lbs'].append(lb)
            data_groups['mid_smpls'] += imb_num_per_cls[lb]


    # read all class label names
    if args.dataset.startswith('CIFAR'):
        classnames = data['train'][1].dataset.classes
    elif args.dataset.startswith('Places'):
        name_txt = open("data/Places_LT_v2/label_name.txt","r").read()
        name_list = name_txt.split("\n")
        name_list = [i.split('/')[2] for i in name_list] # return [class_name id, class_name id,...]
        classnames = [i.split(' ')[0] for i in name_list] # return [class_name, class_name,...]
    elif args.dataset.startswith('ImageNet'):
        with open("data/ImageNet_LT/label_name.txt","r") as f:
            name_dict = ast.literal_eval(f.read())
        name_list = list(name_dict.values())
        classnames = [i.split(',')[0] for i in name_list]

    clip_model = load_clip_to_cpu(vision_backbone_name='RN50') # load CLIP trained txt & vision encoder

    with open(args.clip_config_path) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        print(cfg_dict)
        # below is from https://github.com/Infinidat/munch
        # due to : https://stackoverflow.com/questions/52570869/load-yaml-as-nested-objects-instead-of-dictionary-in-python
        cfg = DefaultMunch.fromDict(cfg_dict)


    imb_clip_model = CLIP_prp_img(cfg,classnames=classnames,clip_model=clip_model)

    # options of image encoder:
    if args.im_enc_type == 'clip_rn50':
        print(f"Image encoder: clip trained resnet-50")
        pass
    elif args.im_enc_type == 'cifar_rn32':
        print(f"Image encoder: cifar resnet-32 from scratch")
        del imb_clip_model.image_encoder
        imb_clip_model.image_encoder = resnet32_w_adpator(adaptor_out_dim=1024).type(args.dtype)

    elif args.im_enc_type == 'caffe_rn152':
        print(f"Image encoder: caffe pretrained resnet-152")
        del imb_clip_model.image_encoder
        imb_clip_model.image_encoder = resnet152_adaptor(caffe_ckpt='Resnet_model_my/resnet152_caffe/resnet152.pth',
                                                  text_enc_dim=1024, dtype=clip_model.dtype)
    elif args.im_enc_type == 'scratch_rn50':
        print(f"Image encoder: scratch resnet-50")
        del imb_clip_model.image_encoder
        imb_clip_model.image_encoder = scratch_resnet50_adaptor(pretrained=False,text_enc_dim=1024).type(args.dtype)# convert image encoder to Halftype as the same text encoder data type in the CLIP

    imb_clip_model.to(device)

    # Build optimizer:
    optim_prompt = optim.SGD(imb_clip_model.prompt_learner.parameters(),
                             lr=args.lr_prompt, momentum=args.momentum, weight_decay=args.wd)
    optim_im = optim.SGD(imb_clip_model.image_encoder.parameters(),
                             lr=args.lr_im, momentum=args.momentum, weight_decay=args.wd)

    # lr-schedular: Must include "last_epoch" keyword for resume training
    if args.lr_type == 'exp':
        lr_sch_im = torch.optim.lr_scheduler.ExponentialLR(optim_im, gamma=args.lr_ratio)
        lr_sch_prompt = torch.optim.lr_scheduler.ExponentialLR(optim_prompt, gamma=args.lr_ratio)
    elif args.lr_type == 'multistep':
        lr_sch_im = torch.optim.lr_scheduler.MultiStepLR(optim_im, milestones=args.list_steplr,gamma=args.lr_ratio)
        lr_sch_prompt = torch.optim.lr_scheduler.MultiStepLR(optim_prompt, milestones=args.list_steplr,gamma=args.lr_ratio)
    elif args.lr_type == 'coslr':
        lr_sch_im = torch.optim.lr_scheduler.CosineAnnealingLR(optim_im, T_max=args.epochs, eta_min=0.)
        lr_sch_prompt = torch.optim.lr_scheduler.CosineAnnealingLR(optim_prompt, T_max=args.epochs, eta_min=0.)


    # build the loss
    all_losses = criterion(args, device, imb_num_per_cls=imb_num_per_cls)
    info = info_store(device=device, tf_writer=writer, imb_num_per_cls=imb_num_per_cls, data_groups=data_groups)




    # Trianing script
    print('\n-----Training Starts-----\n')
    for epoch in range(args.epochs):

        print(f"\n|| Train model epoch {epoch}/{args.epochs}....")
        train_start_time = time.time()
        train(loaders['train'], epoch, args, info, imb_clip_model, optim_prompt, optim_im, all_losses)
        train_epoch_time = time.time() - train_start_time
        print('Train 1epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(train_epoch_time)))

        lr_sch_prompt.step()
        lr_sch_im.step()


        if (epoch % args.infer_at_epoch == 0 and epoch>0) or epoch == args.epochs - 1:
            print('\n|| Testing the models......')
            test_start_time = time.time()

            if args.dataset =="iNaturalist18": # iNaturalist has no test set
                phase = 'val'
            else:
                phase = 'test'

            test(loaders[phase], loaders['train'], epoch, args, info, imb_clip_model)
            test_epoch_time = time.time() - test_start_time
            print('Test time is {:02}h{:02}m{:02}s'.format(*transform_time(test_epoch_time)))
        info.writer.flush()

    info.writer.close()
    print('\n-----KD Training Ends-----\n')






def train(loader, epoch, args, info, imb_clip_model, optim_prompt, optim_im, all_losses):

    batch_time = AverageMeter('Time', ':.3f')
    loss_record = AverageMeter('main_loss', ':.3f')
    end = time.time()
    total_pred = torch.tensor([])
    total_true = torch.tensor([])

    if torch.cuda.device_count() > 1:
        md = 'imb_clip_model.module'
    else:
        md = 'imb_clip_model'

    eval(md).text_encoder.eval()
    eval(md).prompt_learner.train()
    eval(md).image_encoder.train()

    print(f'Set requires grad false in text encoder')
    for param in eval(md).text_encoder.parameters():
        param.requires_grad = False



    for step, (images, labels, _) in enumerate(loader):
        print("\r" + "Epoch: {} Batch :{}".format(epoch, step) + "/" + str(len(loader)),
              end="", flush=True)

        total_true = torch.cat((total_true, labels), 0)
        images = images.to(info.device)
        labels = labels.to(info.device)
        optim_prompt.zero_grad()
        optim_im.zero_grad()
        output = imb_clip_model(images)

        # if stochastic tau for logit adj is selected
        if args.loss_type=='logit_adj' and len(args.stoch_tau_range)==1:
            # if pass only 1 argument in range, then it is deterministic tau
            all_losses.tau = args.stoch_tau_range[0]
        elif args.loss_type=='logit_adj' and len(args.stoch_tau_range)==2:
            lower_bound = float(args.stoch_tau_range[0])
            upper_bound = float(args.stoch_tau_range[1])
            all_losses.tau = (upper_bound - lower_bound) * (np.random.rand()) + lower_bound
        else:
            pass

        loss_main = all_losses.mainloss(logits=output, labels=labels)

        if torch.isnan(loss_main):
            print(f"labels:\n{labels}")
            print(f"loss:\n{loss_main}")

        _, preds = output.max(dim=-1)
        total_pred = torch.cat((total_pred,preds.to('cpu')), 0)

        loss_main.backward()
        if args.grad_clip_max != None:
            torch.nn.utils.clip_grad_norm_(imb_clip_model.parameters(),
                                           max_norm=float(args.grad_clip_max))

        optim_prompt.step()
        optim_im.step()
        loss_record.update(loss_main.detach().item(), labels.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    # End of this epoch do:--------------------------------------------------

    # Overall accuracy
    acc_top1 = (total_pred == total_true).sum().item() / len(total_true)
    many_acc_top1, \
    median_acc_top1, \
    low_acc_top1 = shot_acc(total_pred, total_true, loader)

    # Top-1 accuracy and additional string
    print_str = [
        'Many_shot_accuracy_top1: %.3f'
        % (many_acc_top1),
        'Median_shot_accuracy_top1: %.3f'
        % (median_acc_top1),
        'Low_shot_accuracy_top1: %.3f'
        % (low_acc_top1)
    ]
    print(print_str)
    print(f"Overall Training Acc: {acc_top1}")
    info.writer.add_scalar('Train/Acc@1', acc_top1, epoch)
    info.writer.add_scalar('Train/many_acc_top1', many_acc_top1, epoch)
    info.writer.add_scalar('Train/median_acc_top1', median_acc_top1, epoch)
    info.writer.add_scalar('Train/low_acc_top1', low_acc_top1, epoch)
    info.writer.add_scalar('Train/loss', loss_record.avg, epoch)
    info.writer.add_scalar('lr/lr_im', optim_im.param_groups[0]['lr'], epoch)
    info.writer.add_scalar('lr/lr_prompt', optim_prompt.param_groups[0]['lr'], epoch)

    if epoch % 50 ==0 or epoch == args.epochs-1:
        Save_Prompt(imb_clip_model, args, epoch,
                    save_path=os.path.join(args.tensorboard,'prompt_e'+str(epoch)+'.pth'),
                    optim_prompt=optim_prompt, optim_im=optim_im, loss=loss_main.detach().item())









def test(test_loader, train_loader, epoch, args, info, imb_clip_model):
    batch_time = AverageMeter('Time', ':.3f')
    loss_record = AverageMeter('main_loss', ':.3f')

    total_pred = torch.tensor([])
    total_true = torch.tensor([])
    imb_clip_model.eval()
    end = time.time()


    for step, (images, labels, _) in enumerate(test_loader):
        print("\r" + "Epoch: {} Batch :{}".format(epoch, step) + "/" + str(len(test_loader)),
              end="", flush=True)

        total_true = torch.cat((total_true, labels), 0)
        images = images.to(info.device)
        # labels = labels.to(info.device)

        with torch.no_grad():
            output = imb_clip_model(images)
            _, preds = output.max(dim=-1)
            total_pred = torch.cat((total_pred,preds.to('cpu')), 0)

        batch_time.update(time.time() - end)
        end = time.time()


    # End of this epoch do:--------------------------------------------------
    # Overall accuracy
    acc_top1 = (total_pred == total_true).sum().item() / len(total_true)

    many_acc_top1, \
    median_acc_top1, \
    low_acc_top1 = shot_acc(total_pred, total_true, train_loader) # train loader is used to distinguish what class is i nthe head/mid/low shot groups

    # Top-1 accuracy and additional string
    print_str = [
        'Many_shot_accuracy_top1: %.3f'
        % (many_acc_top1),
        'Median_shot_accuracy_top1: %.3f'
        % (median_acc_top1),
        'Low_shot_accuracy_top1: %.3f'
        % (low_acc_top1)
    ]
    print(print_str)
    print(f"Overall Testing Acc: {acc_top1}")
    info.writer.add_scalar('Test/Acc@1', acc_top1, epoch)
    info.writer.add_scalar('Test/many_acc_top1', many_acc_top1, epoch)
    info.writer.add_scalar('Test/median_acc_top1', median_acc_top1, epoch)
    info.writer.add_scalar('Test/low_acc_top1', low_acc_top1, epoch)



class CLIP_prp_img(nn.Module):
    '''
    README: this clip implementation intend to "only learn the prompt",
    so set "with torch.no_grad()" on image encoder to save memeory
    '''
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, get_feature_only=False):

        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if get_feature_only:
            return image_features, text_features
        else:
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits






class info_store:
    def __init__(self, device, tf_writer, imb_num_per_cls=None, data_groups={},):
        self.tst_record = {'best_top1': 0, 'best_top5': 0}
        self.val_record = {'best_top1': 0, 'best_top5': 0}
        self.device = device
        self.writer = tf_writer # tensorbaord writer
        self.imb_num_per_cls = imb_num_per_cls
        self.data_groups = data_groups






class criterion():

    def __init__(self, args, device, imb_num_per_cls, data_groups={}):
        self.loss_type = args.loss_type
        self.imb_num_per_cls = imb_num_per_cls # Index starts from class0 to classN
        self.device = device
        self.data_groups = data_groups # Indicates which label belongs to head/mid/low category
        # Below is for weighted CE--------------------------------------------
        self.args = args
        self.instance_w = None
        self.w_category = args.w_category
        # Below is for logit-adj and all its variant versions ----------------
        self.tau = None
        self.base_probs = None

        #---------- KD--------------------------------------------------------
        # put all the KD classes here
        # try:
        #     self.kd_loss = {'KD': DistillKL(T=self.args.temp),
        #                     'SPKD': SPKD(reduction='batchmean'),
        #                     # 'SPKD': SPKD_kernelize(reduction='batchmean', kernel_type='L-RBF', ker_coef=args.ker_coef1)
        #                     }
        # except:
        #     print("No KD in use")
        #---------------------------------------------------------------------

    def mainloss(self, logits, labels):
        # Note, since CBloss was written in the function form, not class, here we define a function wrapper
        if self.loss_type == 'CE':
            return F.cross_entropy(input=logits, target=labels)

        elif self.loss_type == 'wCE':

            # Build the weights first if not initialize it yet
            if self.instance_w == None:
                self.instance_w = self.build_w_for_ce()
                self.instance_w = torch.FloatTensor(self.instance_w).to(self.device)

            # create weighted CE:
            return F.cross_entropy(input=logits, target=labels, weight=self.instance_w)


        elif self.loss_type == 'logit_adj':
            # Initialize
            # self.tau = float(args.logit_adj_tau)
            if self.base_probs == None:
                self.base_probs = torch.from_numpy(self.imb_num_per_cls/sum(self.imb_num_per_cls)
                                          ).to(self.device)

            logits_adjusted = logits + torch.log(self.base_probs.pow(float(self.tau)) + 1e-12)
            return F.cross_entropy(input=logits_adjusted, target=labels)


        elif self.loss_type == 'logit_adj_groups':

            # Initialize
            # self.tau = float(args.logit_adj_tau)
            if self.base_probs == None:
                self.base_probs = np.zeros(len(self.imb_num_per_cls),dtype=float)
                head_prob = len(self.data_groups['head']) / len(self.imb_num_per_cls)
                mid_prob = len(self.data_groups['mid']) / len(self.imb_num_per_cls)
                low_prob = len(self.data_groups['low']) / len(self.imb_num_per_cls)

                for cls in range(len(self.imb_num_per_cls)):
                    if cls in self.data_groups['head']:
                        self.base_probs[cls] = head_prob
                    elif cls in self.data_groups['mid']:
                        self.base_probs[cls] = mid_prob
                    else:
                        self.base_probs[cls] = low_prob

                self.base_probs = torch.from_numpy(self.base_probs).to(self.device)

            logits_adjusted = logits + torch.log(self.base_probs.pow(float(self.tau)) + 1e-12)
            return F.cross_entropy(input=logits_adjusted, target=labels)

        else:
            # CB_focal/CB_softmax/CB_sigmoid
            cb_loss_type = self.loss_type.split("_")[1]
            return CB_loss(labels=labels, logits=logits,
                            samples_per_cls=self.imb_num_per_cls,
                            no_of_classes=self.args.num_classes,
                            loss_type=cb_loss_type,
                            beta=float(self.args.beta),
                            gamma=float(self.args.gamma),
                            device=self.device)

    def build_w_for_ce(self):
        # self.w_category has define which category should assign which weights,
        # based on this, create one-to-one weight/instance mapping for weighted CE loss
        w = [0] * self.args.num_classes
        for lb in range(len(self.imb_num_per_cls)):
            if self.imb_num_per_cls[lb] >= 100:
                w[lb] = float(self.w_category[0])  # assign head-cls weight value from user specified args.w_cateory
            elif self.imb_num_per_cls[lb] <= 20:
                w[lb] = float(self.w_category[2])
            else:
                w[lb] = float(self.w_category[1])
        return w





def Save_Prompt(imb_clip_model, args, epoch, save_path, optim_prompt, optim_im, loss):
    torch.save({
        'prompt_learner_state': imb_clip_model.module.prompt_learner.state_dict()
                                if torch.cuda.device_count() > 1 else imb_clip_model.prompt_learner.state_dict(),
        'im_enc_state': imb_clip_model.module.image_encoder.state_dict()
                                if torch.cuda.device_count() > 1 else imb_clip_model.image_encoder.state_dict(),
        'config': args,
        'at_epoch': epoch,
        'optim_prompt_state_dict': optim_prompt.state_dict(),
        'optim_im_state_dict': optim_im.state_dict(),
        'loss': loss,
    }, save_path)








if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PlacesLT')
    parser.add_argument('--seed', type=int, default=88, help='momentum')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--infer_at_epoch', type=int, default=10)

    # Dataset ----------------------------------------------------------------------------------------------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10_LT',
                        choices=['iNaturalist18', 'ImageNet_LT', 'Places_LT', 'CIFAR100_LT', 'CIFAR10_LT'])
    parser.add_argument('--cifar_imb_ratio', default='None', help='options are 0.01, 0.02, 0.1, None')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--resolution', type=int, help='image resolution to use')


    # Image encoder structure/txt encoder prompt initialization
    parser.add_argument('--im_enc_type', type=str, choices=['cifar_rn32','clip_rn50','caffe_rn152','scratch_rn50'])
    parser.add_argument('--clip_config_path', type=str, help='path to yaml config for setting up the text encoder prompt')

    # Training parameters
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--T', type=float, help='Temperature paramter')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay 1e-4')
    parser.add_argument('--lr_prompt', type=float, default=1e-6, help='prompt lr @ epoch=0')
    parser.add_argument('--lr_im', type=float, default=1e-6, help='backbone lr @ epoch=0')
    parser.add_argument('--lr_type', type=str, default='multistep', help='lr schedular',
                        choices=['exp', 'multistep', 'coslr'])
    parser.add_argument('--lr_ratio', type=float, help='learning rate decay ratio')
    parser.add_argument('--list_steplr', type=int, nargs='+',
                        help='Specify the StepLr changes at what epoch')
    parser.add_argument('--grad_clip_max', default=None)
    parser.add_argument('--dtype', type=str, help='data type used', choices=['fp32', 'fp16'], default='fp32')

    # parser.add_argument('--T0', type=float, help='initial cosine annealing temperature value')
    # parser.add_argument('--T_min', type=float, help='smallest cutoff temperature value')

    # Loss -------------------------------------------------------------------------------------------------------------
    parser.add_argument('--beta', default=None, help='for class balanced loss')
    parser.add_argument('--gamma', default=None, help='for class balanaced loss')
    parser.add_argument('--loss_type', type=str, default=None,
                        choices=['CB_focal', 'CB_sigmoid', 'CB_softmax', 'logit_adj', 'logit_adj_groups', 'CE', 'wCE'],
                        help='for class balanaced loss')
    parser.add_argument('--stoch_tau_range', default="False", nargs='+',
                        help="If false then not using stoch tau, "
                             "If lower and upper bound are given, then sample uniformly between them for tau. "
                             "The list should only contain 2 elements")
    parser.add_argument('--w_category', nargs='+', type=float,
                        help='For weighted CE, manually assign weight to H/M/L groups,'
                             'format: [H_w_value, M_w_value, L_w_value]')

    # Saving ------------------------------------------------------------------------------
    parser.add_argument('--tensorboard', type=str, default='./log/debug')

    args = parser.parse_args()
    main(args)