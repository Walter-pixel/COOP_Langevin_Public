from data.dataloader import *
from torch.utils.tensorboard import SummaryWriter
from utils_my.utils import *
import argparse
import time
import copy
from clip_customize.learn_prompt_contrasive_friendly import*
import yaml
from utils_my.munch import DefaultMunch
import ast
from utils_my.SGLD import *
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from data.dataloader import *


'''
This script is the CoOp wth Langevin-SGD MC to sample multiple prompt models.
(1) Only run inference using emsemble hard voting
 
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

    if args.epochs < args.burn_in_epoch+args.max_samples:
        print(f"Error: total epochs must be greater than (burns-in-epoch + max_samples)")
        exit()

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

    data = {x: load_data(data_root=data_root[args.dataset],
                         dataset=args.dataset,
                         phase=x,
                         batch_size=args.bz_trn,
                         batch_size_tst_val=args.bz_tst,
                         num_workers=args.workers,
                         top_k_class=None,
                         reverse=False,
                         cifar_imb_ratio=cifar_imb_ratio,
                         resolution=args.resolution,
                         bal_sampler=args.bal_sampler
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

    # load CLIP trained txt & vision encoder
    if args.im_enc_type == 'clip_rn50':
        clip_model = load_clip_to_cpu(vision_backbone_name='RN50')
    elif args.im_enc_type == 'clip_vitb':
        clip_model = load_clip_to_cpu(vision_backbone_name='ViT-B/32')



    with open(args.clip_config_path) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg_dict['PREC'] = args.dtype # overwrite the config file based on parser dtype
        print(cfg_dict)
        cfg = DefaultMunch.fromDict(cfg_dict)


    imb_clip_model = CLIP_No_Grad_ImEnc(cfg,classnames=classnames,clip_model=clip_model)
    if args.dtype == 'fp16':
        clip_model.type(torch.float16)
        imb_clip_model.dtype = torch.float16
    elif args.dtype == 'fp32':
        clip_model.type(torch.float32)
        imb_clip_model.dtype = torch.float32


    _, imb_clip_model, _ = deploy_model_prp(imb_clip_model, ckpt_path=args.ckpt_path,
                                                                       lr_prompt=args.lr_prompt, wd=args.wd,
                                                                       momentum=args.momentum,
                                                                       device=device, opt_type=args.opt_type)



    info = info_store(device=device, tf_writer=writer, imb_num_per_cls=imb_num_per_cls, data_groups=data_groups,
                      train_loader=loaders['train'])



    # Contrastive Training Script
    print('\n----- evaluation Starts-----\n')

        if epoch<args.burn_in_epoch:
            print(f"\n|| Before burn-in epoch {args.burn_in_epoch}, Training @ {epoch}/{args.epochs}-------------- ")
            train_start_time = time.time()
            train_one_epoch(loaders, epoch, args, info, imb_clip_model, net_dre, optim_prompt, optim_dre,
                                         contr_adj_loss, before_burn_in=True)
            lr_sch_prompt.step()
            lr_sch_dre.step()
            train_epoch_time = time.time() - train_start_time
            print('Train 1 epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(train_epoch_time)))

            if (epoch % args.infer_at_epoch == 0 and epoch > 0):
                # here only evaluate the testing set via passing all classes name into contrastive loss
                print(f"\n|| Before burn-in epoch {args.burn_in_epoch})"
                      f"\n|| One Model Evaluation @ {epoch}/{args.epochs}-------------- ")
                test_start_time = time.time()
                phase = "val" if args.dataset == "iNaturalist18" else "test" # iNaturalist has no test set
                eval_one_model(loaders[phase], args, info, epoch, imb_clip_model, contr_adj_loss, phase)
                test_epoch_time = time.time() - test_start_time
                print('One Model Test time is {:02}h{:02}m{:02}s'.format(*transform_time(test_epoch_time)))

        elif epoch >=args.burn_in_epoch:
            print(f"\n|| After burn-in epoch {args.burn_in_epoch}, Training @ {epoch}/{args.epochs}-------------------")
            train_start_time = time.time()
            train_one_epoch(loaders, epoch, args, info, imb_clip_model, net_dre, optim_prompt, optim_dre,
                                         contr_adj_loss, before_burn_in=False) # start saving models
            lr_sch_prompt.step()
            train_epoch_time = time.time() - train_start_time
            print('Train 1 epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(train_epoch_time)))

            if (epoch % args.infer_at_epoch == 0 and epoch > 0) or epoch ==args.epochs-1:
                # here only evaluate the testing set via passing all classes name into contrastive loss
                '''
                Below evaluates only one current model at hand
                '''
                print(f"\n|| After burn-in epoch {args.burn_in_epoch}, "
                      f"\n|| One Model Testing @ {epoch}/{args.epochs}------")
                test_start_time = time.time()
                phase = "val" if args.dataset == "iNaturalist18" else "test" # iNaturalist has no test set
                eval_one_model(loaders[phase], args, info, epoch, imb_clip_model, contr_adj_loss, phase)
                test_epoch_time = time.time() - test_start_time
                print('One Model Test time is {:02}h{:02}m{:02}s'.format(*transform_time(test_epoch_time)))


                '''
                Below ensemble all cached models together
                '''
                print(f"\n|| After burn-in epoch {args.burn_in_epoch}, "
                      f"\n|| ALL Models Testing @ {epoch}/{args.epochs}------")
                test_start_time = time.time()
                phase = "val" if args.dataset == "iNaturalist18" else "test"  # iNaturalist has no test set

                # check the number of models saved
                if len(info.model_bank) < int(args.max_samples):
                    print(f"Warning: Insufficient models. Only saved {len(info.model_bank)} "
                          f"out of {args.max_samples} models. Skip this evaluation.")
                else:
                    eval_all_cached_models(loaders[phase], args, info, epoch, imb_clip_model, contr_adj_loss, phase)
                test_epoch_time = time.time() - test_start_time
                print('ALL Models Test time is {:02}h{:02}m{:02}s'.format(*transform_time(test_epoch_time)))

        #===============================================================
        info.writer.flush()

    info.writer.close()
    Save_ALL_Prompt(info, args)
    print('\n-----LD Contrastive Training Ends-----\n')







def train_one_epoch(loaders, epoch, args, info, imb_clip_model, net_dre,
                    optim_prompt, optim_dre, contr_adj_loss, before_burn_in=True):
    '''
    before_burn_in: True--> save the model after one epoch for the ensemble use
    '''
    end = time.time()
    batch_time = AverageMeter('Time', ':.3f')
    record = {'loss_main': AverageMeter('loss_main', ':.3f'),
              'loss_density_net': AverageMeter('loss_density_net', ':.3f'),}

    if torch.cuda.device_count() > 1:
        md = 'imb_clip_model.module'
    else:
        md = 'imb_clip_model'

    eval(md).text_encoder.eval()
    eval(md).image_encoder.eval()
    eval(md).prompt_learner.train()
    print(f'Set requires grad false in text & image encoders')
    for param in eval(md).text_encoder.parameters():
        param.requires_grad = False
    for param in eval(md).image_encoder.parameters():
        param.requires_grad = False

    for step, (images, labels, _) in enumerate(loaders['train']):
        print("\r" + "Epoch: {} Batch :{}".format(epoch, step) + "/" + str(len(loaders['train'])), end="", flush=True)
        images = images.to(info.device)
        labels = labels.to(info.device)
        bz = images.size()[0]

        optim_prompt.zero_grad()
        optim_dre.zero_grad()
        image_feat_norm, text_feat_norm = imb_clip_model(images, labels, mode='minibatch')
        image_feat_norm_trn = image_feat_norm.clone().detach()

        net_dre.eval()
        with torch.no_grad():
            weights = net_dre(image_feat_norm)

        loss_main = contr_adj_loss(im_emb_norm=image_feat_norm,
                                   txt_emb_norm=text_feat_norm,
                                   lb=labels, weights=weights)
        loss_main.backward()
        optim_prompt.step()
        record['loss_main'].update(loss_main.detach().item(), labels.size(0))

        # train net_dre with first order moment matching
        net_dre.train()
        imb_clip_model.eval()
        optim_dre.zero_grad()
        optim_prompt.zero_grad()
        images_val, labels_val, _ = next(iter(loaders['val']))
        images_val = images_val.to(info.device)
        labels_val = labels_val.to(info.device)
        bz_val = labels_val.size()[0]

        with torch.no_grad():
            image_feat_norm_val, _ = imb_clip_model(images_val, labels_val, mode='minibatch')

        weights = net_dre(image_feat_norm_trn)


        # weights_val = net_dre(text_feat_norm_val)

        loss_val = (1 / bz ** 2) * torch.mm(torch.mm(weights.t(), image_feat_norm_trn),
                                            torch.mm(image_feat_norm_trn.t(), weights))\
                   - (2 / (bz * bz_val)) * torch.mm(torch.mm(weights.t(), image_feat_norm_trn),
                                                  torch.mm(image_feat_norm_val.t(), torch.ones(bz_val,1).cuda()))   # equation (3)

        loss_val.backward()
        optim_dre.step()
        record['loss_density_net'].update(loss_val.detach().item(), labels_val.size(0))
        batch_time.update(time.time() - end)
        end = time.time()


    # End of this epoch do:----------------------------------
    if before_burn_in:
        pass
    else:
        # save the prompt states @ each epoch after the burns in epoch
        if len(info.model_bank) >= args.max_samples:
            info.model_bank.pop(0)  # delete the earliest saved state dict
        cache_now = copy.deepcopy(eval(md).prompt_learner.state_dict())
        cache_now = move_to(cache_now, 'cpu')# save current prompt state to cpu
        info.model_bank.append(cache_now)
        print('>> Saving weight samples %d/%d' % (len(info.model_bank), args.max_samples))

    info.writer.add_scalar('OneModel/train_loss_main', record['loss_main'].avg, epoch)
    info.writer.add_scalar('OneModel/train_loss_density_net', record['loss_density_net'].avg, epoch)
    info.writer.add_scalar('lr/lr_prompt', optim_prompt.param_groups[0]['lr'], epoch)
    print(f"*- Train Loss main: {record['loss_main'].avg} ||  Prompt lr: {optim_prompt.param_groups[0]['lr']}")
    print(f"*- Train Loss Density Ratio Net: {record['loss_density_net'].avg} ||  Density ratio net lr: {optim_dre.param_groups[0]['lr']}")





def eval_one_model(loader, args, info, epoch, imb_clip_model, contr_adj_loss, phase:str):
    '''
    - Compare image representation wrt "all classes"
    - This function only evaluates current model at hand (only one model) to keep track of training performance
    '''
    batch_time = AverageMeter('Time', ':.3f')
    total_pred = torch.tensor([])
    total_true = torch.tensor([])
    end = time.time()

    if torch.cuda.device_count() > 1:
        md = 'imb_clip_model.module'
    else:
        md = 'imb_clip_model'

    imb_clip_model.eval()  # set all to be evaluation mode

    for step, (images, labels, _) in enumerate(loader):
        print("\r" + "Batch :{}".format(step) + "/" + str(len(loader)),
              end="", flush=True)
        total_pred = torch.tensor([])
        total_true = torch.tensor([])
        total_true = torch.cat((total_true, labels), 0)
        images = images.to(info.device)
        labels = labels.to(info.device)

        with torch.no_grad():
            image_feat_norm, text_feat_norm = imb_clip_model(images, labels, mode='all_classes')

            # text in all classes, so output similarity is logit
            if torch.cuda.device_count() > 1:
                n_gpu = torch.cuda.device_count()
                logits = pairwise_cosine_similarity(image_feat_norm,
                                                    torch.tensor_split(text_feat_norm, n_gpu, dim=0)[0])
            else:
                logits = pairwise_cosine_similarity(image_feat_norm, text_feat_norm)

            prob = F.softmax(logits,dim=-1)
            _, preds = prob.max(dim=-1)
            total_pred = torch.cat((total_pred, preds.to('cpu')), 0)

        batch_time.update(time.time() - end)
        end = time.time()

    # Overall accuracy
    acc_top1 = (total_pred == total_true).sum().item() / len(total_true)
    print(f"\nOverall {phase} acc: {acc_top1}")
    info.writer.add_scalar('OneModel/' + phase + '_Acc@1', acc_top1, epoch)
    many_acc_top1, \
    median_acc_top1, \
    low_acc_top1 = shot_acc(total_pred, total_true, info.train_loader_for_eval_shotacc)
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
    info.writer.add_scalar('OneModel/'+phase+'_acc_many', many_acc_top1, epoch)
    info.writer.add_scalar('OneModel/'+phase+'_acc_median', median_acc_top1, epoch)
    info.writer.add_scalar('OneModel/'+phase+'_acc_few', low_acc_top1, epoch)






def eval_all_cached_models(loader, args, info, epoch, imb_clip_model, contr_adj_loss, phase: str):
    '''
    - Compare image representation wrt "all classes"
    - step-1: Check if your cache have fully saved #Nsample of nmodel. If not, then skip this entire function.
    - stpe-2: Ensemble the cached #Nsample models together by:
                (1) First grab one saved model and run it through the complete dataset for predictions. Once done,
                (2) Then repeat again for the 2nd model.
                    Since the last model is saved at the last index; for keep training purpose, there is no need to
                    reload prompts, just keep using the last saved model.
    '''
    if torch.cuda.device_count() > 1:
        md = 'imb_clip_model.module'
    else:
        md = 'imb_clip_model'

    ensemble_logit = torch.zeros(args.max_samples, len(loader.dataset.labels), args.num_classes) # cache all model's output prob

    for idx, prompt_state in enumerate(info.model_bank):
        print(f"use model index : {idx+1} / {len(info.model_bank)}")

        eval(md).prompt_learner.load_state_dict(prompt_state)  # load a prompt state
        imb_clip_model.eval()  # set all to be evaluation mode

        previous_bz = 0
        for step, (images, labels, _) in enumerate(loader):
            print("\r" + "eval@ step/loader :{}".format(step + 1) +"/"+str(len(loader)), end="", flush=True)

            # # ==========================================================================================================
            # # Below just make sure your data loader is not shuffled, can be commented out if not needed
            # if idx==0 and step==3:
            #     check=labels
            # elif idx==1 and step==3:
            #     if torch.equal(check,labels):
            #         pass
            #     else:
            #         print("ERROR !!: dataloader in eval AllModels cannot be shuffled !")
            #         exit()
            # #===========================================================================================================

            images, labels = images.to(info.device), labels.to(info.device)
            bz = images.size()[0]

            with torch.no_grad():
                image_feat_norm, text_feat_norm = imb_clip_model(images, labels, mode='all_classes')
                # # below is sanity check for multi-gpu setting:
                # # --> text_feat_norm contains #gpu chunck of output, each chunck (partition) should be exactly identical
                # print(f"image_feat_norm dim: {image_feat_norm.size()}")
                # print(f"text_feat_norm dim: {text_feat_norm.size()}")
                # t0, t1, t2, t3 = torch.tensor_split(text_feat_norm, 4, dim=0)
                # print(f"t0 t1 equal: {torch.equal(t0,t1)}")
                # print(f"t1 t2 equal: {torch.equal(t1, t2)}")
                # print(f"t2 t3 equal: {torch.equal(t2, t3)}")

                # text in all classes, so output similarity is logit
                if torch.cuda.device_count() > 1:
                    n_gpu = torch.cuda.device_count()
                    logits = pairwise_cosine_similarity(image_feat_norm,
                                                        torch.tensor_split(text_feat_norm, n_gpu, dim=0)[0])
                else:
                    logits = pairwise_cosine_similarity(image_feat_norm,text_feat_norm)

            ensemble_logit[idx, step*previous_bz:(step*previous_bz+bz), :] = logits.detach().cpu()
            previous_bz=bz

        # # Below use "soft majority voting" that takes average n the probability output across saved models
        # mean_prob = F.softmax(ensemble_logit, dim=-1).mean(dim=0, keepdim=False)
        # pred_final = mean_prob.max(dim=-1)[1]

        # Below is hard ensemble voting
        # refer to : https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/ensemble/voting.py#L281
        pred_individual = torch.argmax(F.softmax(ensemble_logit, dim=-1), dim=-1) # dim=[num_models, dataset_len]
        pred_final = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=pred_individual)


    # Overall accuracy
    total_true = torch.Tensor(loader.dataset.labels)
    acc_top1 = (pred_final == total_true).sum().item() / len(total_true)
    print(f"\nOverall {phase} acc: {acc_top1}")
    info.writer.add_scalar('AllModel/' + phase + '_Acc@1', acc_top1, epoch)
    many_acc_top1, \
    median_acc_top1, \
    low_acc_top1 = shot_acc(pred_final, total_true, info.train_loader_for_eval_shotacc)
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
    info.writer.add_scalar('AllModel/' + phase + '_acc_many', many_acc_top1, epoch)
    info.writer.add_scalar('AllModel/' + phase + '_acc_median', median_acc_top1, epoch)
    info.writer.add_scalar('AllModel/' + phase + '_acc_few', low_acc_top1, epoch)














class CLIP_No_Grad_ImEnc(nn.Module):
    '''
    README: this clip implementation intend to "only learn the prompt",
    so set "with torch.no_grad()" on image encoder to save memory
    '''
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        # self.logit_scale = clip_model.logit_scale #original
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype

    def forward(self, image, labels, mode):
        if mode not in ['all_classes', 'minibatch']:
            raise ValueError("Invalid specification on mode, check PromptLearner forward method for requirement.")

        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner(labels, mode) # the output prompt has been matched to minibatch or all_classes setting
        prompts = prompts.type(self.dtype)

        if mode=="minibatch":
            minibatch_tokenized_prompts = torch.index_select(self.tokenized_prompts, 0, labels.cpu())
            text_features = self.text_encoder(prompts, minibatch_tokenized_prompts.to(prompts.device))
        elif mode=="all_classes":
            allcls_tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, allcls_tokenized_prompts.to(prompts.device))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        return image_features, text_features












class info_store:
    def __init__(self, device, tf_writer, imb_num_per_cls=None, data_groups={}, train_loader=None):
        self.tst_record = {'best_top1': 0, 'best_top5': 0}
        self.val_record = {'best_top1': 0, 'best_top5': 0}
        self.device = device
        self.writer = tf_writer # tensorbaord writer
        self.imb_num_per_cls = imb_num_per_cls
        self.data_groups = data_groups
        self.train_loader_for_eval_shotacc = train_loader
        self.model_bank = [] # create model bank that saves models in cache







def Save_ALL_Prompt(info, args):
    '''
    save multiple prompt states into one torch.save object
    '''
    model_bank_dir = os.path.join(args.tensorboard,'saved_model_bank')
    if not os.path.exists(model_bank_dir):
        os.makedirs(model_bank_dir)
    for idx, prompt_state in enumerate(info.model_bank):
        torch.save(prompt_state, os.path.join(model_bank_dir, f"prp{idx}.pth"))




def deploy_model_prp_eval(imb_clip_model, ckpt_path,lr_prompt, wd, momentum,device, opt_type='sgld', ):

    if torch.cuda.device_count() > 1:
        md = 'imb_clip_model.module'
    else:
        md = 'imb_clip_model'

    print(f"Let's use", torch.cuda.device_count(), "GPUs!")
    if ckpt_path != 'None':
        print(f"Load ckpt to complete unfinished training epoch")
        ckpt = torch.load(ckpt_path)
        imb_clip_model.prompt_learner.load_state_dict(ckpt['prompt_learner_state'])  # load prmpt gen state
        if torch.cuda.device_count() > 1:
            imb_clip_model = nn.DataParallel(imb_clip_model)
        imb_clip_model.to(device)
    return imb_clip_model







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PlacesLT')
    parser.add_argument('--seed', type=int, default=88, help='momentum')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--bz_trn', type=int, default=128, help='batch size')
    parser.add_argument('--bz_tst', type=int, default=128, help='batch size')
    parser.add_argument('--infer_at_epoch', type=int, default=10)

    # resume training:
    parser.add_argument('--ckpt_path', type=str, default='None',
                        help='if need to resume training til unfinished eopch, specify the path to saved ckpt .pth here')

    # Dataset ----------------------------------------------------------------------------------------------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10_LT',
                        choices=['iNaturalist18', 'ImageNet_LT', 'Places_LT', 'CIFAR100_LT', 'CIFAR10_LT'])
    parser.add_argument('--cifar_imb_ratio', default='None', help='options are 0.01, 0.02, 0.1, None')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--resolution', type=int, default=224, help='image resolution to use')
    parser.add_argument('--bal_sampler', type=str, default="False", choices=['True', 'False'],
                        help='if or not using the balanced sampler during training.')


    # Image encoder structure/txt encoder prompt initialization
    parser.add_argument('--im_enc_type', type=str, choices=['cifar_rn32','clip_rn50','caffe_rn152','scratch_rn50','clip_vitb'])
    parser.add_argument('--clip_config_path', type=str, help='path to yaml config for setting up the text encoder prompt')

    # Training parameters
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--temp_sim', type=float, help='temperature applies to similarity, '
                                                       'set it smaller to avoid adjusted logit to negative')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay 1e-4')
    parser.add_argument('--lr_prompt', type=float, default=1e-6, help='backbone lr @ epoch=0')
    parser.add_argument('--lr_type', type=str, default='multistep', help='lr schedular',
                        choices=['exp', 'multistep', 'coslr'])
    parser.add_argument('--lr_ratio', type=float, help='learning rate decay ratio')
    parser.add_argument('--list_steplr', type=int, nargs='+',
                        help='Specify the StepLr changes at what epoch')
    parser.add_argument('--grad_clip_max', default=None)
    parser.add_argument('--dtype', type=str, help='data type used', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--opt_type', type=str, help='data type used', choices=['sgd', 'adam', 'sgld'], default='sgd')

    # for density estimation net
    parser.add_argument('--lr_dre', type=float, default=0.05, help='density estimation net learning rate')
    parser.add_argument('--m_dre', type=float, default=0.9, help='density estimation net momentum')
    parser.add_argument('--wd_dre', type=float, default=1e-4, help='density estimation net weight decay')

    # ensemble model
    parser.add_argument('--burn_in_epoch', type=int, help='epoch that starts Langevin Dynamic MC sampling and saving model')
    parser.add_argument('--max_samples', type=int, default=10, help='After the burns in epoch, totally save how many models.'
                                                                    'save model at the end of an epoch.')

    # Loss --------------------------------------------------------------------------------
    parser.add_argument('--tau', default=1, type=float, help='tau of logit-adj ')
    # Saving ------------------------------------------------------------------------------
    parser.add_argument('--tensorboard', type=str, default='./log/debug')


    args = parser.parse_args()
    main(args)