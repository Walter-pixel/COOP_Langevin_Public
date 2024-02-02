import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


'''
Most of the code borrow from 
https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
- have modify the way yaml config file is read for initialize our clip model using this script
- The prompt generator is contrastive loss friendly:
  (1) you can specify only output prompt for classes in the mini-batch to lower memory consumption during training,
  (2) or you can specify output all classes' prompt together during inference
'''


_tokenizer = _Tokenizer() # use in prompt_learner




def load_clip_to_cpu(vision_backbone_name, download_root: str = None):

    if vision_backbone_name in ["RN50", "RN101","RN50x4","RN50x16", "RN50x64","ViT-B/32","ViT-B/16", "ViT-L/14", "ViT-L/14@336px"] == False:
        print(f"Error: Specified clip vision model is not supported in CLIP ckeckpoint")

    url = clip._MODELS[vision_backbone_name]
    model_path = clip._download(url, download_root or osp.expanduser("./clip_pretrained"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].type(self.text_projection.dtype) @ self.text_projection

        return x


# class PromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.PROMP_LEARNER.N_CTX #cfg.TRAINER.COOP.N_CTX
#         ctx_init = cfg.PROMP_LEARNER.CTX_INIT #cfg.TRAINER.COOP.CTX_INIT
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         # clip_imsize = clip_model.visual.input_resolution
#         # cfg_imsize = cfg.INPUT.SIZE[0]
#         # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
#
#         if ctx_init:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#
#         else:
#             # random initialization
#             if cfg.PROMP_LEARNER.CSC: #cfg.TRAINER.COOP.CSC:
#                 print("Initializing class-specific contexts")
#                 ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             else:
#                 print("Initializing a generic context")
#                 ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")
#
#         self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
#
#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
#
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens
#         # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
#         self.class_token_position = cfg.PROMP_LEARNER.CLASS_TOKEN_POSITION
#
#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
#
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#
#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim)
#                     ctx,  # (n_cls, n_ctx, dim)
#                     suffix,  # (n_cls, *, dim)
#                 ],
#                 dim=1,
#             )
#
#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i: i + 1, :, :]
#                 class_i = suffix[i: i + 1, :name_len, :]
#                 suffix_i = suffix[i: i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         ctx_i_half1,  # (1, n_ctx//2, dim)
#                         class_i,  # (1, name_len, dim)
#                         ctx_i_half2,  # (1, n_ctx//2, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
#
#         elif self.class_token_position == "front":
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i: i + 1, :, :]
#                 class_i = suffix[i: i + 1, :name_len, :]
#                 suffix_i = suffix[i: i + 1, name_len:, :]
#                 ctx_i = ctx[i: i + 1, :, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         class_i,  # (1, name_len, dim)
#                         ctx_i,  # (1, n_ctx, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
#
#         else:
#             raise ValueError
#
#         return prompts






class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.PROMP_LEARNER.N_CTX #cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.PROMP_LEARNER.CTX_INIT #cfg.TRAINER.COOP.CTX_INIT
        ctx_dim = clip_model.ln_final.weight.shape[0]


        # random initialization for context
        if cfg.PROMP_LEARNER.CSC: #cfg.TRAINER.COOP.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.class_token_position = cfg.PROMP_LEARNER.CLASS_TOKEN_POSITION

    def forward(self, labels, mode:str):
        '''
        mode="minibatch":   this only create prompts for classes in the sampled batch
        mode="all_classes": this create prompts for ALL classes
        '''
        bz = labels.size()[0]

        if mode=="minibatch":
            prefix_in_batch = torch.index_select(self.token_prefix, 0, labels)
            suffix_in_batch = torch.index_select(self.token_suffix, 0, labels)

            # if it is class agnostic
            if self.ctx.dim() == 2:
                ctx_matrix_bz = self.ctx.unsqueeze(0).repeat(bz, 1, 1) # simply make copies as the agnostic shares the same context
                prompts = torch.cat(
                    [
                        prefix_in_batch,  # (bz, 1, dim)
                        ctx_matrix_bz,  # (bz, n_ctx, dim)
                        suffix_in_batch,  # (bz, *, dim)
                    ],
                    dim=1,
                )
            # if it is class specific
            elif self.ctx.dim() == 3:
                ctx_matrix_bz = torch.index_select(self.ctx, 0, labels)
                prompts = torch.cat(
                    [
                        prefix_in_batch,  # (bz, 1, dim)
                        ctx_matrix_bz,  # (bz, n_ctx, dim)
                        suffix_in_batch,  # (bz, *, dim)
                    ],
                    dim=1,
                )

        elif mode=="all_classes":
            prefix = self.token_prefix
            suffix = self.token_suffix
            # if it is class agnostic
            if self.ctx.dim() == 2:
                ctx_matrix_bz = self.ctx.unsqueeze(0).repeat(self.n_cls,1,1)  # simply make copies as the agnostic shares the same context
                prompts = torch.cat(
                    [
                        prefix,         # (n_cls, 1, dim)
                        ctx_matrix_bz,  # (n_cls, n_ctx, dim)
                        suffix,         # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            # if it is class specific
            elif self.ctx.dim() == 3:
                prompts = torch.cat(
                    [
                        prefix,     # (n_cls, 1, dim)
                        self.ctx,   # (n_cls, n_ctx, dim)
                        suffix,     # (n_cls, *, dim)
                    ],
                    dim=1,
                )

        return prompts








#==================================================================
#
#
# def build_txt_enc_input_only_batchclass(self, ctx, labels):
#         '''
#         ctx: (n_cls, n_ctx, dim) # combine your prompts of each class with the class name --> to feed to text encoder
#         only create prompts for classes in the sampled batch <to save GPU memory>
#         each image featre corrrespond to its class's prompt (so can create the same prompt twice if you have 2 image samples belong to the same class)
#         '''
#         prefix_in_batch = torch.index_select(self.token_prefix, 0, labels)
#         suffix_in_batch = torch.index_select(self.token_suffix, 0, labels)
#
#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix_in_batch,  # (bz, 1, dim)
#                     ctx,  # (bz, n_ctx, dim)
#                     suffix_in_batch,  # (bz, *, dim)
#                 ],
#                 dim=1,
#             )
#         return prompts
#
#
#
#     def build_txt_enc_input_all_class(self, ctx):
#         '''
#         ctx: (n_cls, n_ctx, dim) # combine your prompts of each class with the class name --> to feed to text encoder
#         '''
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#
#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim)
#                     ctx,  # (n_cls, n_ctx, dim)
#                     suffix,  # (n_cls, *, dim)
#                 ],
#                 dim=1,
#             )
#         return prompts
