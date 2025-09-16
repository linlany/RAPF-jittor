

import copy
import pdb
import numpy as np
from omegaconf import DictConfig

# import clip
import jclip as clip
# import torch
# import torch.nn as nn
import jittor.nn as nn
import jittor as jt
jt.flags.use_cuda = 1

from .utils import get_class_ids_per_task, get_class_names

class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def shrink_cov(cov):
    diag_mean = jt.mean(jt.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = jt.eye(cov.shape[0], device=cov.device)
    alpha1 = 1
    alpha2  = 1
    cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
    return cov_
def sample(mean, cov, size, shrink=False):
    vec = jt.randn(size, mean.shape[-1])
    if shrink:
        cov = shrink_cov(cov)
    sqrt_cov = jt.Var(np.linalg.cholesky(cov.numpy()))
    # pdb.set_trace()
    vec = vec @ sqrt_cov.t()
    vec = vec + mean
    return vec

def cdist_jittor(XA, XB):
    # 计算XA和XB之间的欧几里得距离矩阵
    diff = XA.unsqueeze(1) - XB.unsqueeze(0)
    dist_matrix = jt.sqrt((diff * diff).sum(dim=2))
    return dist_matrix.squeeze()

class VisionClassifier(nn.Module):
    def __init__(self, in_features, num_classes, weight_init=None, activation=None):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=False)
        self.fc = nn.Parameter(self.fc.weight)
        if weight_init is not None:
            self.fc.data = weight_init
        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.Identity()
    
    def add_weight(self, weight):
        self.fc = nn.Parameter(jt.cat([self.fc, weight], dim=0))

    def set_weight(self, weight):
        self.fc = nn.Parameter(weight)


    def execute(self, x):
        # normalize the weights
        x = x / x.norm(dim=-1, keepdim=True)
        # weight = F.normalize(self.fc, p=2, dim=-1)
        weight = self.fc / self.fc.norm(dim=-1, keepdim=True)
        x = nn.linear(x, weight)
        x = self.activation(x)
        return x

class LoRACLIP(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        # self.model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)


        #lora_clip
        # pdb.set_trace()
        self.model, self.transforms = clip.load(name="/huanglinlan/jittor_CLIP/sup_code/ViT-B-16.pkl", lora_k_r=8, lora_v_r=8)
        # for name, param in self.model.named_parameters():
        #     if 'adapter_mlp' in name:
        #         param.requires_grad = True
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Trainable: {name}")

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.current_task = -1
    

    def cur_text_features(self):
        f = self.model.encode_text(self.text_tokens)
        f = f / f.norm(dim=1, keepdim=True)
        return f

    def inference(self, image, text_tokens):
        text_features = self.model.encode_text(text_tokens)
        image_features = self.model.visual(image.type(self.model.dtype), all_tokens=False, adapt=self.attention_adapter)
        # pdb.set_trace()

        # image_features = self.attention_adapter(image_features.type(torch.float32))[:, 0, :]

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def execute(self, image, test=False, all_test=False, return_feature=False,replay=None):
        if test:
            # pdb.set_trace()
            with jt.no_grad():
                if all_test:
                    if return_feature:
                        logits_per_image, _, image_features, __ = self.model(image, self.all_text_tokens, return_feature=return_feature)
                    else:
                        logits_per_image, _ = self.model(image, self.all_text_tokens)
                    # logits_per_image = self.inference(image, self.all_text_tokens)
                else:
                    if return_feature:
                        logits_per_image, _, image_features, __ = self.model(image, self.text_tokens, return_feature=return_feature)
                    else:
                        logits_per_image, _ = self.model(image, self.text_tokens)
                # pdb.set_trace()
                probs = logits_per_image.softmax(dim=-1)
        else:

            if return_feature:
                __, _, image_features, text_features = self.model(image, self.text_tokens, return_feature=return_feature)
                return image_features, text_features
            if replay is not None:
                logits_per_image, _ = self.model(image, self.text_tokens)
                # text_features_for_replay = self.model.encode_text(self.text_tokens[:-self.cfg.increment])
                text_features_for_replay = self.model.encode_text(self.text_tokens)
                text_features_for_replay = text_features_for_replay / text_features_for_replay.norm(dim=1, keepdim=True)
                replay_features = replay / replay.norm(dim=1, keepdim=True)
                replay_logits = replay_features @ text_features_for_replay.t() * 100
            else:
                logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image
                
        if return_feature:
            text_features = self.model.encode_text(self.all_text_tokens)
            return probs, image_features, text_features

        if replay is not None:
            return probs, replay_logits
        return probs

    def adaptation(self, task_id, reset=False):
        self.current_task +=1
            
        self.current_task_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names += self.current_task_class_names
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.current_task_text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_task_class_names]
        ).to(self.device)
        if self.current_task == 0:
            class_names = []
            for i in range(self.cfg.task_num):
                class_names += get_class_names(self.classes_names, self.class_ids_per_task[i])
            self.all_class_names = class_names
            self.all_text_tokens = clip.tokenize(
                [self.prompt_template.format(c) for c in self.all_class_names]
            ).to(self.device)


class ClassIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        # model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        model, self.transforms = clip.load("/huanglinlan/jittor_CLIP/sup_code/ViT-B-16.pkl")
        self.visual = model.visual
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.token_embedding = model.token_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        # pdb.set_trace()
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dtype = jt.float16 if cfg.fp16 else jt.float32
        self.adapter = nn.Linear(512, 512, bias=False)
        self.clip_type = model.dtype


        # old adapter
        self.old_adapter = None
        self.old_edge_samples = []
        self.old_edge_samples_labels = []
        self.old_edge_samples_nearest_labels = []

        # class stat
        self.class_mean_list = []
        self.class_cov_list = []

        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []



    def encode_text(self, text, prompt=False):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # pdb.set_trace()
        x = x[jt.arange(x.shape[0]), text.argmax(dim=-1)[0]] @ self.text_projection

        return x
    
    def encode_image(self, image):
        return self.visual(image)

    
    @jt.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        return class_name_features

    def execute(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, prompt=False):
        image = image
        with jt.no_grad():
            text_features = self.encode_text(self.text_tokens)


        with jt.no_grad():
            image_features = self.encode_image(image)
            original_image_features = image_features.clone()
        if memory_data is not None:
            memory_data = memory_data
            image_features = jt.cat([image_features, memory_data], dim=0)
        if edge_sample is not None:
            edge_sample = edge_sample
            edge_num = edge_sample.shape[0]
            image_features = jt.cat([image_features, edge_sample], dim=0)

        image_features = self.adapter(image_features.detach())

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if edge_sample is not None:
            edge_sample_features = image_features[-edge_num:]
            image_features = image_features[:-edge_num]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        
        probs = logits_per_image
        if not_ini:
            with jt.no_grad():
                old_memory_feature = self.old_adapter(memory_data)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return probs, image_features, old_memory_feature, edge_sample_features
            return probs, image_features, old_memory_feature, text_features
        if ori_ima_f:
            if memory_data is not None:
                image_features = image_features[:-memory_data.shape[0]]
            return probs, original_image_features, image_features
        return probs, image_features, None, None

    def adaptation(self, task_id, threshold=0):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        # pdb.set_trace()
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)
        self.queue_empty = True
        self.hard_pairs = None
        if task_id>0:
            self.old_adapter = copy.deepcopy(self.adapter)
            dist_list = []
            for k, class_name_feature in enumerate(self.class_name_features[:-len(self.class_ids_per_task[task_id])]):
                diff = cdist_jittor(self.class_name_features[-len(self.class_ids_per_task[task_id]):], class_name_feature.unsqueeze(0)).squeeze()
                dist_list.append(diff)
            dist_list = jt.stack(dist_list)
            self.class_diff = dist_list
            mask = self.class_diff < threshold
            indices = jt.nonzero(mask)
            self.hard_new_class = jt.unique(indices[:,1]) + self.cfg.initial_increment+(task_id-1) * self.cfg.increment
            num_hard_class = self.hard_new_class.shape[0]
            self.hard_pairs = indices
            # pdb.set_trace()
            self.hard_pairs[:,1] = self.hard_pairs[:,1]+self.cfg.initial_increment+(task_id-1) * self.cfg.increment
            self.hard_pairs = self.hard_pairs.numpy()
    def get_old_edge_samples(self, batch_size):
        random_select = jt.randperm(self.old_edge_samples.shape[0])[:batch_size]
        return self.old_edge_samples[random_select], self.old_edge_samples_labels[random_select], self.old_edge_samples_nearest_labels[random_select]


    def analyze_mean_cov(self, features, labels):
        label = jt.sort(jt.unique(labels))[0]
        for l in label:
            index = jt.nonzero(labels == l)
            index = index.squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            class_data_temp = class_data.numpy()

            cov = jt.Var(np.cov(class_data_temp.T)) + 1e-4* jt.init.eye(class_data.shape[-1])
            distance = cdist_jittor(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = jt.sort(distance)[0][-10:]
            self.class_edge_distance.append((max_distance.mean()-max_distance.min(), max_distance.max() - max_distance.mean(), max_distance.mean()))
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    def mix_matrix(self):
        if self.old_adapter is not None:
            weight_new = self.adapter.weight.data
            weight_old = self.old_adapter.weight.data
            dist = np.abs(weight_new - weight_old)
            U_old, S_old, V_old = np.linalg.svd(weight_old)
            P_new = U_old.T @ weight_new
            dist = np.abs(P_new - np.diag(S_old)@V_old)
            mask = dist / dist.max()
            mask += 0.5
            mask = np.clip(mask, a_max=1, a_min=0)
            right = P_new * mask + np.diag(S_old)@V_old * (1-mask)
            weight = U_old @ right
            self.adapter.weight.data = weight
            return



class DomainIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False) -> None:
        super().__init__()
        self.model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.text_tokens = None
        self.prompt_template = cfg.prompt_template
        self.device = device

    def forward(self, image):
        with jt.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs

    def tokenize(self, class_names):
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in class_names]
        ).to(self.device)



class TaskAgnosticCLIP(nn.Module):
    pass



def load_model(cfg: DictConfig, device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.
    
    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.
        
    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        if cfg.method == "RAPF":
            return ClassIncrementalCLIP(cfg, device)
        elif cfg.method == "MG-CLIP":
            return LoRACLIP(cfg, device)
        else:
            raise ValueError(f"""
                `{cfg.method}` is not a valid method for class-incremental learning, 
                Please choose from ['RAPF', 'MG-CLIP']
            """)
    elif cfg.scenario == "domain":
        return DomainIncrementalCLIP(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnosticCLIP(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)
    
