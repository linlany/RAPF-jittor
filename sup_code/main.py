import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import pdb
import random
import hydra
import logging
from omegaconf import DictConfig

# import torch
import jittor as jt
import statistics
# from torch.utils.data import DataLoader
from jittor.dataset import DataLoader
# from continuum.metrics import Logger

from tqdm import tqdm
from collections import defaultdict
from continual_clip import utils
from continual_clip.models import load_model, sample
from continual_clip.datasets import build_cl_scenarios, MyImageFolder
import numpy as np
from jittor.dataset import ImageFolder
from continual_clip.models import VisionClassifier

jt.flags.use_cuda = 1

class MetricLogger:
    def __init__(self):
        self.correct = []
        self.total = []
        self.accuracy_per_task = []
        self.accuracy = 0.0
        self.max_task = 0

    def add(self, batch, subset="test"):
        preds, targets, task_ids = batch
        # preds, targets, task_ids 都是 numpy 或 jt.Var
        preds = np.array(preds)
        targets = np.array(targets)
        task_ids = np.array(task_ids)
        # pdb.set_trace()
        if len(self.correct) == 0:
            self.max_task += 1
            self.correct = [0 for _ in range(self.max_task)]
            self.total = [0 for _ in range(self.max_task)]
        for p, t, tid in zip(preds, targets, task_ids):
            tid = int(tid)
            self.total[tid] += 1
            if p == t:
                self.correct[tid] += 1

    def end_task(self):
        self.accuracy_per_task = []
        for c, t in zip(self.correct, self.total):
            if t > 0:
                self.accuracy_per_task.append(c / t)
            else:
                self.accuracy_per_task.append(0.0)
        if len(self.total) > 0 and sum(self.total) > 0:
            self.accuracy = sum(self.correct) / sum(self.total)
        else:
            self.accuracy = 0.0
        # 清空，为下一个任务做准备
        self.correct = []
        self.total = []

def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    # jt.manual_seed(seed)
    # jt.cuda.manual_seed_all(seed)
    # jt.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def intra_cls(logits, y, classes):
    y = y - classes
    logits1 = logits[:, classes:]
    return jt.nn.cross_entropy_loss(logits1, y, reduction='none')

def freeze_non_lora_params(model, keyword: str = "lora"):
    """
    冻结除 LoRA 相关参数以外的所有参数。
    通过参数名中是否包含 keyword 来判断是否为 LoRA 参数。
    """
    # pdb.set_trace()
    for name, p in model.named_parameters():
        if keyword not in name.lower():
            try:
                p.stop_grad()
            except Exception:
                try:
                    p.requires_grad = False
                except Exception:
                    pass
        else:
            try:
                p.start_grad()
            except Exception:
                try:
                    p.requires_grad = True
                except Exception:
                    pass


def run_class_incremental2(cfg, device):
    
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model = load_model(cfg, device)
    freeze_non_lora_params(model)

    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )
    train_dataset, _ = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    # pdb.set_trace()
    model.classes_names = classes_names
    if cfg.visual_clsf:
        if cfg.model_name == "ViT-L/14":
            vision_clsf = VisionClassifier(768, cfg.increment, activation=None)
        else:
            vision_clsf = VisionClassifier(512, cfg.increment, activation=None)
    

    acc_list = []
    metric_logger = MetricLogger()

    p1 = 0
    p2 = 0

    negative_records = 0
    # trainable_params = {k: v for k, v in model.named_parameters() if v.requires_grad}

    # pdb.set_trace()

    for task_id, _ in enumerate(eval_dataset):

        # negative_records = 0

        if task_id == 0:
            targets_bais = 0
        else:
            targets_bais = cfg.initial_increment + (task_id - 1) * cfg.increment
        
        logging.info(f"Evaluation for task {task_id} has started.")
        model.adaptation(task_id)
        # 将model的参数保存

        # 计算未经训练时正类别和负类别的输出平均值
        if task_id == 0:
            model.eval()  # 切换到评估模式
            positive_outputs = []
            negative_outputs = []

            # val_gap_loader = DataLoader(train_dataset[task_id], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
            val_gap_loader = MyImageFolder(data=train_dataset[task_id], transform=model.transforms).set_attrs(batch_size = 64, num_workers =cfg.num_workers, shuffle = True)
            with jt.no_grad():
                for inputs, targets, t in val_gap_loader:
                    # inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                    one_hot_targets = jt.nn.one_hot(targets, outputs.shape[1]).float()
                    positive_outputs.append((outputs * one_hot_targets).sum(dim=1).mean())
                    mask = 1 - one_hot_targets
                    negative_outputs.append(((outputs * mask).sum(dim=1) / mask.sum(dim=1)).mean())
            positive_mean = sum(positive_outputs) / len(positive_outputs)
            negative_mean = sum(negative_outputs) / len(negative_outputs)
            # if  task_id == 0:
            negative_records = negative_mean
            # if task_id == 0:
            logit_size = cfg.increment if task_id>0 else cfg.initial_increment
            bias_logit = jt.full((logit_size,), negative_mean)
            bias_logit[0] = positive_mean
            # pdb.set_trace()
            # pdb.set_trace()
            logging.info(f"positive_records: {positive_mean}")
            logging.info(f"negative_records: {negative_mean}")
            # pdb.set_trace()
        # pdb.set_trace()
        model.train()
        # train_loader = DataLoader(train_dataset[:task_id+1], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
        # train_loader = DataLoader(t_data, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
        train_loader = MyImageFolder(data=train_dataset[task_id], transform=model.transforms).set_attrs(batch_size = cfg.train_batch_size, num_workers =cfg.num_workers, shuffle = True)
        epochs = cfg.epochs

        if epochs>0:
            # filter out the parameters that require grad
            params = list(filter(lambda p: not p.is_stop_grad(), model.parameters()))
            optimizer = jt.optim.Adam(params, lr=cfg.lr) 
            # optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)  
            scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=cfg.lr*0.01)   
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name)
        for i_epoch in range(epochs):

            for bach_i, (inputs, targets, t) in enumerate(train_loader):
                loss_c = jt.Var(0.0).to(device)
                loss = jt.Var(0.0).to(device)

                # targets = targets - targets_bais
                # inputs, targets = inputs.to(device), targets.to(device)

                outputs =  model(inputs)
                # image_f, text_f = model(inputs, return_feature=True)
                if task_id >0:
                    loss_c = intra_cls(outputs,targets,targets_bais).mean()
                else:
                    loss_c = jt.nn.cross_entropy_loss(outputs, targets) 
                loss += loss_c
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                if bach_i % 10 == 0:
                    logging.info(f"Epoch {i_epoch + 1}/{epochs} | Batch {bach_i + 1}/{len(train_loader)} | Loss: {loss.item()} | Loss_c: {loss_c.item()}")
            scheduler.step()

                

        if cfg.visual_clsf:
            # pdb.set_trace()
            model.eval()
            e_num = cfg.visual_clsf_epochs
            vision_clsf_loader = MyImageFolder(data=train_dataset[task_id], transform=model.transforms).set_attrs(batch_size = cfg.train_batch_size, num_workers =cfg.num_workers, shuffle = True)
            features_dict = {}
            with jt.no_grad():
                for inputs, targets, t in vision_clsf_loader:
                    # inputs, targets = inputs.to(device), targets.to(device)
                    _, features, __ = model(inputs, test=True, return_feature=True)
                    for feature, target in zip(features, targets):
                        target = target.item()
                        if target not in features_dict:
                            features_dict[target] = []
                        features_dict[target].append(feature.cpu())
            mean_features = []
            for target in sorted(features_dict.keys()):
                features = jt.stack(features_dict[target])
                mean_feature = features.mean(dim=0)
                mean_features.append(mean_feature.unsqueeze(0))
            mean_features = jt.cat(mean_features)
            if task_id > 0:
                vision_clsf.add_weight(mean_features)
                pass
            else:
                vision_clsf.set_weight(mean_features)
                pass
            optimizer = jt.optim.Adam(vision_clsf.parameters(), lr=cfg.visual_clsf_lr)
            scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, e_num*len(vision_clsf_loader), eta_min=cfg.visual_clsf_lr*0.01)
            for e in range(e_num):
                bach_i = -1
                for inputs, targets, t in vision_clsf_loader:
                    # inputs, targets = inputs.to(device), targets.to(device)
                    # pdb.set_trace()
                    with jt.no_grad():
                        outputs, _ = model(inputs, return_feature=True)
                    # pdb.set_trace()
                    outputs = vision_clsf(outputs)
                    # pdb.set_trace()
                    loss = intra_cls(outputs,targets,targets_bais).mean()
                    # loss = F.cross_entropy(outputs, targets)
                    optimizer.zero_grad()
                    optimizer.backward(loss)
                    optimizer.step()
                    bach_i+=1
                    if bach_i % 10 == 0:
                        logging.info(f"Epoch {e + 1}/{e_num} | Batch {bach_i + 1}/{len(vision_clsf_loader)} | Loss: {loss.item()}")
                    scheduler.step()
            



        eval_loader = MyImageFolder(data=eval_dataset[:task_id + 1], transform=model.transforms).set_attrs(batch_size = cfg.batch_size, num_workers = cfg.num_workers)
        # eval_loader = DataLoader(eval_dataset[:10], batch_size=cfg.batch_size)
        image_feature_list = []
        targets_list = []
        model.eval()
        text_feature_list = []
        correct_per_class = defaultdict(int)
        total_per_class = defaultdict(int)
        for inputs, targets, task_ids in eval_loader:
            # inputs, targets = inputs.to(device), targets.to(device)
            
            with jt.no_grad():
                if cfg.visual_clsf:
                    a = 1
                    b = 4
                    
                    outputs, image_feature, text_feature  = model(inputs, test=True, return_feature=True)
                    vision_outputs = vision_clsf(image_feature)

                    outputs_softmax = jt.nn.softmax(outputs, dim=1)
                    vision_outputs_softmax = jt.nn.softmax(vision_outputs, dim=1)
                    
                    combined_outputs = (a*outputs_softmax + b*vision_outputs_softmax) / (a + b)
                    
                    metric_logger.add([combined_outputs.argmax(dim=1)[0], targets, task_ids], subset="test")
                    # preds = combined_outputs.cpu().argmax(dim=1)
                    # for l,p in zip(targets.cpu(), preds):
                    #     label = l.item()
                    #     total_per_class[label] += 1
                    #     if l == p:
                    #         correct_per_class[label] += 1
                else:
                    outputs = model(inputs, test=True, all_test=cfg.all_test)
                    metric_logger.add([outputs.argmax(dim=1)[0], targets, task_ids], subset="test")
        # class_acc = {}
        # for clas in total_per_class:
        #     acc = correct_per_class[clas] / total_per_class[clas]
        #     class_acc[clas] = acc
        # avg_acc = np.mean(list(class_acc.values()))
        metric_logger.end_task()


        acc_list.append(100 * metric_logger.accuracy)
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger.accuracy, 2),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
            }) + '\n')
            
    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')


def run_class_incremental(cfg, device):

    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model = load_model(cfg, device)
    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )

    train_dataset, _ = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    model.classes_names = classes_names
    acc_list = []
    metric_logger = MetricLogger()
    for task_id, _ in enumerate(eval_dataset):
        logging.info(f"Train for task {task_id} has started.")
        model.adaptation(task_id, threshold=cfg.threshold)
        # pdb.set_trace()
        # train_loader = DataLoader(train_dataset[task_id], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
        train_loader = MyImageFolder(data=train_dataset[task_id], transform=model.transforms).set_attrs(batch_size = cfg.train_batch_size, num_workers =cfg.num_workers, shuffle = True)
        # train_loader = ImageFolder(root="/defaultShare/pubdata/imagenet/val").set_attrs(batch_size = cfg.train_batch_size, num_workers = cfg.num_workers, shuffle = True)
        # epoch
        model.train()
        optimizer = jt.optim.Adam(model.adapter.parameters(), lr=cfg.lr, weight_decay=0.0000)

        milestones = cfg.milestones
        epochs = cfg.epochs

        scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        for i_epoch in range(epochs):
            loss = jt.Var(0.0).to(device)
            loss_c = jt.Var(0.0).to(device)
            loss_hinge = jt.Var(0.0).to(device)
            tqdm_loader = tqdm(train_loader)
            if task_id >0:
                random_class_order_list = list(range(cfg.initial_increment+(task_id-1)*cfg.increment))
                random.shuffle(random_class_order_list)
            batch_id = -1
            # pdb.set_trace()
            for inputs, targets, task_ids in tqdm_loader:
                batch_id += 1
                # inputs, targets = inputs.to(device), targets.to(device)
                sg_inputs = None
                edge_sample = None
                ori_targets = targets.clone()
                if task_id > 0:
                    sg_inputs = []
                    sg_targets = []
                    list_for_one_batch = [random_class_order_list[batch_id*2%len(random_class_order_list)], random_class_order_list[(batch_id*2+1)%len(random_class_order_list)]]
                    for i in list_for_one_batch:
                        sg_inputs.append(sample(model.class_mean_list[i], model.class_cov_list[i],int(10), shrink=cfg.shrinkage))
                        sg_targets.append(jt.ones(int(10))*i)
                    sg_inputs = jt.cat(sg_inputs, dim=0)
                    sg_targets = jt.cat(sg_targets, dim=0)
                    targets = jt.cat([targets, sg_targets], dim=0)
                if model.hard_pairs is not None and model.hard_pairs.shape[0] > 0:
                    edge_sample = []
                    edge_p_target = []
                    edge_n_target = []
                    for hard_pair in model.hard_pairs:
                        edge_sample.append(sample(model.class_mean_list[int(hard_pair[0])], model.class_cov_list[int(hard_pair[0])],int(20), shrink=cfg.shrinkage))
                        edge_p_target.append(jt.ones(int(20))*hard_pair[0])
                        edge_n_target.append(jt.ones(int(20))*hard_pair[1])
                    edge_sample = jt.cat(edge_sample, dim=0)
                    edge_p_target = jt.cat(edge_p_target, dim=0)
                    edge_n_target = jt.cat(edge_n_target, dim=0)
                if task_id > 0:
                    not_ini = True
                else:
                    not_ini = False
                outputs, _, __, edge_sample_features = model(inputs, memory_data=sg_inputs, not_ini=not_ini, edge_sample=edge_sample, prompt=False)



                if task_id>0:
                    if edge_sample is not None:
                        edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
                        edge_target_features = model.class_name_features[edge_p_target]
                        edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
                        edge_nearest_class_features = model.class_name_features[edge_n_target]
                        edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(dim=-1, keepdim=True)
                        loss_hinge = jt.nn.relu(- (edge_sample_features * edge_target_features.clone().detach()).sum(-1) + (edge_sample_features * edge_nearest_class_features.clone().detach()).sum(-1) + 0.1).mean()
                loss_c = jt.nn.cross_entropy_loss(outputs, targets.detach())
                if edge_sample is not None:
                    loss = loss_c + loss_hinge
                else:
                    loss = loss_c 
                optimizer.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                tqdm_loader.set_description(f"Epoch {i_epoch + 1}/{cfg.epochs} | Loss: {loss.item():.4f} | Loss_c: {loss_c.item():.4f}| loss_hinge: {loss_hinge.item():.4f}")
                # if batch_id % 10 == 0:
                #     print(f"Epoch {i_epoch + 1}/{cfg.epochs} | batch:{} | Loss: {loss.item():.4f} | Loss_c: {loss_c.item():.4f}| loss_hinge: {loss_hinge.item():.4f} ")
            scheduler.step()
        # sample_loader = DataLoader(train_dataset[task_id], batch_size=128, shuffle=False, num_workers=cfg.num_workers)
        
        sample_loader = MyImageFolder(data=train_dataset[task_id], transform=model.transforms).set_attrs(batch_size = 128, num_workers = cfg.num_workers, shuffle = False)
        sample_data = []
        sample_target = []
        sample_after_adapt_feature = []
        print('analyze')
        for input, target, task_ids in tqdm(sample_loader):
            input, target = input.to(device), target.to(device)
            with jt.no_grad():
                _, ori_ima_feat, after_adapt_feature = model(input, ori_ima_f=True)
            sample_data.append(ori_ima_feat)
            sample_target.append(target)
            sample_after_adapt_feature.append(after_adapt_feature)
        sample_target = jt.cat(sample_target, dim=0)
        sample_data = jt.cat(sample_data, dim=0)
        sample_after_adapt_feature = jt.cat(sample_after_adapt_feature, dim=0)
        model.analyze_mean_cov(sample_data, sample_target)
        model.mix_matrix()
        model.eval()
        # eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        eval_loader = MyImageFolder(data=eval_dataset[:task_id + 1], transform=model.transforms).set_attrs(batch_size = cfg.batch_size, num_workers = cfg.num_workers)
        for inputs, targets, task_ids in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with jt.no_grad():
                outputs, _, __, ___ = model(inputs)
                jt.nn.softmax(outputs, dim=-1)
            metric_logger.add([outputs.argmax(dim=1)[0].numpy(), targets.numpy(), task_ids], subset="test")
        metric_logger.end_task()
        acc_list.append(100 * metric_logger.accuracy)
        
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger.accuracy, 2),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
            }) + '\n')


    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')





@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    with open(cfg.log_path, 'w+') as f: 
        pass
    # device = jt.device("cuda" if jt.cuda.is_available() else "cpu")
    device = "cuda"

    if cfg.scenario == "class":
        if cfg.method == "RAPF":
            run_class_incremental(cfg, device)
        elif cfg.method == "MG-CLIP":
            run_class_incremental2(cfg, device)
            



    
        

















if __name__ == "__main__":
    jt.flags.use_cuda = 1

    continual_clip()