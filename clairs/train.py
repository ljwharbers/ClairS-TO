# BSD 3-Clause License
#
# Copyright 2023 The University of Hong Kong, Department of Computer Science
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
import tables
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset

import threading
import time
from tqdm import tqdm
from subprocess import run
from argparse import ArgumentParser, SUPPRESS
from torch.utils.tensorboard import SummaryWriter
import csv
from shared.utils import str2bool
import shared.param as param
import clairs.model as model_path
from torch.utils.data import random_split, DataLoader
logging.basicConfig(format='%(message)s', level=logging.INFO)

check_gpu_status = True
try:
    import gpustat
except ModuleNotFoundError:
    check_gpu_status = False

tables.set_blosc_max_threads(512)
os.environ['NUMEXPR_MAX_THREADS'] = '256'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
gamma = 0.7
seed = 100
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class BinFileDataset(Dataset):
    def __init__(self, file_list, file_path, chunk_size, batch_size, debug_mode=False, discard_germline=False, add_af_in_label=False, smoothing=None, pileup=True, train_indel=False):
        ### Configurations
        self.debug_mode = debug_mode
        self.discard_germline = discard_germline
        self.file_list = file_list
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.add_af_in_label = add_af_in_label
        self.random_start_position = None
        self.train_flag = True

        ### Dataset Initialization
        self.table_dataset_list, self.chunk_offset = self._populate_dataset_table(file_list, file_path)
        self.cum_sum = np.cumsum(self.chunk_offset)
        self.total_chunks = sum(self.chunk_offset)
        self.pileup = pileup
        self.train_indel = train_indel
        ### Smoothing
        self.positive = 1 - smoothing if smoothing is not None else 1
        self.negative = smoothing if smoothing is not None else 0

    def _populate_dataset_table(self, file_list, file_path):
        chunk_offset = np.zeros(len(file_list), dtype=int)
        table_dataset_list = []
        for bin_idx, bin_file in enumerate(file_list):
            table_dataset = tables.open_file(os.path.join(file_path, bin_file), 'r')
            table_dataset_list.append(table_dataset)
            chunk_num = (len(table_dataset.root.position) - self.batch_size) // self.chunk_size
            chunk_offset[bin_idx] = chunk_num
        return table_dataset_list, chunk_offset

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        bin_idx, chunk_idx = self._get_file_and_chunk_index(idx)
        start_idx = chunk_idx * self.chunk_size
        start_idx += self.random_start_position if self.random_start_position is not None and self.train_flag else 0
        end_idx = start_idx + self.chunk_size
        num_rows = len(self.table_dataset_list[bin_idx].root.input_matrix)
        if end_idx > num_rows:
            end_idx = num_rows
        if start_idx >= num_rows:
            start_idx = num_rows - 1
        assert end_idx <= num_rows, f"Index out of range: {end_idx} > {num_rows}"
        current_tensor = self.table_dataset_list[bin_idx].root.input_matrix[start_idx:end_idx]

        a_label = self.table_dataset_list[bin_idx].root.a_label[start_idx:end_idx, :2]
        c_label = self.table_dataset_list[bin_idx].root.c_label[start_idx:end_idx, :2]
        g_label = self.table_dataset_list[bin_idx].root.g_label[start_idx:end_idx, :2]
        t_label = self.table_dataset_list[bin_idx].root.t_label[start_idx:end_idx, :2]
        na_label = self.table_dataset_list[bin_idx].root.na_label[start_idx:end_idx, :2]
        nc_label = self.table_dataset_list[bin_idx].root.nc_label[start_idx:end_idx, :2]
        ng_label = self.table_dataset_list[bin_idx].root.ng_label[start_idx:end_idx, :2]
        nt_label = self.table_dataset_list[bin_idx].root.nt_label[start_idx:end_idx, :2]
        if self.train_indel:
            i_label = self.table_dataset_list[bin_idx].root.i_label[start_idx:end_idx, :2]
            d_label = self.table_dataset_list[bin_idx].root.d_label[start_idx:end_idx, :2]
            ni_label = self.table_dataset_list[bin_idx].root.ni_label[start_idx:end_idx, :2]
            nd_label = self.table_dataset_list[bin_idx].root.nd_label[start_idx:end_idx, :2]


        a_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                             in a_label]
        c_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                             in c_label]
        g_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                             in g_label]
        t_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                             in t_label]
        na_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                             in na_label]
        nc_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                             in nc_label]
        ng_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                             in ng_label]
        nt_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                             in nt_label]
        if self.train_indel:
            i_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                in i_label]
            d_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                in d_label]
            ni_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                                 in ni_label]
            nd_label_for_tumor = [[self.negative, self.positive] if np.argmax(item[:2]) == 1 else [self.positive, self.negative] for item
                                 in nd_label]

        a_label_for_tumor = np.array(a_label_for_tumor, dtype=np.float32)
        c_label_for_tumor = np.array(c_label_for_tumor, dtype=np.float32)
        g_label_for_tumor = np.array(g_label_for_tumor, dtype=np.float32)
        t_label_for_tumor = np.array(t_label_for_tumor, dtype=np.float32)
        na_label_for_tumor = np.array(na_label_for_tumor, dtype=np.float32)
        nc_label_for_tumor = np.array(nc_label_for_tumor, dtype=np.float32)
        ng_label_for_tumor = np.array(ng_label_for_tumor, dtype=np.float32)
        nt_label_for_tumor = np.array(nt_label_for_tumor, dtype=np.float32)
        if self.train_indel:
            i_label_for_tumor = np.array(i_label_for_tumor, dtype=np.float32)
            d_label_for_tumor = np.array(d_label_for_tumor, dtype=np.float32)
            ni_label_for_tumor = np.array(ni_label_for_tumor, dtype=np.float32)
            nd_label_for_tumor = np.array(nd_label_for_tumor, dtype=np.float32)

        if not self.pileup:
            current_tensor = np.transpose(current_tensor, (0, 3, 1, 2))

        if not self.train_indel:
            return current_tensor, a_label_for_tumor, c_label_for_tumor, g_label_for_tumor, t_label_for_tumor, na_label_for_tumor, nc_label_for_tumor, ng_label_for_tumor, nt_label_for_tumor
        else:
            return current_tensor, a_label_for_tumor, c_label_for_tumor, g_label_for_tumor, t_label_for_tumor, i_label_for_tumor, d_label_for_tumor, na_label_for_tumor, nc_label_for_tumor, ng_label_for_tumor, nt_label_for_tumor, ni_label_for_tumor, nd_label_for_tumor

    def _get_file_and_chunk_index(self, idx):
        file_idx = np.searchsorted(self.cum_sum, idx, side='right')
        if file_idx > 0:
            chunk_idx = idx - self.cum_sum[file_idx - 1]
        else:
            chunk_idx = idx
        return file_idx, chunk_idx

    def close(self):
        for dataset in self.table_dataset_list:
            dataset.close()

class GPU_Monitor(threading.Thread):
    def __init__(self, gpu_id, interval=5, duration=60, csv_file='gpu_usage.csv'):
        super().__init__()
        self.gpu_id = gpu_id  # The ID of the GPU to monitor
        self.interval = interval  # The time interval (in seconds) between measurements
        self.duration = duration  # The total duration (in seconds) for which to run the monitoring
        self.csv_file = csv_file  # The filename for the CSV output
        self.running = True  # A flag to control the running of the thread
        self.daemon = True  # Set the thread as a daemon so it exits when the main program does

    def run(self):
        print('Monitoring GPU {} for {} seconds'.format(self.gpu_id, self.duration) )
        start_time = time.time()
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the CSV header
            writer.writerow(['Timestamp', 'GPU ID', 'GPU Utilization'])

            while self.running and ((time.time() - start_time) < self.duration):
                stats = gpustat.new_query()  # Get the current GPU statistics
                gpu = stats.gpus[self.gpu_id]  # Only query the specified GPU
                # Record the current time, GPU ID, and GPU utilization
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), gpu.index, gpu.utilization])
                time.sleep(self.interval)  # Wait for the next interval
        print('Monitoring complete')
    def stop(self):
        self.running = False


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        y_pred, y_true = input, target
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_pred = torch.clamp(y_pred, min=1e-9, max=1 - 1e-9)
        cross_entropy = -y_true * torch.log(y_pred)
        weight = ((1 - y_pred) ** self.gamma) * y_true
        FCLoss = cross_entropy * weight

        reduce_fl = torch.mean(torch.sum(FCLoss, dim=1))
        return reduce_fl


class AFLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(AFLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        y_pred, y_true = input, target
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_pred = torch.clamp(y_pred, min=1e-9, max=1 - 1e-9)

        cross_entropy = -y_true * torch.log(y_pred)
        reduce_fl = torch.mean(torch.sum(cross_entropy, dim=1))

        return reduce_fl


def cal_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return round(precision, 6), round(recall, 6), round(f1_score, 6)


def get_label_task(label, label_shape_cum, task):
    if task == 0:
        return label[:label_shape_cum[task]]
    elif task == len(label_shape_cum) - 1:
        return label[label_shape_cum[task - 1]:]
    else:
        return label[label_shape_cum[task - 1]:label_shape_cum[task]]


def cal_class_weight(samples_per_cls, no_of_classes, beta=0.999):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    cls_weights = (1.0 - beta) / np.array(effective_num)
    cls_weights = cls_weights / np.sum(cls_weights) * no_of_classes
    return cls_weights


def get_chunk_list(chunk_offset, train_chunk_num, chunks_per_batch=10, training_dataset_percentage=None):
    need_split_validation_data = training_dataset_percentage is not None
    all_shuffle_chunk_list = []
    training_chunk_list, validation_chunk_list = [], []
    for bin_idx, chunk_num in enumerate(chunk_offset):
        current_chunk_list = [(bin_idx, chunk_idx) for chunk_idx in range(chunk_num)]
        all_shuffle_chunk_list += current_chunk_list
        if need_split_validation_data:
            buffer_chunk_num = chunks_per_batch
            if chunk_num < buffer_chunk_num:
                training_chunk_list += [(bin_idx, chunk_idx) for chunk_idx in range(chunk_num)]
                continue

            training_chunk_num = int((chunk_num - buffer_chunk_num) * training_dataset_percentage)
            validation_chunk_num = int(chunk_num - buffer_chunk_num - training_chunk_num)
            if training_chunk_num > 0:
                training_chunk_list += current_chunk_list[:training_chunk_num]
            if validation_chunk_num > 0:
                validation_chunk_list += current_chunk_list[-validation_chunk_num:]

    if need_split_validation_data:
        return np.array(training_chunk_list), np.array(validation_chunk_list)

    return np.array(all_shuffle_chunk_list[:train_chunk_num]), np.array(all_shuffle_chunk_list[train_chunk_num:])


def exist_file_prefix(exclude_training_samples, f):
    for prefix in exclude_training_samples:
        if prefix in f:
            return True
    return False


def pass_chr(fn, ctg_name_list):
    if ctg_name_list is None or len(ctg_name_list) == 0:
        return True
    for ctg_name in ctg_name_list:
        if ctg_name + '.' in fn:
            return True
    return False


def train_model(args):
    apply_focal_loss = param.apply_focal_loss
    discard_germline = param.discard_germline
    add_l2_regulation_loss = param.add_l2_regulation_loss
    l2_regularization_lambda = param.l2_regularization_lambda
    debug_mode = args.debug_mode
    platform = args.platform
    ctg_name_string = args.ctg_name
    chkpnt_fn = args.chkpnt_fn
    ochk_prefix = args.ochk_prefix
    add_writer = args.add_writer
    smoothing = args.smoothing
    phase_tumor = args.phase_tumor and args.platform != 'ilmn'

    add_validation_dataset = args.random_validation or (args.validation_fn is not None)
    validation_fn = args.validation_fn
    ctg_name_list = ctg_name_string.split(',') if ctg_name_string is not None else []
    exclude_training_samples = args.exclude_training_samples
    exclude_training_samples = set(exclude_training_samples.split(',')) if exclude_training_samples else set()

    if ochk_prefix and not os.path.exists(ochk_prefix):
        output = run('mkdir -p {}'.format(ochk_prefix), shell=True)
        print("[INFO] Model path empty, create folder:{}".format(ochk_prefix))

    if add_writer:
        writer = SummaryWriter("{}/log".format(ochk_prefix))
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    apply_softmax = False if apply_focal_loss else True
    if args.train_aff:
        if args.pileup:
            channel_size = param.pileup_channel_size
            tumor_channel_size = param.tumor_channel_size if phase_tumor else channel_size
            pileup_tensor_shape = [param.no_of_positions, channel_size]  # only tumor
            tensor_shape = pileup_tensor_shape
            model_acgt = model_path.CvT(
                num_classes=2,
                s1_emb_dim=16,  # stage 1 - dimension
                s1_emb_kernel=3,  # stage 1 - conv kernel
                s1_emb_stride=2,  # stage 1 - conv stride
                s1_proj_kernel=3,  # stage 1 - attention ds-conv kernel size
                s1_kv_proj_stride=2,  # stage 1 - attention key / value projection stride
                s1_heads=1,  # stage 1 - heads
                s1_depth=1,  # stage 1 - depth
                s1_mlp_mult=4,  # stage 1 - feedforward expansion factor
                s2_emb_dim=64,  # stage 2 - (same as above)
                s2_emb_kernel=3,
                s2_emb_stride=2,
                s2_proj_kernel=3,
                s2_kv_proj_stride=2,
                s2_heads=3,
                s2_depth=2,
                s2_mlp_mult=4,
                s3_emb_dim=128,  # stage 3 - (same as above)
                s3_emb_kernel=3,
                s3_emb_stride=2,
                s3_proj_kernel=3,
                s3_kv_proj_stride=2,
                s3_heads=4,
                s3_depth=3,
                s3_mlp_mult=4,
                dropout=0.,
                dropout_fc=0.3,
                depth=1,
                width=param.no_of_positions,
                dim=param.pileup_channel_size,
                apply_softmax=False,
                model_type="acgt"
            ).to(device)
    
        if chkpnt_fn is not None:
            model = torch.load(chkpnt_fn, map_location=torch.device(device))
            model_acgt = model['model_acgt']
    
        batch_size, chunk_size = param.trainBatchSize, param.chunk_size
        assert batch_size % chunk_size == 0
        chunks_per_batch = batch_size // chunk_size
        random.seed(param.RANDOM_SEED)
        np.random.seed(param.RANDOM_SEED)
        learning_rate = args.learning_rate if args.learning_rate else param.initialLearningRate
        max_epoch = args.max_epoch if args.max_epoch else param.maxEpoch
        bin_list = os.listdir(args.bin_fn)
    
        bin_list = [f for f in bin_list if
                    pass_chr(f, ctg_name_list) and not exist_file_prefix(exclude_training_samples, f)]
        failed_bin_set = set()
        for bin_file in bin_list:
            try:
                table = tables.open_file(os.path.join(args.bin_fn, bin_file), 'r')
                table.close()
            except:
                print("[WARNING] {} cannot open!".format(bin_file))
                failed_bin_set.add(bin_file)
        bin_list = [f for f in bin_list if f not in failed_bin_set]
        if len(bin_list) == 0:
            print("[ERROR] Cannot find ant binary for model training")
            return
    
        logging.info("[INFO] total {} training bin files: {}".format(len(bin_list), ','.join(bin_list)))
    
        if validation_fn:
            val_list = os.listdir(validation_fn)
            logging.info("[INFO] total {} validation bin files: {}".format(len(val_list), ','.join(val_list)))
            train_dataset = BinFileDataset(bin_list, args.bin_fn, chunk_size, batch_size, debug_mode=False, discard_germline = discard_germline, \
                                           add_af_in_label=param.add_af_in_label, smoothing=smoothing, pileup=args.pileup, train_indel=args.train_indel)
            train_chunk_num = len(train_dataset)
    
            val_dataset = BinFileDataset(val_list, validation_fn, chunk_size, batch_size, debug_mode=debug_mode, discard_germline=discard_germline, \
                                         add_af_in_label=False, smoothing=smoothing, pileup=args.pileup, train_indel=args.train_indel)
            validate_chunk_num = len(val_dataset)
            total_chunks = train_chunk_num + validate_chunk_num
        else:
            total_dataset = BinFileDataset(bin_list, args.bin_fn, chunk_size, batch_size, debug_mode=debug_mode, discard_germline=discard_germline, \
                                           add_af_in_label=param.add_af_in_label, smoothing=smoothing, pileup=args.pileup, train_indel=args.train_indel)
            total_chunks = len(total_dataset)
            training_dataset_percentage = param.trainingDatasetPercentage if add_validation_dataset else None
            if add_validation_dataset:
                total_batches = total_chunks // chunks_per_batch
                validate_chunk_num = int(
                    max(1., np.floor(total_batches * (1 - training_dataset_percentage))) * chunks_per_batch)
                train_chunk_num = int(total_chunks - validate_chunk_num)
    
                train_indices = list(range(train_chunk_num))
                val_indices = list(range(train_chunk_num, train_chunk_num + validate_chunk_num))
    
                train_dataset = Subset(total_dataset, train_indices)
                val_dataset = Subset(total_dataset, val_indices)
    
                #set the training dataset to:no debug mode
                train_dataset.dataset.debug_mode = False
                val_dataset.dataset.add_af_in_label = False
            else:
                train_chunk_num = total_chunks
                train_dataset = total_dataset
                #set the training dataset to:no debug mode
                train_dataset.dataset.debug_mode = False
    
            train_chunk_num = len(train_dataset)
            validate_chunk_num = len(val_dataset) if add_validation_dataset else 0
        train_data_size = train_chunk_num * chunk_size
        validate_data_size = validate_chunk_num * chunk_size
    
        if args.pileup:
            try:
                from torchinfo import summary
                print(summary(model_acgt, input_size=tuple([100] + tensor_shape), device=device))
            except:
                pass
    
        if add_validation_dataset:
            train_dataloader = DataLoader(train_dataset, batch_size=chunks_per_batch, shuffle=True, num_workers=args.torch_dataset_num_workers, prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)
            validate_dataloader = DataLoader(val_dataset, batch_size=chunks_per_batch, shuffle=False, num_workers=args.torch_dataset_num_workers, prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=chunks_per_batch, shuffle=True, num_workers=args.torch_dataset_num_workers, prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)
    
        criterion = FocalLoss() if apply_focal_loss else nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        if param.add_af_in_label:
            af_loss = AFLoss().to(device)
    
        optimizer = optim.Adam([{'params': model_acgt.parameters(), 'lr': learning_rate, 'weight_decay': param.weight_decay}])
    
        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
        train_steps = train_data_size // batch_size
        validate_steps = validate_data_size // batch_size
        print("[INFO] Using GPU for model training: {}".format(True if device == 'cuda' else False))
        print("[INFO] The size of dataset: {}".format(train_data_size))
        print("[INFO] The training batch size: {}".format(batch_size))
        print("[INFO] The training learning_rate: {}".format(learning_rate))
        print("[INFO] The output model folder: {}".format(ochk_prefix))
        print("[INFO] Apply focal loss in training: {}".format(apply_focal_loss))
        print("[INFO] Discard germline in training: {}".format(discard_germline))
        print("[INFO] Add L2 regularization to model parameters: {}".format(add_l2_regulation_loss))
        print('[INFO] Train steps:{}'.format(train_steps))
        print('[INFO] Validate steps:{}'.format(validate_steps))
    
        training_loss, validation_loss = 0.0, 0.0
        training_acgt_loss, validation_acgt_loss = 0.0, 0.0
        training_step, validation_step = 0, 0
        val_best_acgt_f1 = 0
        val_best_acgt_epoch = 0
        echo_each_step = 200
    
        if args.gpu_csv_fn is not None and check_gpu_status:
            monitor = GPU_Monitor(gpu_id=0, interval=1, duration=120, csv_file = args.gpu_csv_fn)  # Monitor for 5 minutes
            monitor.start()
    
        for epoch in range(1, max_epoch + 1):
            epoch_loss = 0
            epoch_acgt_loss = 0
            a_fp, a_tp, a_fn = 0, 0, 0
            c_fp, c_tp, c_fn = 0, 0, 0
            g_fp, g_tp, g_fn = 0, 0, 0
            t_fp, t_tp, t_fn = 0, 0, 0
    
            #set a random start position for each epoch training
            np.random.seed(epoch)
            train_dataset.dataset.random_start_position = np.random.randint(0, chunk_size)
            train_dataset.dataset.train_flag = True
            t = tqdm(enumerate(train_dataloader), total=train_steps, position=0, leave=True)
            v = tqdm(enumerate(validate_dataloader), total=validate_steps, position=0,
                     leave=True) if not debug_mode else enumerate(validate_dataloader)
    
            model_acgt.train()
            for batch_idx, (data, a_label, c_label, g_label, t_label, na_label, nc_label, ng_label, nt_label) in t:
                t.set_description('EPOCH {}'.format(epoch))
                if not args.pileup:
                    data = data.reshape(-1, param.channel_size, param.tumor_matrix_depth_dict[platform], param.no_of_positions)
                    data = data.to(device) / 100.0
                else:
                    data = data.reshape(-1, param.no_of_positions, param.pileup_channel_size)
                    data = data.to(device)
                # data = data.to(device)
                a_label, c_label, g_label, t_label, na_label, nc_label, ng_label, nt_label = \
                    a_label.reshape(-1, 2).to(device), c_label.reshape(-1, 2).to(device), g_label.reshape(-1, 2).to(device), t_label.reshape(-1, 2).to(device),\
                        na_label.reshape(-1, 2).to(device), nc_label.reshape(-1, 2).to(device), ng_label.reshape(-1, 2).to(device), nt_label.reshape(-1, 2).to(device)
    
                a_output_logit, c_output_logit, g_output_logit, t_output_logit = model_acgt(data)
                a_output_logit, c_output_logit, g_output_logit, t_output_logit = \
                    a_output_logit.contiguous(), c_output_logit.contiguous(), g_output_logit.contiguous(), t_output_logit.contiguous()
                y_a_truth = torch.argmax(a_label, axis=1)
                y_c_truth = torch.argmax(c_label, axis=1)
                y_g_truth = torch.argmax(g_label, axis=1)
                y_t_truth = torch.argmax(t_label, axis=1)
    
                optimizer.zero_grad()
    
                a_loss = criterion(input=a_output_logit, target=a_label) if apply_focal_loss else criterion(a_output_logit,
                                                                                                            y_a_truth)
                c_loss = criterion(input=c_output_logit, target=c_label) if apply_focal_loss else criterion(c_output_logit,
                                                                                                            y_c_truth)
                g_loss = criterion(input=g_output_logit, target=g_label) if apply_focal_loss else criterion(g_output_logit,
                                                                                                            y_g_truth)
                t_loss = criterion(input=t_output_logit, target=t_label) if apply_focal_loss else criterion(t_output_logit,
                                                                                                            y_t_truth)
    
                l2_regularization_acgt_loss = sum([l2_regularization_lambda * 0.5 * params.norm(2) ** 2 for params in
                                              model_acgt.parameters()]) if add_l2_regulation_loss else 0.0
    
                loss_acgt = a_loss + c_loss + g_loss + t_loss + l2_regularization_acgt_loss
    
                loss = loss_acgt
    
                loss.backward()
    
                optimizer.step()
    
                training_step += 1
                training_loss += loss.item()
                training_acgt_loss += loss_acgt.item()
                if add_writer:
                    if training_step % echo_each_step == echo_each_step - 1:
                        writer.add_scalar('training acgt loss', training_acgt_loss / echo_each_step, training_step)
                    training_loss = 0.0
                    training_acgt_loss = 0.0
    
                y_a_truth = y_a_truth.cpu().numpy()
                y_c_truth = y_c_truth.cpu().numpy()
                y_g_truth = y_g_truth.cpu().numpy()
                y_t_truth = y_t_truth.cpu().numpy()
    
                y_a_pred = a_output_logit.argmax(dim=1).cpu().numpy()
                y_c_pred = c_output_logit.argmax(dim=1).cpu().numpy()
                y_g_pred = g_output_logit.argmax(dim=1).cpu().numpy()
                y_t_pred = t_output_logit.argmax(dim=1).cpu().numpy()
    
                arg_index = 1
    
                a_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                a_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                a_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                c_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                c_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                c_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                g_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                g_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                g_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                t_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                t_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                t_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
    
                acgt_fp = a_fp + c_fp + g_fp + t_fp
                acgt_fn = a_fn + c_fn + g_fn + t_fn
                acgt_tp = a_tp + c_tp + g_tp + t_tp
    
                if batch_idx + 1 == train_steps:
                    break
    
                epoch_loss += loss
                epoch_acgt_loss += loss_acgt
                el_acgt = epoch_acgt_loss.detach().cpu().numpy()
                acgt_precision, acgt_recall, acgt_f1_score = cal_metrics(acgt_tp, acgt_fp, acgt_fn)
                t.set_postfix(
                    {'acgt loss': el_acgt, 'acgt precision': acgt_precision, 'acgt recall': acgt_recall,
                     'acgt f1 score': acgt_f1_score})
    
                t.update(1)
    
            # validation
            val_a_fp, val_a_tp, val_a_fn = 0, 0, 0
            val_c_fp, val_c_tp, val_c_fn = 0, 0, 0
            val_g_fp, val_g_tp, val_g_fn = 0, 0, 0
            val_t_fp, val_t_tp, val_t_fn = 0, 0, 0
            val_epoch_loss = 0
            val_epoch_acgt_loss = 0
            model_acgt.eval()
            for batch_idx, (data, a_label, c_label, g_label, t_label, na_label, nc_label, ng_label, nt_label) in v:
                if not debug_mode:
                    v.set_description('VAL EPOCH {}'.format(epoch))
                if not args.pileup:
                    data = data.reshape(-1, param.channel_size, param.tumor_matrix_depth_dict[platform], param.no_of_positions)
                    data = data.to(device) / 100.0
                else:
                    data = data.reshape(-1, param.no_of_positions, param.pileup_channel_size)
                    data = data.to(device)
                # data = data.to(device)
                a_label, c_label, g_label, t_label, na_label, nc_label, ng_label, nt_label = \
                    a_label.reshape(-1, 2).to(device), c_label.reshape(-1, 2).to(device), g_label.reshape(-1, 2).to(device), t_label.reshape(-1, 2).to(device),\
                        na_label.reshape(-1, 2).to(device), nc_label.reshape(-1, 2).to(device), ng_label.reshape(-1, 2).to(device), nt_label.reshape(-1, 2).to(device)
    
                with torch.no_grad():
                    a_output_logit, c_output_logit, g_output_logit, t_output_logit = model_acgt(data)
    
                y_a_truth = torch.argmax(a_label, axis=1)
                y_c_truth = torch.argmax(c_label, axis=1)
                y_g_truth = torch.argmax(g_label, axis=1)
                y_t_truth = torch.argmax(t_label, axis=1)
    
                optimizer.zero_grad()
    
                a_loss = criterion(input=a_output_logit, target=a_label) if apply_focal_loss else criterion(a_output_logit,
                                                                                                            y_a_truth)
                c_loss = criterion(input=c_output_logit, target=c_label) if apply_focal_loss else criterion(c_output_logit,
                                                                                                            y_c_truth)
                g_loss = criterion(input=g_output_logit, target=g_label) if apply_focal_loss else criterion(g_output_logit,
                                                                                                            y_g_truth)
                t_loss = criterion(input=t_output_logit, target=t_label) if apply_focal_loss else criterion(t_output_logit,
                                                                                                            y_t_truth)
                loss_acgt = a_loss + c_loss + g_loss + t_loss
    
                loss = loss_acgt
    
                validation_loss += loss.item()
                validation_acgt_loss += loss_acgt.item()
                if add_writer:
                    if validation_step % echo_each_step == echo_each_step - 1:
                        writer.add_scalar('validation acgt loss', validation_acgt_loss / echo_each_step, validation_step)
                    validation_loss = 0.0
                    validation_acgt_loss = 0.0
    
                y_a_truth = y_a_truth.cpu().numpy()
                y_c_truth = y_c_truth.cpu().numpy()
                y_g_truth = y_g_truth.cpu().numpy()
                y_t_truth = y_t_truth.cpu().numpy()
    
                y_a_pred = a_output_logit.argmax(dim=1).cpu().numpy()
                y_c_pred = c_output_logit.argmax(dim=1).cpu().numpy()
                y_g_pred = g_output_logit.argmax(dim=1).cpu().numpy()
                y_t_pred = t_output_logit.argmax(dim=1).cpu().numpy()
    
                a_output_pro = torch.softmax(a_output_logit, dim=1)
                c_output_pro = torch.softmax(c_output_logit, dim=1)
                g_output_pro = torch.softmax(g_output_logit, dim=1)
                t_output_pro = torch.softmax(t_output_logit, dim=1)
    
                arg_index = 1
                if debug_mode:
                    if device == 'cuda':
                        data_numpy = data.cpu().numpy()
                    else:
                        data_numpy = data.numpy()
                    a_cpu_logit = a_output_pro.cpu().numpy()
                    c_cpu_logit = c_output_pro.cpu().numpy()
                    g_cpu_logit = g_output_pro.cpu().numpy()
                    t_cpu_logit = t_output_pro.cpu().numpy()
                    for idx, (x, y) in enumerate(zip(y_a_truth, y_a_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'A_FN', x, y, a_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'A_FP', x, y, a_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_c_truth, y_c_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'C_FN', x, y, c_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'C_FP', x, y, c_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_g_truth, y_g_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'G_FN', x, y, g_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'G_FP', x, y, g_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_t_truth, y_t_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'T_FN', x, y, t_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'T_FP', x, y, t_cpu_logit[idx])
    
                val_a_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                val_a_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                val_a_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                val_c_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                val_c_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                val_c_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                val_g_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                val_g_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                val_g_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                val_t_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                val_t_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                val_t_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
    
                val_acgt_fp = val_a_fp + val_c_fp + val_g_fp + val_t_fp
                val_acgt_fn = val_a_fn + val_c_fn + val_g_fn + val_t_fn
                val_acgt_tp = val_a_tp + val_c_tp + val_g_tp + val_t_tp
    
                if batch_idx + 1 == validate_steps:
                    break
    
                val_epoch_loss += loss
                val_epoch_acgt_loss += loss_acgt
                el_acgt = val_epoch_acgt_loss.detach().cpu().numpy()
                val_acgt_precision, val_acgt_recall, val_acgt_f1_score = cal_metrics(val_acgt_tp, val_acgt_fp, val_acgt_fn)
                if not debug_mode:
                    v.set_postfix(
                        {'val acgt loss': el_acgt, 'val acgt precision': val_acgt_precision,
                         'val acgt recall': val_acgt_recall,
                         'val acgt f1 score': val_acgt_f1_score})
    
                    v.update(1)
    
            # leanrning rate decay in each epoch end
            lr_scheduler.step()
            save_path = os.path.join(ochk_prefix, "{}.pkl".format(epoch)) if ochk_prefix is not None else "{}.pkl".format(
                epoch)
            print(save_path)
            torch.save({'model_acgt': model_acgt}, save_path)
    
            if val_acgt_f1_score > val_best_acgt_f1:
                save_path = os.path.join(ochk_prefix, "best_acgt_f1.pkl") if ochk_prefix is not None else "best_acgt_f1.pkl"
                print(epoch)
                print(save_path)
                torch.save({'model_acgt': model_acgt}, save_path)
                val_best_acgt_f1 = val_acgt_f1_score
                val_best_acgt_epoch = epoch
    
        print(("[INFO] Val best acgt f1 score: {}".format(val_best_acgt_f1)))
        print(("[INFO] Val best acgt epoch: {}".format(val_best_acgt_epoch)))

    if args.train_neg:
        if args.pileup:
            channel_size = param.pileup_channel_size
            tumor_channel_size = param.tumor_channel_size if phase_tumor else channel_size
            pileup_tensor_shape = [param.no_of_positions, channel_size]  # only tumor
            tensor_shape = pileup_tensor_shape
            model_nacgt = model_path.BiGRU_NACGT(apply_softmax=apply_softmax,
                                                 num_classes=2,
                                                 channel_size=tensor_shape[1],
                                                 model_type="nacgt").to(device)

        if chkpnt_fn is not None:
            model = torch.load(chkpnt_fn, map_location=torch.device(device))
            model_nacgt = model['model_nacgt']

        batch_size, chunk_size = param.trainBatchSize, param.chunk_size
        assert batch_size % chunk_size == 0
        chunks_per_batch = batch_size // chunk_size
        random.seed(param.RANDOM_SEED)
        np.random.seed(param.RANDOM_SEED)
        learning_rate = args.learning_rate if args.learning_rate else param.initialLearningRate
        max_epoch = args.max_epoch if args.max_epoch else param.maxEpoch
        bin_list = os.listdir(args.bin_fn)

        bin_list = [f for f in bin_list if
                    pass_chr(f, ctg_name_list) and not exist_file_prefix(exclude_training_samples, f)]
        failed_bin_set = set()
        for bin_file in bin_list:
            try:
                table = tables.open_file(os.path.join(args.bin_fn, bin_file), 'r')
                table.close()
            except:
                print("[WARNING] {} cannot open!".format(bin_file))
                failed_bin_set.add(bin_file)
        bin_list = [f for f in bin_list if f not in failed_bin_set]
        if len(bin_list) == 0:
            print("[ERROR] Cannot find ant binary for model training")
            return

        logging.info("[INFO] total {} training bin files: {}".format(len(bin_list), ','.join(bin_list)))

        if validation_fn:
            val_list = os.listdir(validation_fn)
            logging.info("[INFO] total {} validation bin files: {}".format(len(val_list), ','.join(val_list)))
            train_dataset = BinFileDataset(bin_list, args.bin_fn, chunk_size, batch_size, debug_mode=False,
                                           discard_germline=discard_germline, \
                                           add_af_in_label=param.add_af_in_label, smoothing=smoothing,
                                           pileup=args.pileup, train_indel=args.train_indel)
            train_chunk_num = len(train_dataset)

            val_dataset = BinFileDataset(val_list, validation_fn, chunk_size, batch_size, debug_mode=debug_mode,
                                         discard_germline=discard_germline, \
                                         add_af_in_label=False, smoothing=smoothing, pileup=args.pileup,
                                         train_indel=args.train_indel)
            validate_chunk_num = len(val_dataset)
            total_chunks = train_chunk_num + validate_chunk_num
        else:
            total_dataset = BinFileDataset(bin_list, args.bin_fn, chunk_size, batch_size, debug_mode=debug_mode,
                                           discard_germline=discard_germline, \
                                           add_af_in_label=param.add_af_in_label, smoothing=smoothing,
                                           pileup=args.pileup, train_indel=args.train_indel)
            total_chunks = len(total_dataset)
            training_dataset_percentage = param.trainingDatasetPercentage if add_validation_dataset else None
            if add_validation_dataset:
                total_batches = total_chunks // chunks_per_batch
                validate_chunk_num = int(
                    max(1., np.floor(total_batches * (1 - training_dataset_percentage))) * chunks_per_batch)
                train_chunk_num = int(total_chunks - validate_chunk_num)

                train_indices = list(range(train_chunk_num))
                val_indices = list(range(train_chunk_num, train_chunk_num + validate_chunk_num))

                train_dataset = Subset(total_dataset, train_indices)
                val_dataset = Subset(total_dataset, val_indices)

                # set the training dataset to:no debug mode
                train_dataset.dataset.debug_mode = False
                val_dataset.dataset.add_af_in_label = False
            else:
                train_chunk_num = total_chunks
                train_dataset = total_dataset
                # set the training dataset to:no debug mode
                train_dataset.dataset.debug_mode = False

            train_chunk_num = len(train_dataset)
            validate_chunk_num = len(val_dataset) if add_validation_dataset else 0
        train_data_size = train_chunk_num * chunk_size
        validate_data_size = validate_chunk_num * chunk_size

        if args.pileup:
            try:
                from torchinfo import summary
                print(summary(model_nacgt, input_size=tuple([100] + tensor_shape), device=device))
            except:
                pass

        if add_validation_dataset:
            train_dataloader = DataLoader(train_dataset, batch_size=chunks_per_batch, shuffle=True,
                                          num_workers=args.torch_dataset_num_workers,
                                          prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)
            validate_dataloader = DataLoader(val_dataset, batch_size=chunks_per_batch, shuffle=False,
                                             num_workers=args.torch_dataset_num_workers,
                                             prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=chunks_per_batch, shuffle=True,
                                          num_workers=args.torch_dataset_num_workers,
                                          prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)

        criterion = FocalLoss() if apply_focal_loss else nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        if param.add_af_in_label:
            af_loss = AFLoss().to(device)

        optimizer = optim.Adam(
            [{'params': model_nacgt.parameters(), 'lr': learning_rate, 'weight_decay': param.weight_decay}])

        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_steps = train_data_size // batch_size
        validate_steps = validate_data_size // batch_size
        print("[INFO] Using GPU for model training: {}".format(True if device == 'cuda' else False))
        print("[INFO] The size of dataset: {}".format(train_data_size))
        print("[INFO] The training batch size: {}".format(batch_size))
        print("[INFO] The training learning_rate: {}".format(learning_rate))
        print("[INFO] The output model folder: {}".format(ochk_prefix))
        print("[INFO] Apply focal loss in training: {}".format(apply_focal_loss))
        print("[INFO] Discard germline in training: {}".format(discard_germline))
        print("[INFO] Add L2 regularization to model parameters: {}".format(add_l2_regulation_loss))
        print('[INFO] Train steps:{}'.format(train_steps))
        print('[INFO] Validate steps:{}'.format(validate_steps))

        training_loss, validation_loss = 0.0, 0.0
        training_nacgt_loss, validation_nacgt_loss = 0.0, 0.0
        training_step, validation_step = 0, 0
        val_best_nacgt_f1 = 0
        val_best_nacgt_epoch = 0
        echo_each_step = 200

        if args.gpu_csv_fn is not None and check_gpu_status:
            monitor = GPU_Monitor(gpu_id=0, interval=1, duration=120, csv_file=args.gpu_csv_fn)  # Monitor for 5 minutes
            monitor.start()

        for epoch in range(1, max_epoch + 1):
            epoch_loss = 0
            epoch_nacgt_loss = 0
            na_fp, na_tp, na_fn = 0, 0, 0
            nc_fp, nc_tp, nc_fn = 0, 0, 0
            ng_fp, ng_tp, ng_fn = 0, 0, 0
            nt_fp, nt_tp, nt_fn = 0, 0, 0

            # set a random start position for each epoch training
            np.random.seed(epoch)
            train_dataset.dataset.random_start_position = np.random.randint(0, chunk_size)
            train_dataset.dataset.train_flag = True
            t = tqdm(enumerate(train_dataloader), total=train_steps, position=0, leave=True)
            v = tqdm(enumerate(validate_dataloader), total=validate_steps, position=0,
                     leave=True) if not debug_mode else enumerate(validate_dataloader)

            model_nacgt.train()
            for batch_idx, (data, a_label, c_label, g_label, t_label, na_label, nc_label, ng_label, nt_label) in t:
                t.set_description('EPOCH {}'.format(epoch))
                if not args.pileup:
                    data = data.reshape(-1, param.channel_size, param.tumor_matrix_depth_dict[platform],
                                        param.no_of_positions)
                    data = data.to(device) / 100.0
                else:
                    data = data.reshape(-1, param.no_of_positions, param.pileup_channel_size)
                    data = data.to(device)
                # data = data.to(device)
                a_label, c_label, g_label, t_label, na_label, nc_label, ng_label, nt_label = \
                    a_label.reshape(-1, 2).to(device), c_label.reshape(-1, 2).to(device), g_label.reshape(-1, 2).to(
                        device), t_label.reshape(-1, 2).to(device), \
                        na_label.reshape(-1, 2).to(device), nc_label.reshape(-1, 2).to(device), ng_label.reshape(-1,
                                                                                                                 2).to(
                        device), nt_label.reshape(-1, 2).to(device)

                na_output_logit, nc_output_logit, ng_output_logit, nt_output_logit = model_nacgt(data)
                na_output_logit, nc_output_logit, ng_output_logit, nt_output_logit = \
                    na_output_logit.contiguous(), nc_output_logit.contiguous(), ng_output_logit.contiguous(), nt_output_logit.contiguous()
                y_na_truth = torch.argmax(na_label, axis=1)
                y_nc_truth = torch.argmax(nc_label, axis=1)
                y_ng_truth = torch.argmax(ng_label, axis=1)
                y_nt_truth = torch.argmax(nt_label, axis=1)

                optimizer.zero_grad()

                na_loss = criterion(input=na_output_logit, target=na_label) if apply_focal_loss else criterion(
                    na_output_logit,
                    y_na_truth)
                nc_loss = criterion(input=nc_output_logit, target=nc_label) if apply_focal_loss else criterion(
                    nc_output_logit,
                    y_nc_truth)
                ng_loss = criterion(input=ng_output_logit, target=ng_label) if apply_focal_loss else criterion(
                    ng_output_logit,
                    y_ng_truth)
                nt_loss = criterion(input=nt_output_logit, target=nt_label) if apply_focal_loss else criterion(
                    nt_output_logit,
                    y_nt_truth)

                l2_regularization_nacgt_loss = sum([l2_regularization_lambda * 0.5 * params.norm(2) ** 2 for params in
                                                    model_nacgt.parameters()]) if add_l2_regulation_loss else 0.0

                loss_nacgt = na_loss + nc_loss + ng_loss + nt_loss + l2_regularization_nacgt_loss

                loss = loss_nacgt

                loss.backward()

                optimizer.step()

                training_step += 1
                training_loss += loss.item()
                training_nacgt_loss += loss_nacgt.item()
                if add_writer:
                    if training_step % echo_each_step == echo_each_step - 1:
                        writer.add_scalar('training nacgt loss', training_nacgt_loss / echo_each_step, training_step)
                    training_loss = 0.0
                    training_nacgt_loss = 0.0

                y_na_truth = y_na_truth.cpu().numpy()
                y_nc_truth = y_nc_truth.cpu().numpy()
                y_ng_truth = y_ng_truth.cpu().numpy()
                y_nt_truth = y_nt_truth.cpu().numpy()

                y_na_pred = na_output_logit.argmax(dim=1).cpu().numpy()
                y_nc_pred = nc_output_logit.argmax(dim=1).cpu().numpy()
                y_ng_pred = ng_output_logit.argmax(dim=1).cpu().numpy()
                y_nt_pred = nt_output_logit.argmax(dim=1).cpu().numpy()

                arg_index = 1

                na_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                na_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                na_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                nc_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                nc_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                nc_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                ng_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                ng_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                ng_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                nt_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                nt_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                nt_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])

                nacgt_fp = na_fp + nc_fp + ng_fp + nt_fp
                nacgt_fn = na_fn + nc_fn + ng_fn + nt_fn
                nacgt_tp = na_tp + nc_tp + ng_tp + nt_tp

                if batch_idx + 1 == train_steps:
                    break

                epoch_loss += loss
                epoch_nacgt_loss += loss_nacgt
                el_nacgt = epoch_nacgt_loss.detach().cpu().numpy()
                nacgt_precision, nacgt_recall, nacgt_f1_score = cal_metrics(nacgt_tp, nacgt_fp, nacgt_fn)
                t.set_postfix(
                    {'nacgt loss': el_nacgt, 'nacgt precision': nacgt_precision, 'nacgt recall': nacgt_recall,
                     'nacgt f1 score': nacgt_f1_score})

                t.update(1)

            # validation
            val_na_fp, val_na_tp, val_na_fn = 0, 0, 0
            val_nc_fp, val_nc_tp, val_nc_fn = 0, 0, 0
            val_ng_fp, val_ng_tp, val_ng_fn = 0, 0, 0
            val_nt_fp, val_nt_tp, val_nt_fn = 0, 0, 0
            val_epoch_loss = 0
            val_epoch_nacgt_loss = 0
            model_nacgt.eval()
            for batch_idx, (data, a_label, c_label, g_label, t_label, na_label, nc_label, ng_label, nt_label) in v:
                if not debug_mode:
                    v.set_description('VAL EPOCH {}'.format(epoch))
                if not args.pileup:
                    data = data.reshape(-1, param.channel_size, param.tumor_matrix_depth_dict[platform],
                                        param.no_of_positions)
                    data = data.to(device) / 100.0
                else:
                    data = data.reshape(-1, param.no_of_positions, param.pileup_channel_size)
                    data = data.to(device)
                # data = data.to(device)
                a_label, c_label, g_label, t_label, na_label, nc_label, ng_label, nt_label = \
                    a_label.reshape(-1, 2).to(device), c_label.reshape(-1, 2).to(device), g_label.reshape(-1, 2).to(
                        device), t_label.reshape(-1, 2).to(device), \
                        na_label.reshape(-1, 2).to(device), nc_label.reshape(-1, 2).to(device), ng_label.reshape(-1,
                                                                                                                 2).to(
                        device), nt_label.reshape(-1, 2).to(device)

                with torch.no_grad():
                    na_output_logit, nc_output_logit, ng_output_logit, nt_output_logit = model_nacgt(data)

                y_na_truth = torch.argmax(na_label, axis=1)
                y_nc_truth = torch.argmax(nc_label, axis=1)
                y_ng_truth = torch.argmax(ng_label, axis=1)
                y_nt_truth = torch.argmax(nt_label, axis=1)

                optimizer.zero_grad()

                na_loss = criterion(input=na_output_logit, target=na_label) if apply_focal_loss else criterion(
                    na_output_logit,
                    y_na_truth)
                nc_loss = criterion(input=nc_output_logit, target=nc_label) if apply_focal_loss else criterion(
                    nc_output_logit,
                    y_nc_truth)
                ng_loss = criterion(input=ng_output_logit, target=ng_label) if apply_focal_loss else criterion(
                    ng_output_logit,
                    y_ng_truth)
                nt_loss = criterion(input=nt_output_logit, target=nt_label) if apply_focal_loss else criterion(
                    nt_output_logit,
                    y_nt_truth)

                loss_nacgt = na_loss + nc_loss + ng_loss + nt_loss

                loss = loss_nacgt

                validation_loss += loss.item()
                validation_nacgt_loss += loss_nacgt.item()
                if add_writer:
                    if validation_step % echo_each_step == echo_each_step - 1:
                        writer.add_scalar('validation nacgt loss', validation_nacgt_loss / echo_each_step,
                                          validation_step)
                    validation_loss = 0.0
                    validation_nacgt_loss = 0.0

                y_na_truth = y_na_truth.cpu().numpy()
                y_nc_truth = y_nc_truth.cpu().numpy()
                y_ng_truth = y_ng_truth.cpu().numpy()
                y_nt_truth = y_nt_truth.cpu().numpy()

                y_na_pred = na_output_logit.argmax(dim=1).cpu().numpy()
                y_nc_pred = nc_output_logit.argmax(dim=1).cpu().numpy()
                y_ng_pred = ng_output_logit.argmax(dim=1).cpu().numpy()
                y_nt_pred = nt_output_logit.argmax(dim=1).cpu().numpy()

                na_output_pro = torch.softmax(na_output_logit, dim=1)
                nc_output_pro = torch.softmax(nc_output_logit, dim=1)
                ng_output_pro = torch.softmax(ng_output_logit, dim=1)
                nt_output_pro = torch.softmax(nt_output_logit, dim=1)

                arg_index = 1
                if debug_mode:
                    if device == 'cuda':
                        data_numpy = data.cpu().numpy()
                    else:
                        data_numpy = data.numpy()
                    na_cpu_logit = na_output_pro.cpu().numpy()
                    nc_cpu_logit = nc_output_pro.cpu().numpy()
                    ng_cpu_logit = ng_output_pro.cpu().numpy()
                    nt_cpu_logit = nt_output_pro.cpu().numpy()
                    for idx, (x, y) in enumerate(zip(y_na_truth, y_na_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'NA_FN', x, y, na_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'NA_FP', x, y, na_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_nc_truth, y_nc_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'NC_FN', x, y, nc_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'NC_FP', x, y, nc_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_ng_truth, y_ng_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'NG_FN', x, y, ng_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'NG_FP', x, y, ng_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_nt_truth, y_nt_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'NT_FN', x, y, nt_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'NT_FP', x, y, nt_cpu_logit[idx])

                val_na_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                val_na_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                val_na_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                val_nc_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                val_nc_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                val_nc_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                val_ng_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                val_ng_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                val_ng_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                val_nt_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                val_nt_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                val_nt_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])

                val_nacgt_fp = val_na_fp + val_nc_fp + val_ng_fp + val_nt_fp
                val_nacgt_fn = val_na_fn + val_nc_fn + val_ng_fn + val_nt_fn
                val_nacgt_tp = val_na_tp + val_nc_tp + val_ng_tp + val_nt_tp

                if batch_idx + 1 == validate_steps:
                    break

                val_epoch_loss += loss
                val_epoch_nacgt_loss += loss_nacgt
                el_nacgt = val_epoch_nacgt_loss.detach().cpu().numpy()
                val_nacgt_precision, val_nacgt_recall, val_nacgt_f1_score = cal_metrics(val_nacgt_tp, val_nacgt_fp,
                                                                                        val_nacgt_fn)
                if not debug_mode:
                    v.set_postfix(
                        {'val nacgt loss': el_nacgt, 'val nacgt precision': val_nacgt_precision,
                         'val nacgt recall': val_nacgt_recall,
                         'val nacgt f1 score': val_nacgt_f1_score})

                    v.update(1)

            # leanrning rate decay in each epoch end
            lr_scheduler.step()
            save_path = os.path.join(ochk_prefix,
                                     "{}.pkl".format(epoch)) if ochk_prefix is not None else "{}.pkl".format(
                epoch)
            print(save_path)
            torch.save({'model_nacgt': model_nacgt}, save_path)

            if val_nacgt_f1_score > val_best_nacgt_f1:
                save_path = os.path.join(ochk_prefix,
                                         "best_nacgt_f1.pkl") if ochk_prefix is not None else "best_nacgt_f1.pkl"
                print(epoch)
                print(save_path)
                torch.save({'model_nacgt': model_nacgt}, save_path)
                val_best_nacgt_f1 = val_nacgt_f1_score
                val_best_nacgt_epoch = epoch

        print(("[INFO] Val best nacgt f1 score: {}".format(val_best_nacgt_f1)))
        print(("[INFO] Val best nacgt epoch: {}".format(val_best_nacgt_epoch)))

    if add_writer:
        writer.close()

    train_dataset.dataset.close()
    if add_validation_dataset:
        val_dataset.dataset.close()


def train_model_indel(args):
    apply_focal_loss = param.apply_focal_loss
    discard_germline = param.discard_germline
    add_l2_regulation_loss = param.add_l2_regulation_loss
    l2_regularization_lambda = param.l2_regularization_lambda
    debug_mode = args.debug_mode
    platform = args.platform
    ctg_name_string = args.ctg_name
    chkpnt_fn = args.chkpnt_fn
    ochk_prefix = args.ochk_prefix
    add_writer = args.add_writer
    smoothing = args.smoothing
    phase_tumor = args.phase_tumor and args.platform != 'ilmn'

    add_validation_dataset = args.random_validation or (args.validation_fn is not None)
    validation_fn = args.validation_fn
    ctg_name_list = ctg_name_string.split(',') if ctg_name_string is not None else []
    exclude_training_samples = args.exclude_training_samples
    exclude_training_samples = set(exclude_training_samples.split(',')) if exclude_training_samples else set()

    if ochk_prefix and not os.path.exists(ochk_prefix):
        output = run('mkdir -p {}'.format(ochk_prefix), shell=True)
        print("[INFO] Model path empty, create folder:{}".format(ochk_prefix))

    if add_writer:
        writer = SummaryWriter("{}/log".format(ochk_prefix))
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    apply_softmax = False if apply_focal_loss else True
    if args.train_aff:
        if args.pileup:
            channel_size = param.pileup_channel_size
            tumor_channel_size = param.tumor_channel_size if phase_tumor else channel_size
            pileup_tensor_shape = [param.no_of_positions, channel_size]  # only tumor
            tensor_shape = pileup_tensor_shape
            model_acgt = model_path.CvT_Indel(
                num_classes=2,
                s1_emb_dim=16,  # stage 1 - dimension
                s1_emb_kernel=3,  # stage 1 - conv kernel
                s1_emb_stride=2,  # stage 1 - conv stride
                s1_proj_kernel=3,  # stage 1 - attention ds-conv kernel size
                s1_kv_proj_stride=2,  # stage 1 - attention key / value projection stride
                s1_heads=1,  # stage 1 - heads
                s1_depth=1,  # stage 1 - depth
                s1_mlp_mult=4,  # stage 1 - feedforward expansion factor
                s2_emb_dim=64,  # stage 2 - (same as above)
                s2_emb_kernel=3,
                s2_emb_stride=2,
                s2_proj_kernel=3,
                s2_kv_proj_stride=2,
                s2_heads=3,
                s2_depth=2,
                s2_mlp_mult=4,
                s3_emb_dim=128,  # stage 3 - (same as above)
                s3_emb_kernel=3,
                s3_emb_stride=2,
                s3_proj_kernel=3,
                s3_kv_proj_stride=2,
                s3_heads=4,
                s3_depth=3,
                s3_mlp_mult=4,
                dropout=0.,
                dropout_fc=0.3,
                depth=1,
                width=param.no_of_positions,
                dim=param.pileup_channel_size,
                apply_softmax=False,
                model_type="acgt"
            ).to(device)

        if chkpnt_fn is not None:
            model = torch.load(chkpnt_fn, map_location=torch.device(device))
            model_acgt = model['model_acgt']

        batch_size, chunk_size = param.trainBatchSize, param.chunk_size
        assert batch_size % chunk_size == 0
        chunks_per_batch = batch_size // chunk_size
        random.seed(param.RANDOM_SEED)
        np.random.seed(param.RANDOM_SEED)
        learning_rate = args.learning_rate if args.learning_rate else param.initialLearningRate
        max_epoch = args.max_epoch if args.max_epoch else param.maxEpoch
        bin_list = os.listdir(args.bin_fn)

        bin_list = [f for f in bin_list if
                    pass_chr(f, ctg_name_list) and not exist_file_prefix(exclude_training_samples, f)]
        failed_bin_set = set()
        for bin_file in bin_list:
            try:
                table = tables.open_file(os.path.join(args.bin_fn, bin_file), 'r')
                table.close()
            except:
                print("[WARNING] {} cannot open!".format(bin_file))
                failed_bin_set.add(bin_file)
        bin_list = [f for f in bin_list if f not in failed_bin_set]
        if len(bin_list) == 0:
            print("[ERROR] Cannot find ant binary for model training")
            return

        logging.info("[INFO] total {} training bin files: {}".format(len(bin_list), ','.join(bin_list)))

        if validation_fn:
            val_list = os.listdir(validation_fn)
            logging.info("[INFO] total {} validation bin files: {}".format(len(val_list), ','.join(val_list)))
            train_dataset = BinFileDataset(bin_list, args.bin_fn, chunk_size, batch_size, debug_mode=False,
                                           discard_germline=discard_germline, \
                                           add_af_in_label=param.add_af_in_label, smoothing=smoothing, pileup=args.pileup, train_indel=args.train_indel)
            train_chunk_num = len(train_dataset)

            val_dataset = BinFileDataset(val_list, validation_fn, chunk_size, batch_size, debug_mode=debug_mode,
                                         discard_germline=discard_germline, \
                                         add_af_in_label=False, smoothing=smoothing, pileup=args.pileup, train_indel=args.train_indel)
            validate_chunk_num = len(val_dataset)
            total_chunks = train_chunk_num + validate_chunk_num
        else:
            total_dataset = BinFileDataset(bin_list, args.bin_fn, chunk_size, batch_size, debug_mode=debug_mode,
                                           discard_germline=discard_germline, \
                                           add_af_in_label=param.add_af_in_label, smoothing=smoothing, pileup=args.pileup, train_indel=args.train_indel)
            total_chunks = len(total_dataset)
            training_dataset_percentage = param.trainingDatasetPercentage if add_validation_dataset else None
            if add_validation_dataset:
                total_batches = total_chunks // chunks_per_batch
                validate_chunk_num = int(
                    max(1., np.floor(total_batches * (1 - training_dataset_percentage))) * chunks_per_batch)
                train_chunk_num = int(total_chunks - validate_chunk_num)

                train_indices = list(range(train_chunk_num))
                val_indices = list(range(train_chunk_num, train_chunk_num + validate_chunk_num))

                train_dataset = Subset(total_dataset, train_indices)
                val_dataset = Subset(total_dataset, val_indices)

                # set the training dataset to:no debug mode
                train_dataset.dataset.debug_mode = False
                val_dataset.dataset.add_af_in_label = False
            else:
                train_chunk_num = total_chunks
                train_dataset = total_dataset
                # set the training dataset to:no debug mode
                train_dataset.dataset.debug_mode = False

            train_chunk_num = len(train_dataset)
            validate_chunk_num = len(val_dataset) if add_validation_dataset else 0
        train_data_size = train_chunk_num * chunk_size
        validate_data_size = validate_chunk_num * chunk_size

        if args.pileup:
            try:
                from torchinfo import summary
                print(summary(model_acgt, input_size=tuple([100] + tensor_shape), device=device))
            except:
                pass

        if add_validation_dataset:
            train_dataloader = DataLoader(train_dataset, batch_size=chunks_per_batch, shuffle=True,
                                          num_workers=args.torch_dataset_num_workers,
                                          prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)
            validate_dataloader = DataLoader(val_dataset, batch_size=chunks_per_batch, shuffle=False,
                                             num_workers=args.torch_dataset_num_workers,
                                             prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=chunks_per_batch, shuffle=True,
                                          num_workers=args.torch_dataset_num_workers,
                                          prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)

        criterion = FocalLoss() if apply_focal_loss else nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        if param.add_af_in_label:
            af_loss = AFLoss().to(device)

        optimizer = optim.Adam([{'params': model_acgt.parameters(), 'lr': learning_rate, 'weight_decay': param.weight_decay}])

        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_steps = train_data_size // batch_size
        validate_steps = validate_data_size // batch_size
        print("[INFO] Using GPU for model training: {}".format(True if device == 'cuda' else False))
        print("[INFO] The size of dataset: {}".format(train_data_size))
        print("[INFO] The training batch size: {}".format(batch_size))
        print("[INFO] The training learning_rate: {}".format(learning_rate))
        print("[INFO] The output model folder: {}".format(ochk_prefix))
        print("[INFO] Apply focal loss in training: {}".format(apply_focal_loss))
        print("[INFO] Discard germline in training: {}".format(discard_germline))
        print("[INFO] Add L2 regularization to model parameters: {}".format(add_l2_regulation_loss))
        print('[INFO] Train steps:{}'.format(train_steps))
        print('[INFO] Validate steps:{}'.format(validate_steps))

        training_loss, validation_loss = 0.0, 0.0
        training_acgt_loss, validation_acgt_loss = 0.0, 0.0
        training_step, validation_step = 0, 0
        val_best_acgt_f1 = 0
        val_best_acgt_epoch = 0
        echo_each_step = 200

        if args.gpu_csv_fn is not None and check_gpu_status:
            monitor = GPU_Monitor(gpu_id=0, interval=1, duration=120, csv_file=args.gpu_csv_fn)  # Monitor for 5 minutes
            monitor.start()

        for epoch in range(1, max_epoch + 1):
            epoch_loss = 0
            epoch_acgt_loss = 0
            a_fp, a_tp, a_fn = 0, 0, 0
            c_fp, c_tp, c_fn = 0, 0, 0
            g_fp, g_tp, g_fn = 0, 0, 0
            t_fp, t_tp, t_fn = 0, 0, 0
            i_fp, i_tp, i_fn = 0, 0, 0
            d_fp, d_tp, d_fn = 0, 0, 0

            t = tqdm(enumerate(train_dataloader), total=train_steps, position=0, leave=True)
            v = tqdm(enumerate(validate_dataloader), total=validate_steps, position=0,
                     leave=True) if not debug_mode else enumerate(validate_dataloader)
            model_acgt.train()
            for batch_idx, (data, a_label, c_label, g_label, t_label, i_label, d_label, na_label, nc_label, ng_label, nt_label, ni_label, nd_label) in t:
                t.set_description('EPOCH {}'.format(epoch))
                if not args.pileup:
                    data = data.reshape(-1, param.channel_size, param.tumor_matrix_depth_dict[platform], param.no_of_positions)
                    data = data.to(device) / 100.0
                else:
                    data = data.reshape(-1, param.no_of_positions, param.pileup_channel_size)
                    data = data.to(device)
                # data = data.to(device)
                a_label, c_label, g_label, t_label, i_label, d_label, na_label, nc_label, ng_label, nt_label, ni_label, nd_label = \
                    a_label.reshape(-1, 2).to(device), c_label.reshape(-1, 2).to(device), g_label.reshape(-1, 2).to(device), t_label.reshape(-1, 2).to(device), i_label.reshape(-1, 2).to(device), d_label.reshape(-1, 2).to(device),\
                        na_label.reshape(-1, 2).to(device), nc_label.reshape(-1, 2).to(device), ng_label.reshape(-1, 2).to(device), nt_label.reshape(-1, 2).to(device), ni_label.reshape(-1, 2).to(device), nd_label.reshape(-1, 2).to(device)

                a_output_logit, c_output_logit, g_output_logit, t_output_logit, i_output_logit, d_output_logit = model_acgt(data)
                a_output_logit, c_output_logit, g_output_logit, t_output_logit, i_output_logit, d_output_logit = \
                    a_output_logit.contiguous(), c_output_logit.contiguous(), g_output_logit.contiguous(), t_output_logit.contiguous(), i_output_logit.contiguous(), d_output_logit.contiguous()
                y_a_truth = torch.argmax(a_label, axis=1)
                y_c_truth = torch.argmax(c_label, axis=1)
                y_g_truth = torch.argmax(g_label, axis=1)
                y_t_truth = torch.argmax(t_label, axis=1)
                y_i_truth = torch.argmax(i_label, axis=1)
                y_d_truth = torch.argmax(d_label, axis=1)

                optimizer.zero_grad()

                a_loss = criterion(input=a_output_logit, target=a_label) if apply_focal_loss else criterion(a_output_logit,
                                                                                                            y_a_truth)
                c_loss = criterion(input=c_output_logit, target=c_label) if apply_focal_loss else criterion(c_output_logit,
                                                                                                            y_c_truth)
                g_loss = criterion(input=g_output_logit, target=g_label) if apply_focal_loss else criterion(g_output_logit,
                                                                                                            y_g_truth)
                t_loss = criterion(input=t_output_logit, target=t_label) if apply_focal_loss else criterion(t_output_logit,
                                                                                                            y_t_truth)
                i_loss = criterion(input=i_output_logit, target=i_label) if apply_focal_loss else criterion(i_output_logit,
                                                                                                            y_i_truth)
                d_loss = criterion(input=d_output_logit, target=d_label) if apply_focal_loss else criterion(d_output_logit,
                                                                                                            y_d_truth)

                l2_regularization_acgt_loss = sum([l2_regularization_lambda * 0.5 * params.norm(2) ** 2 for params in
                                              model_acgt.parameters()]) if add_l2_regulation_loss else 0.0

                loss_acgt = 1/4 * (a_loss + c_loss + g_loss + t_loss) + 1/2 * (i_loss + d_loss) + l2_regularization_acgt_loss

                loss = loss_acgt

                loss.backward()

                optimizer.step()

                training_step += 1
                training_loss += loss.item()
                training_acgt_loss += loss_acgt.item()
                if add_writer:
                    if training_step % echo_each_step == echo_each_step - 1:
                        writer.add_scalar('training acgt loss', training_acgt_loss / echo_each_step, training_step)
                    training_loss = 0.0
                    training_acgt_loss = 0.0

                y_a_truth = y_a_truth.cpu().numpy()
                y_c_truth = y_c_truth.cpu().numpy()
                y_g_truth = y_g_truth.cpu().numpy()
                y_t_truth = y_t_truth.cpu().numpy()
                y_i_truth = y_i_truth.cpu().numpy()
                y_d_truth = y_d_truth.cpu().numpy()

                y_a_pred = a_output_logit.argmax(dim=1).cpu().numpy()
                y_c_pred = c_output_logit.argmax(dim=1).cpu().numpy()
                y_g_pred = g_output_logit.argmax(dim=1).cpu().numpy()
                y_t_pred = t_output_logit.argmax(dim=1).cpu().numpy()
                y_i_pred = i_output_logit.argmax(dim=1).cpu().numpy()
                y_d_pred = d_output_logit.argmax(dim=1).cpu().numpy()

                arg_index = 1

                a_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                a_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                a_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                c_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                c_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                c_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                g_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                g_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                g_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                t_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                t_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                t_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                i_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_i_truth, y_i_pred)])
                i_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_i_truth, y_i_pred)])
                i_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_i_truth, y_i_pred)])
                d_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_d_truth, y_d_pred)])
                d_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_d_truth, y_d_pred)])
                d_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_d_truth, y_d_pred)])

                acgt_fp = a_fp + c_fp + g_fp + t_fp + i_fp + d_fp
                acgt_fn = a_fn + c_fn + g_fn + t_fn + i_fn + d_fn
                acgt_tp = a_tp + c_tp + g_tp + t_tp + i_tp + d_tp

                if batch_idx + 1 == train_steps:
                    break

                epoch_loss += loss
                epoch_acgt_loss += loss_acgt
                el_acgt = epoch_acgt_loss.detach().cpu().numpy()
                acgt_precision, acgt_recall, acgt_f1_score = cal_metrics(acgt_tp, acgt_fp, acgt_fn)
                t.set_postfix(
                    {'acgt loss': el_acgt, 'acgt precision': acgt_precision, 'acgt recall': acgt_recall,
                     'acgt f1 score': acgt_f1_score
                     }
                )

                t.update(1)

            # validation
            val_a_fp, val_a_tp, val_a_fn = 0, 0, 0
            val_c_fp, val_c_tp, val_c_fn = 0, 0, 0
            val_g_fp, val_g_tp, val_g_fn = 0, 0, 0
            val_t_fp, val_t_tp, val_t_fn = 0, 0, 0
            val_i_fp, val_i_tp, val_i_fn = 0, 0, 0
            val_d_fp, val_d_tp, val_d_fn = 0, 0, 0
            val_epoch_loss = 0
            val_epoch_acgt_loss = 0
            model_acgt.eval()
            for batch_idx, (data, a_label, c_label, g_label, t_label, i_label, d_label, na_label, nc_label, ng_label, nt_label, ni_label, nd_label) in v:
                if not debug_mode:
                    v.set_description('VAL EPOCH {}'.format(epoch))
                if not args.pileup:
                    data = data.reshape(-1, param.channel_size, param.tumor_matrix_depth_dict[platform],
                                        param.no_of_positions)
                    data = data.to(device) / 100.0
                else:
                    data = data.reshape(-1, param.no_of_positions, param.pileup_channel_size)
                    data = data.to(device)
                # data = data.to(device)
                a_label, c_label, g_label, t_label, i_label, d_label, na_label, nc_label, ng_label, nt_label, ni_label, nd_label = \
                    a_label.reshape(-1, 2).to(device), c_label.reshape(-1, 2).to(device), g_label.reshape(-1, 2).to(device), t_label.reshape(-1, 2).to(device), i_label.reshape(-1, 2).to(device), d_label.reshape(-1, 2).to(device),\
                        na_label.reshape(-1, 2).to(device), nc_label.reshape(-1, 2).to(device), ng_label.reshape(-1, 2).to(device), nt_label.reshape(-1, 2).to(device), ni_label.reshape(-1, 2).to(device), nd_label.reshape(-1, 2).to(device)

                with torch.no_grad():
                    a_output_logit, c_output_logit, g_output_logit, t_output_logit, i_output_logit, d_output_logit = model_acgt(
                        data)

                y_a_truth = torch.argmax(a_label, axis=1)
                y_c_truth = torch.argmax(c_label, axis=1)
                y_g_truth = torch.argmax(g_label, axis=1)
                y_t_truth = torch.argmax(t_label, axis=1)
                y_i_truth = torch.argmax(i_label, axis=1)
                y_d_truth = torch.argmax(d_label, axis=1)

                optimizer.zero_grad()

                a_loss = criterion(input=a_output_logit, target=a_label) if apply_focal_loss else criterion(
                    a_output_logit,
                    y_a_truth)
                c_loss = criterion(input=c_output_logit, target=c_label) if apply_focal_loss else criterion(
                    c_output_logit,
                    y_c_truth)
                g_loss = criterion(input=g_output_logit, target=g_label) if apply_focal_loss else criterion(
                    g_output_logit,
                    y_g_truth)
                t_loss = criterion(input=t_output_logit, target=t_label) if apply_focal_loss else criterion(
                    t_output_logit,
                    y_t_truth)
                i_loss = criterion(input=i_output_logit, target=i_label) if apply_focal_loss else criterion(
                    i_output_logit,
                    y_i_truth)
                d_loss = criterion(input=d_output_logit, target=d_label) if apply_focal_loss else criterion(
                    d_output_logit,
                    y_d_truth)

                loss_acgt = 1 / 4 * (a_loss + c_loss + g_loss + t_loss) + 1 / 2 * (i_loss + d_loss)

                loss = loss_acgt

                validation_loss += loss.item()
                validation_acgt_loss += loss_acgt.item()
                if add_writer:
                    if validation_step % echo_each_step == echo_each_step - 1:
                        writer.add_scalar('validation acgt loss', validation_acgt_loss / echo_each_step, validation_step)
                    validation_loss = 0.0
                    validation_acgt_loss = 0.0

                y_a_truth = y_a_truth.cpu().numpy()
                y_c_truth = y_c_truth.cpu().numpy()
                y_g_truth = y_g_truth.cpu().numpy()
                y_t_truth = y_t_truth.cpu().numpy()
                y_i_truth = y_i_truth.cpu().numpy()
                y_d_truth = y_d_truth.cpu().numpy()

                y_a_pred = a_output_logit.argmax(dim=1).cpu().numpy()
                y_c_pred = c_output_logit.argmax(dim=1).cpu().numpy()
                y_g_pred = g_output_logit.argmax(dim=1).cpu().numpy()
                y_t_pred = t_output_logit.argmax(dim=1).cpu().numpy()
                y_i_pred = i_output_logit.argmax(dim=1).cpu().numpy()
                y_d_pred = d_output_logit.argmax(dim=1).cpu().numpy()

                a_output_pro = torch.softmax(a_output_logit, dim=1)
                c_output_pro = torch.softmax(c_output_logit, dim=1)
                g_output_pro = torch.softmax(g_output_logit, dim=1)
                t_output_pro = torch.softmax(t_output_logit, dim=1)
                i_output_pro = torch.softmax(i_output_logit, dim=1)
                d_output_pro = torch.softmax(d_output_logit, dim=1)

                arg_index = 1
                if debug_mode:
                    if device == 'cuda':
                        data_numpy = data.cpu().numpy()
                    else:
                        data_numpy = data.numpy()
                    a_cpu_logit = a_output_pro.cpu().numpy()
                    c_cpu_logit = c_output_pro.cpu().numpy()
                    g_cpu_logit = g_output_pro.cpu().numpy()
                    t_cpu_logit = t_output_pro.cpu().numpy()
                    i_cpu_logit = i_output_pro.cpu().numpy()
                    d_cpu_logit = d_output_pro.cpu().numpy()
                    for idx, (x, y) in enumerate(zip(y_a_truth, y_a_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'A_FN', x, y, a_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'A_FP', x, y, a_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_c_truth, y_c_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'C_FN', x, y, c_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'C_FP', x, y, c_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_g_truth, y_g_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'G_FN', x, y, g_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'G_FP', x, y, g_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_t_truth, y_t_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'T_FN', x, y, t_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'T_FP', x, y, t_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_i_truth, y_i_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'I_FN', x, y, i_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'I_FP', x, y, i_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_d_truth, y_d_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'D_FN', x, y, d_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'D_FP', x, y, d_cpu_logit[idx])

                val_a_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                val_a_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                val_a_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_a_truth, y_a_pred)])
                val_c_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                val_c_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                val_c_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_c_truth, y_c_pred)])
                val_g_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                val_g_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                val_g_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_g_truth, y_g_pred)])
                val_t_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                val_t_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                val_t_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_t_truth, y_t_pred)])
                val_i_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_i_truth, y_i_pred)])
                val_i_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_i_truth, y_i_pred)])
                val_i_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_i_truth, y_i_pred)])
                val_d_fp += sum([True if x != arg_index and y == arg_index else False for x, y in zip(y_d_truth, y_d_pred)])
                val_d_fn += sum([True if x == arg_index and y != arg_index else False for x, y in zip(y_d_truth, y_d_pred)])
                val_d_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_d_truth, y_d_pred)])

                val_acgt_fp = val_a_fp + val_c_fp + val_g_fp + val_t_fp + val_i_fp + val_d_fp
                val_acgt_fn = val_a_fn + val_c_fn + val_g_fn + val_t_fn + val_i_fn + val_d_fn
                val_acgt_tp = val_a_tp + val_c_tp + val_g_tp + val_t_tp + val_i_tp + val_d_tp

                if batch_idx + 1 == validate_steps:
                    break

                val_epoch_loss += loss
                val_epoch_acgt_loss += loss_acgt
                el_acgt = val_epoch_acgt_loss.detach().cpu().numpy()
                val_acgt_precision, val_acgt_recall, val_acgt_f1_score = cal_metrics(val_acgt_tp, val_acgt_fp, val_acgt_fn)
                if not debug_mode:
                    v.set_postfix(
                        {'val acgt loss': el_acgt, 'val acgt precision': val_acgt_precision,
                         'val acgt recall': val_acgt_recall,
                         'val acgt f1 score': val_acgt_f1_score}
                    )

                    v.update(1)

            # leanrning rate decay in each epoch end
            lr_scheduler.step()
            save_path = os.path.join(ochk_prefix, "{}.pkl".format(epoch)) if ochk_prefix is not None else "{}.pkl".format(
                epoch)
            print(save_path)
            torch.save({'model_acgt': model_acgt}, save_path)

            if val_acgt_f1_score > val_best_acgt_f1:
                save_path = os.path.join(ochk_prefix, "best_acgt_f1.pkl") if ochk_prefix is not None else "best_acgt_f1.pkl"
                print(epoch)
                print(save_path)
                torch.save({'model_acgt': model_acgt}, save_path)
                val_best_acgt_f1 = val_acgt_f1_score
                val_best_acgt_epoch = epoch

        print(("[INFO] Val best acgt f1 score: {}".format(val_best_acgt_f1)))
        print(("[INFO] Val best acgt epoch: {}".format(val_best_acgt_epoch)))

    if args.train_neg:
        if args.pileup:
            channel_size = param.pileup_channel_size
            tumor_channel_size = param.tumor_channel_size if phase_tumor else channel_size
            pileup_tensor_shape = [param.no_of_positions, channel_size]  # only tumor
            tensor_shape = pileup_tensor_shape
            model_nacgt = model_path.BiGRU_NACGT_Indel(apply_softmax=apply_softmax,
                                                       num_classes=2,
                                                       channel_size=tensor_shape[1],
                                                       model_type="nacgt").to(device)

        if chkpnt_fn is not None:
            model = torch.load(chkpnt_fn, map_location=torch.device(device))
            model_nacgt = model['model_nacgt']

        batch_size, chunk_size = param.trainBatchSize, param.chunk_size
        assert batch_size % chunk_size == 0
        chunks_per_batch = batch_size // chunk_size
        random.seed(param.RANDOM_SEED)
        np.random.seed(param.RANDOM_SEED)
        learning_rate = args.learning_rate if args.learning_rate else param.initialLearningRate
        max_epoch = args.max_epoch if args.max_epoch else param.maxEpoch
        bin_list = os.listdir(args.bin_fn)

        bin_list = [f for f in bin_list if
                    pass_chr(f, ctg_name_list) and not exist_file_prefix(exclude_training_samples, f)]
        failed_bin_set = set()
        for bin_file in bin_list:
            try:
                table = tables.open_file(os.path.join(args.bin_fn, bin_file), 'r')
                table.close()
            except:
                print("[WARNING] {} cannot open!".format(bin_file))
                failed_bin_set.add(bin_file)
        bin_list = [f for f in bin_list if f not in failed_bin_set]
        if len(bin_list) == 0:
            print("[ERROR] Cannot find ant binary for model training")
            return

        logging.info("[INFO] total {} training bin files: {}".format(len(bin_list), ','.join(bin_list)))

        if validation_fn:
            val_list = os.listdir(validation_fn)
            logging.info("[INFO] total {} validation bin files: {}".format(len(val_list), ','.join(val_list)))
            train_dataset = BinFileDataset(bin_list, args.bin_fn, chunk_size, batch_size, debug_mode=False,
                                           discard_germline=discard_germline, \
                                           add_af_in_label=param.add_af_in_label, smoothing=smoothing,
                                           pileup=args.pileup, train_indel=args.train_indel)
            train_chunk_num = len(train_dataset)

            val_dataset = BinFileDataset(val_list, validation_fn, chunk_size, batch_size, debug_mode=debug_mode,
                                         discard_germline=discard_germline, \
                                         add_af_in_label=False, smoothing=smoothing, pileup=args.pileup,
                                         train_indel=args.train_indel)
            validate_chunk_num = len(val_dataset)
            total_chunks = train_chunk_num + validate_chunk_num
        else:
            total_dataset = BinFileDataset(bin_list, args.bin_fn, chunk_size, batch_size, debug_mode=debug_mode,
                                           discard_germline=discard_germline, \
                                           add_af_in_label=param.add_af_in_label, smoothing=smoothing,
                                           pileup=args.pileup, train_indel=args.train_indel)
            total_chunks = len(total_dataset)
            training_dataset_percentage = param.trainingDatasetPercentage if add_validation_dataset else None
            if add_validation_dataset:
                total_batches = total_chunks // chunks_per_batch
                validate_chunk_num = int(
                    max(1., np.floor(total_batches * (1 - training_dataset_percentage))) * chunks_per_batch)
                train_chunk_num = int(total_chunks - validate_chunk_num)

                train_indices = list(range(train_chunk_num))
                val_indices = list(range(train_chunk_num, train_chunk_num + validate_chunk_num))

                train_dataset = Subset(total_dataset, train_indices)
                val_dataset = Subset(total_dataset, val_indices)

                # set the training dataset to:no debug mode
                train_dataset.dataset.debug_mode = False
                val_dataset.dataset.add_af_in_label = False
            else:
                train_chunk_num = total_chunks
                train_dataset = total_dataset
                # set the training dataset to:no debug mode
                train_dataset.dataset.debug_mode = False

            train_chunk_num = len(train_dataset)
            validate_chunk_num = len(val_dataset) if add_validation_dataset else 0
        train_data_size = train_chunk_num * chunk_size
        validate_data_size = validate_chunk_num * chunk_size

        if args.pileup:
            try:
                from torchinfo import summary
                print(summary(model_nacgt, input_size=tuple([100] + tensor_shape), device=device))
            except:
                pass

        if add_validation_dataset:
            train_dataloader = DataLoader(train_dataset, batch_size=chunks_per_batch, shuffle=True,
                                          num_workers=args.torch_dataset_num_workers,
                                          prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)
            validate_dataloader = DataLoader(val_dataset, batch_size=chunks_per_batch, shuffle=False,
                                             num_workers=args.torch_dataset_num_workers,
                                             prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=chunks_per_batch, shuffle=True,
                                          num_workers=args.torch_dataset_num_workers,
                                          prefetch_factor=args.torch_dataset_prefetch_factor, pin_memory=True)

        criterion = FocalLoss() if apply_focal_loss else nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        if param.add_af_in_label:
            af_loss = AFLoss().to(device)

        optimizer = optim.Adam(
            [{'params': model_nacgt.parameters(), 'lr': learning_rate, 'weight_decay': param.weight_decay}])

        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_steps = train_data_size // batch_size
        validate_steps = validate_data_size // batch_size
        print("[INFO] Using GPU for model training: {}".format(True if device == 'cuda' else False))
        print("[INFO] The size of dataset: {}".format(train_data_size))
        print("[INFO] The training batch size: {}".format(batch_size))
        print("[INFO] The training learning_rate: {}".format(learning_rate))
        print("[INFO] The output model folder: {}".format(ochk_prefix))
        print("[INFO] Apply focal loss in training: {}".format(apply_focal_loss))
        print("[INFO] Discard germline in training: {}".format(discard_germline))
        print("[INFO] Add L2 regularization to model parameters: {}".format(add_l2_regulation_loss))
        print('[INFO] Train steps:{}'.format(train_steps))
        print('[INFO] Validate steps:{}'.format(validate_steps))

        training_loss, validation_loss = 0.0, 0.0
        training_nacgt_loss, validation_nacgt_loss = 0.0, 0.0
        training_step, validation_step = 0, 0
        val_best_nacgt_f1 = 0
        val_best_nacgt_epoch = 0
        echo_each_step = 200

        if args.gpu_csv_fn is not None and check_gpu_status:
            monitor = GPU_Monitor(gpu_id=0, interval=1, duration=120, csv_file=args.gpu_csv_fn)  # Monitor for 5 minutes
            monitor.start()

        for epoch in range(1, max_epoch + 1):
            epoch_loss = 0
            epoch_nacgt_loss = 0
            na_fp, na_tp, na_fn = 0, 0, 0
            nc_fp, nc_tp, nc_fn = 0, 0, 0
            ng_fp, ng_tp, ng_fn = 0, 0, 0
            nt_fp, nt_tp, nt_fn = 0, 0, 0
            ni_fp, ni_tp, ni_fn = 0, 0, 0
            nd_fp, nd_tp, nd_fn = 0, 0, 0

            t = tqdm(enumerate(train_dataloader), total=train_steps, position=0, leave=True)
            v = tqdm(enumerate(validate_dataloader), total=validate_steps, position=0,
                     leave=True) if not debug_mode else enumerate(validate_dataloader)
            model_nacgt.train()
            for batch_idx, (
            data, a_label, c_label, g_label, t_label, i_label, d_label, na_label, nc_label, ng_label, nt_label,
            ni_label, nd_label) in t:
                t.set_description('EPOCH {}'.format(epoch))
                if not args.pileup:
                    data = data.reshape(-1, param.channel_size, param.tumor_matrix_depth_dict[platform],
                                        param.no_of_positions)
                    data = data.to(device) / 100.0
                else:
                    data = data.reshape(-1, param.no_of_positions, param.pileup_channel_size)
                    data = data.to(device)
                # data = data.to(device)
                a_label, c_label, g_label, t_label, i_label, d_label, na_label, nc_label, ng_label, nt_label, ni_label, nd_label = \
                    a_label.reshape(-1, 2).to(device), c_label.reshape(-1, 2).to(device), g_label.reshape(-1, 2).to(
                        device), t_label.reshape(-1, 2).to(device), i_label.reshape(-1, 2).to(device), d_label.reshape(
                        -1, 2).to(device), \
                        na_label.reshape(-1, 2).to(device), nc_label.reshape(-1, 2).to(device), ng_label.reshape(-1,
                                                                                                                 2).to(
                        device), nt_label.reshape(-1, 2).to(device), ni_label.reshape(-1, 2).to(
                        device), nd_label.reshape(-1, 2).to(device)

                na_output_logit, nc_output_logit, ng_output_logit, nt_output_logit, ni_output_logit, nd_output_logit = model_nacgt(
                    data)
                na_output_logit, nc_output_logit, ng_output_logit, nt_output_logit, ni_output_logit, nd_output_logit = \
                    na_output_logit.contiguous(), nc_output_logit.contiguous(), ng_output_logit.contiguous(), nt_output_logit.contiguous(), ni_output_logit.contiguous(), nd_output_logit.contiguous()
                y_na_truth = torch.argmax(na_label, axis=1)
                y_nc_truth = torch.argmax(nc_label, axis=1)
                y_ng_truth = torch.argmax(ng_label, axis=1)
                y_nt_truth = torch.argmax(nt_label, axis=1)
                y_ni_truth = torch.argmax(ni_label, axis=1)
                y_nd_truth = torch.argmax(nd_label, axis=1)

                optimizer.zero_grad()

                na_loss = criterion(input=na_output_logit, target=na_label) if apply_focal_loss else criterion(
                    na_output_logit,
                    y_na_truth)
                nc_loss = criterion(input=nc_output_logit, target=nc_label) if apply_focal_loss else criterion(
                    nc_output_logit,
                    y_nc_truth)
                ng_loss = criterion(input=ng_output_logit, target=ng_label) if apply_focal_loss else criterion(
                    ng_output_logit,
                    y_ng_truth)
                nt_loss = criterion(input=nt_output_logit, target=nt_label) if apply_focal_loss else criterion(
                    nt_output_logit,
                    y_nt_truth)
                ni_loss = criterion(input=ni_output_logit, target=ni_label) if apply_focal_loss else criterion(
                    ni_output_logit,
                    y_ni_truth)
                nd_loss = criterion(input=nd_output_logit, target=nd_label) if apply_focal_loss else criterion(
                    nd_output_logit,
                    y_nd_truth)

                l2_regularization_nacgt_loss = sum([l2_regularization_lambda * 0.5 * params.norm(2) ** 2 for params in
                                                    model_nacgt.parameters()]) if add_l2_regulation_loss else 0.0

                loss_nacgt = 1 / 4 * (na_loss + nc_loss + ng_loss + nt_loss) + 1 / 2 * (
                        ni_loss + nd_loss) + l2_regularization_nacgt_loss

                loss = loss_nacgt

                loss.backward()

                optimizer.step()

                training_step += 1
                training_loss += loss.item()
                training_nacgt_loss += loss_nacgt.item()
                if add_writer:
                    if training_step % echo_each_step == echo_each_step - 1:
                        writer.add_scalar('training nacgt loss', training_nacgt_loss / echo_each_step, training_step)
                    training_loss = 0.0
                    training_nacgt_loss = 0.0

                y_na_truth = y_na_truth.cpu().numpy()
                y_nc_truth = y_nc_truth.cpu().numpy()
                y_ng_truth = y_ng_truth.cpu().numpy()
                y_nt_truth = y_nt_truth.cpu().numpy()
                y_ni_truth = y_ni_truth.cpu().numpy()
                y_nd_truth = y_nd_truth.cpu().numpy()

                y_na_pred = na_output_logit.argmax(dim=1).cpu().numpy()
                y_nc_pred = nc_output_logit.argmax(dim=1).cpu().numpy()
                y_ng_pred = ng_output_logit.argmax(dim=1).cpu().numpy()
                y_nt_pred = nt_output_logit.argmax(dim=1).cpu().numpy()
                y_ni_pred = ni_output_logit.argmax(dim=1).cpu().numpy()
                y_nd_pred = nd_output_logit.argmax(dim=1).cpu().numpy()

                arg_index = 1

                na_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                na_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                na_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                nc_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                nc_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                nc_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                ng_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                ng_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                ng_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                nt_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                nt_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                nt_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                ni_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_ni_truth, y_ni_pred)])
                ni_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_ni_truth, y_ni_pred)])
                ni_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_ni_truth, y_ni_pred)])
                nd_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nd_truth, y_nd_pred)])
                nd_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nd_truth, y_nd_pred)])
                nd_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nd_truth, y_nd_pred)])

                nacgt_fp = na_fp + nc_fp + ng_fp + nt_fp + ni_fp + nd_fp
                nacgt_fn = na_fn + nc_fn + ng_fn + nt_fn + ni_fn + nd_fn
                nacgt_tp = na_tp + nc_tp + ng_tp + nt_tp + ni_tp + nd_tp

                if batch_idx + 1 == train_steps:
                    break

                epoch_loss += loss
                epoch_nacgt_loss += loss_nacgt
                el_nacgt = epoch_nacgt_loss.detach().cpu().numpy()
                nacgt_precision, nacgt_recall, nacgt_f1_score = cal_metrics(nacgt_tp, nacgt_fp, nacgt_fn)
                t.set_postfix(
                    {'nacgt loss': el_nacgt, 'nacgt precision': nacgt_precision, 'nacgt recall': nacgt_recall,
                     'nacgt f1 score': nacgt_f1_score
                     }
                )

                t.update(1)

            # validation
            val_na_fp, val_na_tp, val_na_fn = 0, 0, 0
            val_nc_fp, val_nc_tp, val_nc_fn = 0, 0, 0
            val_ng_fp, val_ng_tp, val_ng_fn = 0, 0, 0
            val_nt_fp, val_nt_tp, val_nt_fn = 0, 0, 0
            val_ni_fp, val_ni_tp, val_ni_fn = 0, 0, 0
            val_nd_fp, val_nd_tp, val_nd_fn = 0, 0, 0
            val_epoch_loss = 0
            val_epoch_nacgt_loss = 0
            model_nacgt.eval()
            for batch_idx, (
            data, a_label, c_label, g_label, t_label, i_label, d_label, na_label, nc_label, ng_label, nt_label,
            ni_label, nd_label) in v:
                if not debug_mode:
                    v.set_description('VAL EPOCH {}'.format(epoch))
                if not args.pileup:
                    data = data.reshape(-1, param.channel_size, param.tumor_matrix_depth_dict[platform],
                                        param.no_of_positions)
                    data = data.to(device) / 100.0
                else:
                    data = data.reshape(-1, param.no_of_positions, param.pileup_channel_size)
                    data = data.to(device)
                # data = data.to(device)
                a_label, c_label, g_label, t_label, i_label, d_label, na_label, nc_label, ng_label, nt_label, ni_label, nd_label = \
                    a_label.reshape(-1, 2).to(device), c_label.reshape(-1, 2).to(device), g_label.reshape(-1, 2).to(
                        device), t_label.reshape(-1, 2).to(device), i_label.reshape(-1, 2).to(device), d_label.reshape(
                        -1, 2).to(device), \
                        na_label.reshape(-1, 2).to(device), nc_label.reshape(-1, 2).to(device), ng_label.reshape(-1,
                                                                                                                 2).to(
                        device), nt_label.reshape(-1, 2).to(device), ni_label.reshape(-1, 2).to(
                        device), nd_label.reshape(-1, 2).to(device)

                with torch.no_grad():
                    na_output_logit, nc_output_logit, ng_output_logit, nt_output_logit, ni_output_logit, nd_output_logit = model_nacgt(
                        data)

                y_na_truth = torch.argmax(na_label, axis=1)
                y_nc_truth = torch.argmax(nc_label, axis=1)
                y_ng_truth = torch.argmax(ng_label, axis=1)
                y_nt_truth = torch.argmax(nt_label, axis=1)
                y_ni_truth = torch.argmax(ni_label, axis=1)
                y_nd_truth = torch.argmax(nd_label, axis=1)

                optimizer.zero_grad()

                na_loss = criterion(input=na_output_logit, target=na_label) if apply_focal_loss else criterion(
                    na_output_logit,
                    y_na_truth)
                nc_loss = criterion(input=nc_output_logit, target=nc_label) if apply_focal_loss else criterion(
                    nc_output_logit,
                    y_nc_truth)
                ng_loss = criterion(input=ng_output_logit, target=ng_label) if apply_focal_loss else criterion(
                    ng_output_logit,
                    y_ng_truth)
                nt_loss = criterion(input=nt_output_logit, target=nt_label) if apply_focal_loss else criterion(
                    nt_output_logit,
                    y_nt_truth)
                ni_loss = criterion(input=ni_output_logit, target=ni_label) if apply_focal_loss else criterion(
                    ni_output_logit,
                    y_ni_truth)
                nd_loss = criterion(input=nd_output_logit, target=nd_label) if apply_focal_loss else criterion(
                    nd_output_logit,
                    y_nd_truth)

                loss_nacgt = 1 / 4 * (na_loss + nc_loss + ng_loss + nt_loss) + 1 / 2 * (ni_loss + nd_loss)

                loss = loss_nacgt

                validation_loss += loss.item()
                validation_nacgt_loss += loss_nacgt.item()
                if add_writer:
                    if validation_step % echo_each_step == echo_each_step - 1:
                        writer.add_scalar('validation nacgt loss', validation_nacgt_loss / echo_each_step,
                                          validation_step)
                    validation_loss = 0.0
                    validation_nacgt_loss = 0.0

                y_na_truth = y_na_truth.cpu().numpy()
                y_nc_truth = y_nc_truth.cpu().numpy()
                y_ng_truth = y_ng_truth.cpu().numpy()
                y_nt_truth = y_nt_truth.cpu().numpy()
                y_ni_truth = y_ni_truth.cpu().numpy()
                y_nd_truth = y_nd_truth.cpu().numpy()

                y_na_pred = na_output_logit.argmax(dim=1).cpu().numpy()
                y_nc_pred = nc_output_logit.argmax(dim=1).cpu().numpy()
                y_ng_pred = ng_output_logit.argmax(dim=1).cpu().numpy()
                y_nt_pred = nt_output_logit.argmax(dim=1).cpu().numpy()
                y_ni_pred = ni_output_logit.argmax(dim=1).cpu().numpy()
                y_nd_pred = nd_output_logit.argmax(dim=1).cpu().numpy()

                na_output_pro = torch.softmax(na_output_logit, dim=1)
                nc_output_pro = torch.softmax(nc_output_logit, dim=1)
                ng_output_pro = torch.softmax(ng_output_logit, dim=1)
                nt_output_pro = torch.softmax(nt_output_logit, dim=1)
                ni_output_pro = torch.softmax(ni_output_logit, dim=1)
                nd_output_pro = torch.softmax(nd_output_logit, dim=1)

                arg_index = 1
                if debug_mode:
                    if device == 'cuda':
                        data_numpy = data.cpu().numpy()
                    else:
                        data_numpy = data.numpy()
                    na_cpu_logit = na_output_pro.cpu().numpy()
                    nc_cpu_logit = nc_output_pro.cpu().numpy()
                    ng_cpu_logit = ng_output_pro.cpu().numpy()
                    nt_cpu_logit = nt_output_pro.cpu().numpy()
                    ni_cpu_logit = ni_output_pro.cpu().numpy()
                    nd_cpu_logit = nd_output_pro.cpu().numpy()
                    for idx, (x, y) in enumerate(zip(y_na_truth, y_na_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'NA_FN', x, y, na_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'NA_FP', x, y, na_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_nc_truth, y_nc_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'NC_FN', x, y, nc_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'NC_FP', x, y, nc_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_ng_truth, y_ng_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'NG_FN', x, y, ng_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'NG_FP', x, y, ng_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_nt_truth, y_nt_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'NT_FN', x, y, nt_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'NT_FP', x, y, nt_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_ni_truth, y_ni_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'NI_FN', x, y, ni_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'NI_FP', x, y, ni_cpu_logit[idx])
                    for idx, (x, y) in enumerate(zip(y_nd_truth, y_nd_pred)):
                        if x == arg_index and y != arg_index:
                            print(idx, 'ND_FN', x, y, nd_cpu_logit[idx])
                        if x != arg_index and y == arg_index:
                            print(idx, 'ND_FP', x, y, nd_cpu_logit[idx])

                val_na_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                val_na_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                val_na_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_na_truth, y_na_pred)])
                val_nc_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                val_nc_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                val_nc_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nc_truth, y_nc_pred)])
                val_ng_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                val_ng_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                val_ng_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_ng_truth, y_ng_pred)])
                val_nt_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                val_nt_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                val_nt_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nt_truth, y_nt_pred)])
                val_ni_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_ni_truth, y_ni_pred)])
                val_ni_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_ni_truth, y_ni_pred)])
                val_ni_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_ni_truth, y_ni_pred)])
                val_nd_fp += sum(
                    [True if x != arg_index and y == arg_index else False for x, y in zip(y_nd_truth, y_nd_pred)])
                val_nd_fn += sum(
                    [True if x == arg_index and y != arg_index else False for x, y in zip(y_nd_truth, y_nd_pred)])
                val_nd_tp += sum([True if x == y and x == arg_index else False for x, y in zip(y_nd_truth, y_nd_pred)])

                val_nacgt_fp = val_na_fp + val_nc_fp + val_ng_fp + val_nt_fp + val_ni_fp + val_nd_fp
                val_nacgt_fn = val_na_fn + val_nc_fn + val_ng_fn + val_nt_fn + val_ni_fn + val_nd_fn
                val_nacgt_tp = val_na_tp + val_nc_tp + val_ng_tp + val_nt_tp + val_ni_tp + val_nd_tp

                if batch_idx + 1 == validate_steps:
                    break

                val_epoch_loss += loss
                val_epoch_nacgt_loss += loss_nacgt
                el_nacgt = val_epoch_nacgt_loss.detach().cpu().numpy()
                val_nacgt_precision, val_nacgt_recall, val_nacgt_f1_score = cal_metrics(val_nacgt_tp, val_nacgt_fp,
                                                                                        val_nacgt_fn)
                if not debug_mode:
                    v.set_postfix(
                        {'val nacgt loss': el_nacgt, 'val nacgt precision': val_nacgt_precision,
                         'val nacgt recall': val_nacgt_recall,
                         'val nacgt f1 score': val_nacgt_f1_score}
                    )

                    v.update(1)

            # leanrning rate decay in each epoch end
            lr_scheduler.step()
            save_path = os.path.join(ochk_prefix,
                                     "{}.pkl".format(epoch)) if ochk_prefix is not None else "{}.pkl".format(
                epoch)
            print(save_path)
            torch.save({'model_nacgt': model_nacgt}, save_path)

            if val_nacgt_f1_score > val_best_nacgt_f1:
                save_path = os.path.join(ochk_prefix,
                                         "best_nacgt_f1.pkl") if ochk_prefix is not None else "best_nacgt_f1.pkl"
                print(epoch)
                print(save_path)
                torch.save({'model_nacgt': model_nacgt}, save_path)
                val_best_nacgt_f1 = val_nacgt_f1_score
                val_best_nacgt_epoch = epoch

        print(("[INFO] Val best nacgt f1 score: {}".format(val_best_nacgt_f1)))
        print(("[INFO] Val best nacgt epoch: {}".format(val_best_nacgt_epoch)))

    if add_writer:
        writer.close()

    train_dataset.dataset.close()
    if add_validation_dataset:
        val_dataset.dataset.close()


def main():
    parser = ArgumentParser(description="Train a somatic model")

    parser.add_argument('--platform', type=str, default="ont",
                        help="Select the sequencing platform of the input. Default: %(default)s")

    parser.add_argument('--ctg_name', type=str, default=None,
                        help="The name of sequence to be processed")

    parser.add_argument('--bin_fn', type=str, default="", required=True,
                        help="Binary tensor input, support multiple bin readers using pytables")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a model to resume training or for fine-tuning")

    parser.add_argument('--ochk_prefix', type=str, default=None,
                        help="Prefix for model output after each epoch")

    # options for advanced users
    parser.add_argument('--max_epoch', type=int, default=None,
                        help="Maximum number of training epochs")

    parser.add_argument('--smoothing', type=int, default=param.smoothing,
                        help="Label smoothing with epsilon in training")

    parser.add_argument('--learning_rate', type=float, default=None,
                        help="Set the initial learning rate, default: %(default)s")

    parser.add_argument('--exclude_training_samples', type=str, default=None,
                        help="Define training samples to be excluded")

    parser.add_argument('--torch_dataset_num_workers', type=int, default=6,
                        help="Threads for torch dataset to preload datasets")

    parser.add_argument('--torch_dataset_prefetch_factor', type=int, default=6,
                        help="Prefetch factor for torch dataset to preload datasets")

    # mutually-incompatible validation options
    vgrp = parser.add_mutually_exclusive_group()
    vgrp.add_argument('--random_validation', action='store_true',
                      help="Use random sample of dataset for validation, default: %(default)s")

    vgrp.add_argument('--validation_fn', type=str, default=None,
                      help="Binary tensor input for use in validation: %(default)s")

    # Internal process control
    ## use siamese network in training
    parser.add_argument('--use_siam', action='store_true',
                        help=SUPPRESS)

    ## use contrastive loss in training
    parser.add_argument('--add_contrastive', action='store_true',
                        help=SUPPRESS)

    ## In pileup training mode or not
    parser.add_argument('--pileup', action='store_true',
                        help=SUPPRESS)

    ## Add indel length for training and calling, default true for full alignment
    parser.add_argument('--add_indel_length', type=str2bool, default=False,
                        help=SUPPRESS)

    ## use resnet for model training
    parser.add_argument('--use_resnet', type=str2bool, default=False,
                        help=SUPPRESS)

    ## add logging writer using torchvision
    parser.add_argument('--add_writer', type=str2bool, default=False,
                        help=SUPPRESS)

    ## Debug mode
    parser.add_argument('--debug_mode', type=str2bool, default=False,
                        help=SUPPRESS)

    parser.add_argument('--phase_tumor', type=str2bool, default=False,
                        help=SUPPRESS)

    parser.add_argument('--gpu_csv_fn', type=str, default=None,
                        help=SUPPRESS)

    parser.add_argument('--train_indel', type=str2bool, default=False,
                        help=SUPPRESS)

    parser.add_argument('--train_aff', type=str2bool, default=False,
                        help=SUPPRESS)

    parser.add_argument('--train_neg', type=str2bool, default=False,
                        help=SUPPRESS)

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    if not args.train_indel:
        train_model(args)
    else:
        train_model_indel(args)


if __name__ == "__main__":
    main()
