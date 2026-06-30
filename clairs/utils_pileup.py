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

import sys
import os
import gc
import shlex
import tables
import numpy as np
import heapq
from random import random
from collections import defaultdict
from itertools import product

from shared.interval_tree import bed_tree_from, is_region_in
from shared.utils import subprocess_popen, IUPAC_base_to_num_dict as BASE2NUM

FILTERS = tables.Filters(complib='blosc:lz4hc', complevel=5)
shuffle_bin_size = 3000
PREFIX_CHAR_STR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def setup_environment():
    gc.enable()


def batches_from(iterable, item_from, batch_size=1):
    iterable = iter(iterable)
    while True:
        chunk = []
        for _ in range(batch_size):
            try:
                chunk.append(item_from(next(iterable)))
            except StopIteration:
                yield chunk
                return
        yield chunk


def variant_map_from(var_fn, tree, is_tree_empty):
    Y = {}
    miss_variant_set = set()
    if var_fn is None:
        return Y, miss_variant_set

    f = subprocess_popen(shlex.split("gzip -fdc %s" % (var_fn)))
    for row in f.stdout:
        columns = row.split()
        ctg_name, position_str = columns[0], columns[1]
        genotype1, genotype2 = columns[-2], columns[-1]
        key = ctg_name + ":" + position_str
        if genotype1 == '-1' or genotype2 == '-1':
            miss_variant_set.add(key)
            continue
        if not (is_tree_empty or is_region_in(tree, ctg_name, int(position_str))):
            continue

        Y[key] = output_labels_from_vcf_columns(columns)

    f.stdout.close()
    f.wait()
    return Y, miss_variant_set


def write_table_dict(table_dict, tumor_matrix, a_label, c_label, g_label, t_label, na_label, nc_label, ng_label, nt_label, pos, total, tumor_alt_info,
                     tensor_shape, pileup):
    input_matrix = []
    from shared.param import pileup_channel_size
    channel_size = pileup_channel_size
    tumor_channel_size = param.tumor_channel_size
    apply_normalize = False
    if apply_normalize:
        tumor_coverage = float(tumor_alt_info.split('-')[0])

        if tumor_coverage == 0:
            return total
        for idx in range(param.no_of_positions):
            input_matrix += [float(item) / tumor_coverage for item in
                             tumor_matrix[idx * tumor_channel_size: (idx + 1) * tumor_channel_size]]
    else:
        for idx in range(param.no_of_positions):
            input_matrix += tumor_matrix[idx * tumor_channel_size: (idx + 1) * tumor_channel_size]

    if len(input_matrix) != param.no_of_positions * (channel_size):
        return total
    table_dict['input_matrix'].append(input_matrix)
    table_dict['position'].append(pos)
    table_dict['a_label'].append(a_label)
    table_dict['c_label'].append(c_label)
    table_dict['g_label'].append(g_label)
    table_dict['t_label'].append(t_label)
    table_dict['na_label'].append(na_label)
    table_dict['nc_label'].append(nc_label)
    table_dict['ng_label'].append(ng_label)
    table_dict['nt_label'].append(nt_label)
    table_dict['tumor_alt_info'].append(tumor_alt_info)

    return total + 1


def write_table_dict_indel(table_dict, tumor_matrix, a_label, c_label, g_label, t_label, i_label, d_label, na_label, nc_label, ng_label, nt_label, ni_label, nd_label, pos, total, tumor_alt_info,
                     tensor_shape, pileup):
    input_matrix = []
    from shared.param import pileup_channel_size
    channel_size = pileup_channel_size
    tumor_channel_size = param.tumor_channel_size
    apply_normalize = False
    if apply_normalize:
        tumor_coverage = float(tumor_alt_info.split('-')[0])

        if tumor_coverage == 0:
            return total
        for idx in range(param.no_of_positions):
            input_matrix += [float(item) / tumor_coverage for item in
                             tumor_matrix[idx * tumor_channel_size: (idx + 1) * tumor_channel_size]]
    else:
        for idx in range(param.no_of_positions):
            input_matrix += tumor_matrix[idx * tumor_channel_size: (idx + 1) * tumor_channel_size]

    if len(input_matrix) != param.no_of_positions * (channel_size):
        return total
    table_dict['input_matrix'].append(input_matrix)
    table_dict['position'].append(pos)
    table_dict['a_label'].append(a_label)
    table_dict['c_label'].append(c_label)
    table_dict['g_label'].append(g_label)
    table_dict['t_label'].append(t_label)
    table_dict['i_label'].append(i_label)
    table_dict['d_label'].append(d_label)
    table_dict['na_label'].append(na_label)
    table_dict['nc_label'].append(nc_label)
    table_dict['ng_label'].append(ng_label)
    table_dict['nt_label'].append(nt_label)
    table_dict['ni_label'].append(ni_label)
    table_dict['nd_label'].append(nd_label)
    table_dict['tumor_alt_info'].append(tumor_alt_info)

    return total + 1


def update_table_dict():
    table_dict = {}
    table_dict['input_matrix'] = []
    table_dict['tumor_alt_info'] = []
    table_dict['position'] = []
    table_dict['a_label'] = []
    table_dict['c_label'] = []
    table_dict['g_label'] = []
    table_dict['t_label'] = []
    table_dict['na_label'] = []
    table_dict['nc_label'] = []
    table_dict['ng_label'] = []
    table_dict['nt_label'] = []
    return table_dict


def update_table_dict_indel():
    table_dict = {}
    table_dict['input_matrix'] = []
    table_dict['tumor_alt_info'] = []
    table_dict['position'] = []
    table_dict['a_label'] = []
    table_dict['c_label'] = []
    table_dict['g_label'] = []
    table_dict['t_label'] = []
    table_dict['i_label'] = []
    table_dict['d_label'] = []
    table_dict['na_label'] = []
    table_dict['nc_label'] = []
    table_dict['ng_label'] = []
    table_dict['nt_label'] = []
    table_dict['ni_label'] = []
    table_dict['nd_label'] = []
    return table_dict


def write_table_file(table_file, table_dict, tensor_shape, float_type):
    """
    Write pileup or full alignment tensor into compressed bin file.
    table_dict: dictionary include all training information (tensor position, label, altnative bases).
    string: input tensor string, need add padding to meet the depth requirement.
    tree: dictionary(contig name : intervaltree) for quick region querying.
    miss_variant_set:  sometimes there will have true variant missing after downsampling reads.
    is_allow_duplicate_chr_pos: whether allow duplicate positions when training, if there exists downsampled data, lower depth will add a random prefix character.
    non_variant_subsample_ratio: define a maximum non variant ratio for training, we always expect use more non variant data, while it would greatly increase training
    time, especially in ont data, here we usually use 1:1 or 1:2 for variant candidate: non variant candidate.
    """
    float_type = 'float32'
    try:
        input_matrix = np.array(table_dict['input_matrix'], np.dtype(float_type)).reshape([-1] + tensor_shape)
    except:
        return
    table_file.root.input_matrix.append(input_matrix)

    table_file.root.tumor_alt_info.append(np.array(table_dict['tumor_alt_info']).reshape(-1, 1))
    table_file.root.position.append(np.array(table_dict['position']).reshape(-1, 1))
    table_file.root.a_label.append(np.array(table_dict['a_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.c_label.append(np.array(table_dict['c_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.g_label.append(np.array(table_dict['g_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.t_label.append(np.array(table_dict['t_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.na_label.append(np.array(table_dict['na_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.nc_label.append(np.array(table_dict['nc_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.ng_label.append(np.array(table_dict['ng_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.nt_label.append(np.array(table_dict['nt_label'], np.dtype(float_type)).reshape(-1, 2))
    table_dict = update_table_dict()

    return table_dict


def write_table_file_indel(table_file, table_dict, tensor_shape, float_type):
    """
    Write pileup or full alignment tensor into compressed bin file.
    table_dict: dictionary include all training information (tensor position, label, altnative bases).
    string: input tensor string, need add padding to meet the depth requirement.
    tree: dictionary(contig name : intervaltree) for quick region querying.
    miss_variant_set:  sometimes there will have true variant missing after downsampling reads.
    is_allow_duplicate_chr_pos: whether allow duplicate positions when training, if there exists downsampled data, lower depth will add a random prefix character.
    non_variant_subsample_ratio: define a maximum non variant ratio for training, we always expect use more non variant data, while it would greatly increase training
    time, especially in ont data, here we usually use 1:1 or 1:2 for variant candidate: non variant candidate.
    """
    float_type = 'float32'
    try:
        input_matrix = np.array(table_dict['input_matrix'], np.dtype(float_type)).reshape([-1] + tensor_shape)
    except:
        return
    table_file.root.input_matrix.append(input_matrix)

    table_file.root.tumor_alt_info.append(np.array(table_dict['tumor_alt_info']).reshape(-1, 1))
    table_file.root.position.append(np.array(table_dict['position']).reshape(-1, 1))
    table_file.root.a_label.append(np.array(table_dict['a_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.c_label.append(np.array(table_dict['c_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.g_label.append(np.array(table_dict['g_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.t_label.append(np.array(table_dict['t_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.i_label.append(np.array(table_dict['i_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.d_label.append(np.array(table_dict['d_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.na_label.append(np.array(table_dict['na_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.nc_label.append(np.array(table_dict['nc_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.ng_label.append(np.array(table_dict['ng_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.nt_label.append(np.array(table_dict['nt_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.ni_label.append(np.array(table_dict['ni_label'], np.dtype(float_type)).reshape(-1, 2))
    table_file.root.nd_label.append(np.array(table_dict['nd_label'], np.dtype(float_type)).reshape(-1, 2))
    table_dict = update_table_dict_indel()

    return table_dict


def print_bin_size(path, prefix=None):
    import tables
    import os
    total = 0
    for file_name in os.listdir(path):
        if prefix and not file_name.startswith(prefix):
            continue
        table = tables.open_file(os.path.join(path, file_name), 'r')
        print("[INFO] {} size is: {}".format(file_name, len(table.root.label)))
        total += len(table.root.label)
    print('[INFO] total: {}'.format(total))


def get_key_list(input_dict, shuffle=True):
    output_list = []
    for key, infos in input_dict.items():
        if 'tumor' not in infos:
            continue
        tumor_index_list = range(len(infos['tumor']))
        for x in tumor_index_list:
            output_list.append((key, x))
    if shuffle == True:
        np.random.shuffle(output_list)
    return output_list


def bin_reader_generator_from(subprocess_process, Y, is_tree_empty, tree, miss_variant_set,
                              is_allow_duplicate_chr_pos=False, non_variant_subsample_ratio=1.0, is_tumor=False):
    """
    Bin reader generator for bin file generation.
    subprocess_list: a list includes all tensor generator of each tensor file.
    Y: dictionary (contig name: label information) to store all variant and non variant information.
    tree: dictionary(contig name : intervaltree) for quick region querying.
    miss_variant_set:  sometimes there will have true variant missing after downsampling reads.
    is_allow_duplicate_chr_pos: whether allow duplicate positions when training, if there exists downsampled data, lower depth will add a random prefix character.
    non_variant_subsample_ratio: define a maximum non variant ratio for training, we always expect use more non variant data, while it would greatly increase training
    time, especially in ont data, here we usually use 1:1 or 1:2 for variant candidate: non variant candidate.
    """
    dup_pos_end_flag = False
    pre_pos = None
    for row_idx, row in enumerate(subprocess_process.stdout):
        columns = row.split("\t")
        try:
            chrom, center_pos, seq, string, alt_info, tumor_tag, variant_type, ref_center, alt_center = columns
        except:
            continue
        alt_info = alt_info.rstrip()
        if not (is_tree_empty or is_region_in(tree, chrom, int(center_pos))):
            continue
        seq = seq.upper()
        if seq[param.flankingBaseNum] not in 'ACGT':
            continue
        key = chrom + ":" + center_pos
        is_reference = key not in Y

        if key in miss_variant_set:
            continue

        if is_reference and non_variant_subsample_ratio < 1.0 and random() >= non_variant_subsample_ratio:
            continue

        if pre_pos == center_pos:
            dup_pos_end_flag = False
        elif pre_pos and center_pos != pre_pos:
            dup_pos_end_flag = True
        pre_pos = center_pos
        key = chrom + ":" + center_pos
        yield (int(center_pos), key, is_tumor, string, alt_info, seq, variant_type, dup_pos_end_flag, ref_center, alt_center.strip())

    subprocess_process.stdout.close()
    subprocess_process.wait()


def heapq_merge_generator_from(tumor_bin_reader_generator):
    tensor_infos_set = set()
    X = defaultdict(defaultdict)
    batch_count = 0
    tumor_keys = set()
    for tensor_infos in tumor_bin_reader_generator:
        center_pos, key, is_tumor, string, alt_info, seq, somatic_flag, dup_pos_end_flag, ref_center, alt_center = tensor_infos
        if is_tumor:
            tumor_keys.add(key)
        tensor_infos_set.add(key)
        tumor_flag = 'tumor' if is_tumor else 'normal'
        tensor_list = string.split(" ")

        if batch_count >= shuffle_bin_size and is_tumor and dup_pos_end_flag:
            yield X, batch_count
            X = defaultdict(defaultdict)
            batch_count = 1

        if tumor_flag in X[key]:
            X[key][tumor_flag].append((tensor_list, alt_info, seq, somatic_flag, ref_center, alt_center))
        else:
            X[key][tumor_flag] = [(tensor_list, alt_info, seq, somatic_flag, ref_center, alt_center)]
        batch_count += 1

    if len(X):
        yield X, batch_count


def get_training_array(args,
                       tumor_tensor_fn,
                       var_fn,
                       bed_fn,
                       bin_fn,
                       shuffle=True,
                       is_allow_duplicate_chr_pos=True,
                       chunk_id=None,
                       chunk_num=None,
                       platform='ont',
                       pileup=False,
                       maximum_non_variant_ratio=None,
                       candidate_details_fn_prefix=None,
                       merge_bins=False):
    phase_tumor = args.phase_tumor
    tree = bed_tree_from(bed_file_path=bed_fn)
    is_tree_empty = len(tree.keys()) == 0
    Y, miss_variant_set = variant_map_from(var_fn, tree, is_tree_empty)

    global param
    float_type = 'int32'
    if pileup:
        import shared.param as param
    else:
        import shared.param as param
        float_type = 'int8'

    from shared.param import pileup_channel_size
    channel_size = pileup_channel_size
    param.tumor_channel_size = param.tumor_channel_size + 16 if phase_tumor else channel_size

    tensor_shape = [param.no_of_positions, param.tumor_channel_size]  # normal and tumor
    non_variant_subsample_ratio = maximum_non_variant_ratio if maximum_non_variant_ratio is not None else 1.0

    # normal_tensor_list = []
    tumor_tensor_list = []
    if os.path.exists(tumor_tensor_fn):
        if not (os.path.exists(tumor_tensor_fn)):
            return 0
        tumor_tensor_list.append(tumor_tensor_fn)
    else:
        tumor_tensor_info = tumor_tensor_fn.split('/')
        tumor_directry, file_prefix = '/'.join(tumor_tensor_info[:-1]), tumor_tensor_info[-1]

        for file_name in os.listdir(tumor_directry):
            if file_name.startswith(file_prefix + '_') or file_name.startswith(
                    file_prefix + '.'):  # add '_.' to avoid add other prefix chr
                tumor_tensor_list.append(os.path.join(tumor_directry, file_name))

    tables.set_blosc_max_threads(64)
    int_atom = tables.Atom.from_dtype(np.dtype(float_type))
    float_atom = tables.Atom.from_dtype(np.dtype('float32'))
    string_atom = tables.StringAtom(itemsize=param.no_of_positions + 50)
    long_string_atom = tables.StringAtom(itemsize=30000)  # max alt_info length
    table_file = tables.open_file(bin_fn, mode='w', filters=FILTERS)
    table_file.create_earray(where='/', name='input_matrix', atom=float_atom, shape=[0] + tensor_shape,
                             filters=FILTERS)
    table_file.create_earray(where='/', name='position', atom=string_atom, shape=(0, 1), filters=FILTERS)
    table_file.create_earray(where='/', name='a_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='c_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='g_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='t_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='na_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='nc_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='ng_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='nt_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='tumor_alt_info', atom=long_string_atom, shape=(0, 1), filters=FILTERS)

    table_dict = update_table_dict()
    total_compressed = 0
    total = 0
    for tumor_tensor_fn in tumor_tensor_list:
        tumor_subprocess_process = subprocess_popen(shlex.split("{} -fdc {}".format(param.zstd, tumor_tensor_fn)))
        tumor_bin_reader_generator = bin_reader_generator_from(subprocess_process=tumor_subprocess_process,
                                                               Y=Y,
                                                               is_tree_empty=is_tree_empty,
                                                               tree=tree,
                                                               miss_variant_set=miss_variant_set,
                                                               is_allow_duplicate_chr_pos=is_allow_duplicate_chr_pos,
                                                               non_variant_subsample_ratio=non_variant_subsample_ratio,
                                                               is_tumor=True)

        for X, batch_total in heapq_merge_generator_from(tumor_bin_reader_generator):
            total += batch_total
            all_pos_list = get_key_list(X)

            for key, tumor_index in all_pos_list:
                infos = X[key]
                tumor_infos = infos['tumor'][tumor_index]
                tumor_tensor, tumor_alt_info, seq, variant_type, ref_center, alt_center = tumor_infos[0], tumor_infos[1], tumor_infos[2], tumor_infos[3], tumor_infos[4], tumor_infos[5]
                pos_infos = key + ':' + seq + ':' + variant_type

                if alt_center not in ['A', 'C', 'G', 'T']:
                    continue
                if alt_center == 'A':
                    # acgt_label = [1, 0, 0, 0]
                    # nacgt_label = [0, 1, 1, 1]
                    a_label = [0, 1]
                    c_label = [1, 0]
                    g_label = [1, 0]
                    t_label = [1, 0]
                    na_label = [1, 0]
                    nc_label = [0, 1]
                    ng_label = [0, 1]
                    nt_label = [0, 1]
                elif alt_center == 'C':
                    # acgt_label = [0, 1, 0, 0]
                    # nacgt_label = [1, 0, 1, 1]
                    a_label = [1, 0]
                    c_label = [0, 1]
                    g_label = [1, 0]
                    t_label = [1, 0]
                    na_label = [0, 1]
                    nc_label = [1, 0]
                    ng_label = [0, 1]
                    nt_label = [0, 1]
                elif alt_center == 'G':
                    # acgt_label = [0, 0, 1, 0]
                    # nacgt_label = [1, 1, 0, 1]
                    a_label = [1, 0]
                    c_label = [1, 0]
                    g_label = [0, 1]
                    t_label = [1, 0]
                    na_label = [0, 1]
                    nc_label = [0, 1]
                    ng_label = [1, 0]
                    nt_label = [0, 1]
                elif alt_center == 'T':
                    # acgt_label = [0, 0, 0, 1]
                    # nacgt_label = [1, 1, 1, 0]
                    a_label = [1, 0]
                    c_label = [1, 0]
                    g_label = [1, 0]
                    t_label = [0, 1]
                    na_label = [0, 1]
                    nc_label = [0, 1]
                    ng_label = [0, 1]
                    nt_label = [1, 0]

                # add af Infos here
                add_af_in_label = param.add_af_in_label
                if add_af_in_label:
                    support_alt_dict = {}
                    max_tumor_af = 0
                    tumor_depth, tumor_alt = tumor_alt_info.split('-')[:2]
                    tumor_alt = tumor_alt.split(' ')
                    tumor_alt_dict = dict(zip(tumor_alt[::2], [int(item) for item in tumor_alt[1::2]])) if len(
                        tumor_alt) else {}
                    for tumor_alt, tumor_count in tumor_alt_dict.items():
                        if tumor_alt[0] != 'X':
                            continue
                        tumor_af = tumor_count / float(tumor_depth)
                        support_alt_dict[tumor_alt] = [tumor_af]

                        max_tumor_af = max(tumor_af, max_tumor_af)

                    alt_type_list = sorted(support_alt_dict.items(), key=lambda x: x[1][0], reverse=True)
                    if len(alt_type_list) == 0:
                        print('not found', tumor_alt_dict)
                        continue
                    best_match_alt, best_af_list = alt_type_list[0]
                    tumor_af = best_af_list

                    if tumor_af is None:
                        print('skip')
                        continue
                total_compressed = write_table_dict(table_dict=table_dict,
                                                    tumor_matrix=tumor_tensor,
                                                    a_label=a_label,
                                                    c_label=c_label,
                                                    g_label=g_label,
                                                    t_label=t_label,
                                                    na_label=na_label,
                                                    nc_label=nc_label,
                                                    ng_label=ng_label,
                                                    nt_label=nt_label,
                                                    pos=pos_infos,
                                                    total=total_compressed,
                                                    tumor_alt_info=tumor_alt_info,
                                                    tensor_shape=tensor_shape,
                                                    pileup=pileup,
                                                    )

                if total_compressed % 500 == 0 and total_compressed > 0:
                    table_dict = write_table_file(table_file, table_dict, tensor_shape, float_type)

        if total_compressed % 500 != 0 and total_compressed > 0:
            table_dict = write_table_file(table_file, table_dict, tensor_shape, float_type)

    table_file.close()
    print("[INFO] Compressed %d/%d tensor" % (total_compressed, total), file=sys.stderr)


def get_training_array_indel(args,
                       tumor_tensor_fn,
                       var_fn,
                       bed_fn,
                       bin_fn,
                       shuffle=True,
                       is_allow_duplicate_chr_pos=True,
                       chunk_id=None,
                       chunk_num=None,
                       platform='ont',
                       pileup=False,
                       maximum_non_variant_ratio=None,
                       candidate_details_fn_prefix=None,
                       merge_bins=False):
    phase_tumor = args.phase_tumor
    tree = bed_tree_from(bed_file_path=bed_fn)
    is_tree_empty = len(tree.keys()) == 0
    Y, miss_variant_set = variant_map_from(var_fn, tree, is_tree_empty)

    global param
    float_type = 'int32'
    if pileup:
        import shared.param as param
    else:
        import shared.param as param
        float_type = 'int8'

    from shared.param import pileup_channel_size
    channel_size = pileup_channel_size
    param.tumor_channel_size = param.tumor_channel_size + 16 if phase_tumor else channel_size

    tensor_shape = [param.no_of_positions, param.tumor_channel_size]  # normal and tumor
    non_variant_subsample_ratio = maximum_non_variant_ratio if maximum_non_variant_ratio is not None else 1.0

    tumor_tensor_list = []
    if os.path.exists(tumor_tensor_fn):
        if not (os.path.exists(tumor_tensor_fn)):
            return 0
        tumor_tensor_list.append(tumor_tensor_fn)
    else:
        tumor_tensor_info = tumor_tensor_fn.split('/')
        tumor_directry, file_prefix = '/'.join(tumor_tensor_info[:-1]), tumor_tensor_info[-1]

        for file_name in os.listdir(tumor_directry):
            if file_name.startswith(file_prefix + '_') or file_name.startswith(
                    file_prefix + '.'):  # add '_.' to avoid add other prefix chr
                tumor_tensor_list.append(os.path.join(tumor_directry, file_name))

    tables.set_blosc_max_threads(64)
    int_atom = tables.Atom.from_dtype(np.dtype(float_type))
    float_atom = tables.Atom.from_dtype(np.dtype('float32'))
    string_atom = tables.StringAtom(itemsize=param.no_of_positions + 50)
    long_string_atom = tables.StringAtom(itemsize=30000)  # max alt_info length
    table_file = tables.open_file(bin_fn, mode='w', filters=FILTERS)
    table_file.create_earray(where='/', name='input_matrix', atom=float_atom, shape=[0] + tensor_shape,
                             filters=FILTERS)
    table_file.create_earray(where='/', name='position', atom=string_atom, shape=(0, 1), filters=FILTERS)
    table_file.create_earray(where='/', name='a_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='c_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='g_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='t_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='i_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='d_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='na_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='nc_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='ng_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='nt_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='ni_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='nd_label', atom=float_atom, shape=(0, 2), filters=FILTERS)
    table_file.create_earray(where='/', name='tumor_alt_info', atom=long_string_atom, shape=(0, 1), filters=FILTERS)

    table_dict = update_table_dict_indel()
    total_compressed = 0
    total = 0
    for tumor_tensor_fn in tumor_tensor_list:

        tumor_subprocess_process = subprocess_popen(shlex.split("{} -fdc {}".format(param.zstd, tumor_tensor_fn)))

        tumor_bin_reader_generator = bin_reader_generator_from(subprocess_process=tumor_subprocess_process,
                                                               Y=Y,
                                                               is_tree_empty=is_tree_empty,
                                                               tree=tree,
                                                               miss_variant_set=miss_variant_set,
                                                               is_allow_duplicate_chr_pos=is_allow_duplicate_chr_pos,
                                                               non_variant_subsample_ratio=non_variant_subsample_ratio,
                                                               is_tumor=True)

        for X, batch_total in heapq_merge_generator_from(tumor_bin_reader_generator):
            total += batch_total
            all_pos_list = get_key_list(X)

            for key, tumor_index in all_pos_list:
                infos = X[key]
                tumor_infos = infos['tumor'][tumor_index]

                tumor_tensor, tumor_alt_info, seq, variant_type, ref_center, alt_center = tumor_infos[0], tumor_infos[1], tumor_infos[2], tumor_infos[3], tumor_infos[4], tumor_infos[5]
                pos_infos = key + ':' + seq + ':' + variant_type

                tumor_depth, tumor_alt = tumor_alt_info.split('-')[:2]
                tumor_alt = tumor_alt.split(' ')
                if tumor_alt == ['']:
                    continue
                if variant_type == 'homo_germline' or variant_type == 'hetero_germline' or variant_type == 'homo_somatic' or variant_type == "hetero_somatic":
                    if tumor_alt[0][0] == 'I':
                        a_label = [1, 0]
                        c_label = [1, 0]
                        g_label = [1, 0]
                        t_label = [1, 0]
                        i_label = [0, 1]
                        d_label = [1, 0]
                        na_label = [0, 1]
                        nc_label = [0, 1]
                        ng_label = [0, 1]
                        nt_label = [0, 1]
                        ni_label = [1, 0]
                        nd_label = [0, 1]
                    elif tumor_alt[0][0] == 'D':
                        a_label = [1, 0]
                        c_label = [1, 0]
                        g_label = [1, 0]
                        t_label = [1, 0]
                        i_label = [1, 0]
                        d_label = [0, 1]
                        na_label = [0, 1]
                        nc_label = [0, 1]
                        ng_label = [0, 1]
                        nt_label = [0, 1]
                        ni_label = [0, 1]
                        nd_label = [1, 0]
                    else:
                        continue
                elif variant_type == 'ref':
                    if ref_center == 'A':
                        a_label = [0, 1]
                        c_label = [1, 0]
                        g_label = [1, 0]
                        t_label = [1, 0]
                        i_label = [1, 0]
                        d_label = [1, 0]
                        na_label = [1, 0]
                        nc_label = [0, 1]
                        ng_label = [0, 1]
                        nt_label = [0, 1]
                        ni_label = [0, 1]
                        nd_label = [0, 1]
                    elif ref_center == 'C':
                        a_label = [1, 0]
                        c_label = [0, 1]
                        g_label = [1, 0]
                        t_label = [1, 0]
                        i_label = [1, 0]
                        d_label = [1, 0]
                        na_label = [0, 1]
                        nc_label = [1, 0]
                        ng_label = [0, 1]
                        nt_label = [0, 1]
                        ni_label = [0, 1]
                        nd_label = [0, 1]
                    elif ref_center == 'G':
                        a_label = [1, 0]
                        c_label = [1, 0]
                        g_label = [0, 1]
                        t_label = [1, 0]
                        i_label = [1, 0]
                        d_label = [1, 0]
                        na_label = [0, 1]
                        nc_label = [0, 1]
                        ng_label = [1, 0]
                        nt_label = [0, 1]
                        ni_label = [0, 1]
                        nd_label = [0, 1]
                    elif ref_center == 'T':
                        a_label = [1, 0]
                        c_label = [1, 0]
                        g_label = [1, 0]
                        t_label = [0, 1]
                        i_label = [1, 0]
                        d_label = [1, 0]
                        na_label = [0, 1]
                        nc_label = [0, 1]
                        ng_label = [0, 1]
                        nt_label = [1, 0]
                        ni_label = [0, 1]
                        nd_label = [0, 1]
                    else:
                        continue
                else:
                    continue

                add_af_in_label = param.add_af_in_label
                if add_af_in_label:
                    support_alt_dict = {}
                    max_tumor_af = 0
                    tumor_depth, tumor_alt = tumor_alt_info.split('-')[:2]
                    tumor_alt = tumor_alt.split(' ')
                    tumor_alt_dict = dict(zip(tumor_alt[::2], [int(item) for item in tumor_alt[1::2]])) if len(
                        tumor_alt) else {}
                    for tumor_alt, tumor_count in tumor_alt_dict.items():
                        if tumor_alt[0] != 'X':
                            continue
                        tumor_af = tumor_count / float(tumor_depth)
                        support_alt_dict[tumor_alt] = [tumor_af]

                        max_tumor_af = max(tumor_af, max_tumor_af)

                    alt_type_list = sorted(support_alt_dict.items(), key=lambda x: x[1][0], reverse=True)
                    if len(alt_type_list) == 0:
                        print('not found', tumor_alt_dict)
                        continue
                    best_match_alt, best_af_list = alt_type_list[0]
                    tumor_af = best_af_list

                    if tumor_af is None:
                        print('skip')
                        continue
                total_compressed = write_table_dict_indel(table_dict=table_dict,
                                                    tumor_matrix=tumor_tensor,
                                                    a_label=a_label,
                                                    c_label=c_label,
                                                    g_label=g_label,
                                                    t_label=t_label,
                                                    i_label=i_label,
                                                    d_label=d_label,
                                                    na_label=na_label,
                                                    nc_label=nc_label,
                                                    ng_label=ng_label,
                                                    nt_label=nt_label,
                                                    ni_label=ni_label,
                                                    nd_label=nd_label,
                                                    pos=pos_infos,
                                                    total=total_compressed,
                                                    tumor_alt_info=tumor_alt_info,
                                                    tensor_shape=tensor_shape,
                                                    pileup=pileup,
                                                    )

                if total_compressed % 500 == 0 and total_compressed > 0:
                    table_dict = write_table_file_indel(table_file, table_dict, tensor_shape, float_type)

        if total_compressed % 500 != 0 and total_compressed > 0:
            table_dict = write_table_file_indel(table_file, table_dict, tensor_shape, float_type)

    table_file.close()
    print("[INFO] Compressed %d/%d tensor" % (total_compressed, total), file=sys.stderr)
