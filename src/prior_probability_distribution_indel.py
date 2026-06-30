
from argparse import ArgumentParser, SUPPRESS
import numpy as np

def calculate_prior_probability_distribution(args):

    st_predict_file = args.input_st_predict_file
    st_prior_file = args.output_st_prior_file
    n = args.n_bins

    st_predict_data = open(st_predict_file, 'r')

    p_a_na_list = []
    p_c_nc_list = []
    p_g_ng_list = []
    p_t_nt_list = []
    p_i_ni_list = []
    p_d_nd_list = []
    ref_base_list = []
    tumor_alt_info_list = []
    variant_type_list = []

    for line in st_predict_data.readlines():
        line = line.rstrip().split('\t')
        chromosome, position, reference_base, tumor_alt_info, _, _, prediction_a, prediction_c, prediction_g, prediction_t, prediction_i, prediction_d, \
            prediction_na, prediction_nc, prediction_ng, prediction_nt, prediction_ni, prediction_nd = line[:18]
        # chromosome, position, reference_base, tumor_alt_info, prediction_a, prediction_c, prediction_g, prediction_t, prediction_i, prediction_d, \
        #     prediction_na, prediction_nc, prediction_ng, prediction_nt, prediction_ni, prediction_nd = line[:16]
        probabilities_a = [float(item) for item in prediction_a.split()]
        probabilities_c = [float(item) for item in prediction_c.split()]
        probabilities_g = [float(item) for item in prediction_g.split()]
        probabilities_t = [float(item) for item in prediction_t.split()]
        probabilities_i = [float(item) for item in prediction_i.split()]
        probabilities_d = [float(item) for item in prediction_d.split()]
        probabilities_na = [float(item) for item in prediction_na.split()]
        probabilities_nc = [float(item) for item in prediction_nc.split()]
        probabilities_ng = [float(item) for item in prediction_ng.split()]
        probabilities_nt = [float(item) for item in prediction_nt.split()]
        probabilities_ni = [float(item) for item in prediction_ni.split()]
        probabilities_nd = [float(item) for item in prediction_nd.split()]
        probabilities_is_a = probabilities_a[1]
        probabilities_is_c = probabilities_c[1]
        probabilities_is_g = probabilities_g[1]
        probabilities_is_t = probabilities_t[1]
        probabilities_is_i = probabilities_i[1]
        probabilities_is_d = probabilities_d[1]
        probabilities_is_na = probabilities_na[0]
        probabilities_is_nc = probabilities_nc[0]
        probabilities_is_ng = probabilities_ng[0]
        probabilities_is_nt = probabilities_nt[0]
        probabilities_is_ni = probabilities_ni[0]
        probabilities_is_nd = probabilities_nd[0]

        p_a_na_list.append([probabilities_is_a, probabilities_is_na])
        p_c_nc_list.append([probabilities_is_c, probabilities_is_nc])
        p_g_ng_list.append([probabilities_is_g, probabilities_is_ng])
        p_t_nt_list.append([probabilities_is_t, probabilities_is_nt])
        p_i_ni_list.append([probabilities_is_i, probabilities_is_ni])
        p_d_nd_list.append([probabilities_is_d, probabilities_is_nd])

        ref_base = line[22]
        # ref_base = line[21]
        ref_base_list.append(ref_base)
        tumor_alt_info_list.append(tumor_alt_info)
        variant_type = line[21]
        # variant_type = line[20]
        variant_type_list.append(variant_type)

    p_a_na_matrix = np.array(p_a_na_list)
    p_c_nc_matrix = np.array(p_c_nc_list)
    p_g_ng_matrix = np.array(p_g_ng_list)
    p_t_nt_matrix = np.array(p_t_nt_list)
    p_i_ni_matrix = np.array(p_i_ni_list)
    p_d_nd_matrix = np.array(p_d_nd_list)

    p_a_matrix = p_a_na_matrix[:, 0]
    p_na_matrix = p_a_na_matrix[:, 1]
    sorted_p_a_matrix = np.sort(p_a_matrix)
    sorted_p_na_matrix = np.sort(p_na_matrix)

    p_c_matrix = p_c_nc_matrix[:, 0]
    p_nc_matrix = p_c_nc_matrix[:, 1]
    sorted_p_c_matrix = np.sort(p_c_matrix)
    sorted_p_nc_matrix = np.sort(p_nc_matrix)

    p_g_matrix = p_g_ng_matrix[:, 0]
    p_ng_matrix = p_g_ng_matrix[:, 1]
    sorted_p_g_matrix = np.sort(p_g_matrix)
    sorted_p_ng_matrix = np.sort(p_ng_matrix)

    p_t_matrix = p_t_nt_matrix[:, 0]
    p_nt_matrix = p_t_nt_matrix[:, 1]
    sorted_p_t_matrix = np.sort(p_t_matrix)
    sorted_p_nt_matrix = np.sort(p_nt_matrix)

    p_i_matrix = p_i_ni_matrix[:, 0]
    p_ni_matrix = p_i_ni_matrix[:, 1]
    sorted_p_i_matrix = np.sort(p_i_matrix)
    sorted_p_ni_matrix = np.sort(p_ni_matrix)

    p_d_matrix = p_d_nd_matrix[:, 0]
    p_nd_matrix = p_d_nd_matrix[:, 1]
    sorted_p_d_matrix = np.sort(p_d_matrix)
    sorted_p_nd_matrix = np.sort(p_nd_matrix)

    num_splits = 10

    split_p_a_matrix_indices = np.linspace(0, len(sorted_p_a_matrix), num=num_splits+1, dtype=int)
    split_p_na_matrix_indices = np.linspace(0, len(sorted_p_na_matrix), num=num_splits+1, dtype=int)

    split_p_c_matrix_indices = np.linspace(0, len(sorted_p_c_matrix), num=num_splits+1, dtype=int)
    split_p_nc_matrix_indices = np.linspace(0, len(sorted_p_nc_matrix), num=num_splits+1, dtype=int)

    split_p_g_matrix_indices = np.linspace(0, len(sorted_p_g_matrix), num=num_splits+1, dtype=int)
    split_p_ng_matrix_indices = np.linspace(0, len(sorted_p_ng_matrix), num=num_splits+1, dtype=int)

    split_p_t_matrix_indices = np.linspace(0, len(sorted_p_t_matrix), num=num_splits+1, dtype=int)
    split_p_nt_matrix_indices = np.linspace(0, len(sorted_p_nt_matrix), num=num_splits+1, dtype=int)

    split_p_i_matrix_indices = np.linspace(0, len(sorted_p_i_matrix), num=num_splits+1, dtype=int)
    split_p_ni_matrix_indices = np.linspace(0, len(sorted_p_ni_matrix), num=num_splits+1, dtype=int)

    split_p_d_matrix_indices = np.linspace(0, len(sorted_p_d_matrix), num=num_splits+1, dtype=int)
    split_p_nd_matrix_indices = np.linspace(0, len(sorted_p_nd_matrix), num=num_splits+1, dtype=int)

    split_p_a_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_a_matrix_indices[i]
        end_index = split_p_a_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_a_matrix[start_index:end_index])
        split_point_value = sorted_p_a_matrix[split_point_index]
        if i == 0:
            split_p_a_matrix_points.append(0)
        split_p_a_matrix_points.append(split_point_value)

    split_p_na_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_na_matrix_indices[i]
        end_index = split_p_na_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_na_matrix[start_index:end_index])
        split_point_value = sorted_p_na_matrix[split_point_index]
        if i == 0:
            split_p_na_matrix_points.append(0)
        split_p_na_matrix_points.append(split_point_value)

    split_p_c_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_c_matrix_indices[i]
        end_index = split_p_c_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_c_matrix[start_index:end_index])
        split_point_value = sorted_p_c_matrix[split_point_index]
        if i == 0:
            split_p_c_matrix_points.append(0)
        split_p_c_matrix_points.append(split_point_value)

    split_p_nc_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_nc_matrix_indices[i]
        end_index = split_p_nc_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_nc_matrix[start_index:end_index])
        split_point_value = sorted_p_nc_matrix[split_point_index]
        if i == 0:
            split_p_nc_matrix_points.append(0)
        split_p_nc_matrix_points.append(split_point_value)

    split_p_g_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_g_matrix_indices[i]
        end_index = split_p_g_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_g_matrix[start_index:end_index])
        split_point_value = sorted_p_g_matrix[split_point_index]
        if i == 0:
            split_p_g_matrix_points.append(0)
        split_p_g_matrix_points.append(split_point_value)

    split_p_ng_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_ng_matrix_indices[i]
        end_index = split_p_ng_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_ng_matrix[start_index:end_index])
        split_point_value = sorted_p_ng_matrix[split_point_index]
        if i == 0:
            split_p_ng_matrix_points.append(0)
        split_p_ng_matrix_points.append(split_point_value)

    split_p_t_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_t_matrix_indices[i]
        end_index = split_p_t_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_t_matrix[start_index:end_index])
        split_point_value = sorted_p_t_matrix[split_point_index]
        if i == 0:
            split_p_t_matrix_points.append(0)
        split_p_t_matrix_points.append(split_point_value)

    split_p_nt_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_nt_matrix_indices[i]
        end_index = split_p_nt_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_nt_matrix[start_index:end_index])
        split_point_value = sorted_p_nt_matrix[split_point_index]
        if i == 0:
            split_p_nt_matrix_points.append(0)
        split_p_nt_matrix_points.append(split_point_value)

    split_p_i_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_i_matrix_indices[i]
        end_index = split_p_i_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_i_matrix[start_index:end_index])
        split_point_value = sorted_p_i_matrix[split_point_index]
        if i == 0:
            split_p_i_matrix_points.append(0)
        split_p_i_matrix_points.append(split_point_value)

    split_p_ni_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_ni_matrix_indices[i]
        end_index = split_p_ni_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_ni_matrix[start_index:end_index])
        split_point_value = sorted_p_ni_matrix[split_point_index]
        if i == 0:
            split_p_ni_matrix_points.append(0)
        split_p_ni_matrix_points.append(split_point_value)

    split_p_d_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_d_matrix_indices[i]
        end_index = split_p_d_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_d_matrix[start_index:end_index])
        split_point_value = sorted_p_d_matrix[split_point_index]
        if i == 0:
            split_p_d_matrix_points.append(0)
        split_p_d_matrix_points.append(split_point_value)

    split_p_nd_matrix_points = []
    for i in range(num_splits):
        start_index = split_p_nd_matrix_indices[i]
        end_index = split_p_nd_matrix_indices[i+1]
        split_point_index = start_index + np.argmax(sorted_p_nd_matrix[start_index:end_index])
        split_point_value = sorted_p_nd_matrix[split_point_index]
        if i == 0:
            split_p_nd_matrix_points.append(0)
        split_p_nd_matrix_points.append(split_point_value)

    split_p_a_points = np.expand_dims(np.array(split_p_a_matrix_points[1:]), axis=0)
    split_p_na_points = np.expand_dims(np.array(split_p_na_matrix_points[1:]), axis=0)

    split_p_c_points = np.expand_dims(np.array(split_p_c_matrix_points[1:]), axis=0)
    split_p_nc_points = np.expand_dims(np.array(split_p_nc_matrix_points[1:]), axis=0)

    split_p_g_points = np.expand_dims(np.array(split_p_g_matrix_points[1:]), axis=0)
    split_p_ng_points = np.expand_dims(np.array(split_p_ng_matrix_points[1:]), axis=0)

    split_p_t_points = np.expand_dims(np.array(split_p_t_matrix_points[1:]), axis=0)
    split_p_nt_points = np.expand_dims(np.array(split_p_nt_matrix_points[1:]), axis=0)

    split_p_i_points = np.expand_dims(np.array(split_p_i_matrix_points[1:]), axis=0)
    split_p_ni_points = np.expand_dims(np.array(split_p_ni_matrix_points[1:]), axis=0)

    split_p_d_points = np.expand_dims(np.array(split_p_d_matrix_points[1:]), axis=0)
    split_p_nd_points = np.expand_dims(np.array(split_p_nd_matrix_points[1:]), axis=0)

    conditional_probs_a_na = np.zeros((10, 10))
    conditional_probs_c_nc = np.zeros((10, 10))
    conditional_probs_g_ng = np.zeros((10, 10))
    conditional_probs_t_nt = np.zeros((10, 10))
    conditional_probs_i_ni = np.zeros((10, 10))
    conditional_probs_d_nd = np.zeros((10, 10))

    for i in range(10):
        for j in range(10):
            count_a = 0
            count_a_na = 0
            count_c = 0
            count_c_nc = 0
            count_g = 0
            count_g_ng = 0
            count_t = 0
            count_t_nt = 0
            count_i = 0
            count_i_ni = 0
            count_d = 0
            count_d_nd = 0
            for k in range(len(p_a_na_matrix)):
                if p_a_na_matrix[k, 0] > split_p_a_matrix_points[i] and p_a_na_matrix[k, 0] <= split_p_a_matrix_points[i+1]:
                    if p_a_na_matrix[k, 1] > split_p_na_matrix_points[j] and p_a_na_matrix[k, 1] <=split_p_na_matrix_points[j + 1]:
                        count_a_na += 1
                        if variant_type_list[k] == 'ref' and ref_base_list[k] == 'A':
                            count_a += 1
                if p_c_nc_matrix[k, 0] > split_p_c_matrix_points[i] and p_c_nc_matrix[k, 0] <= split_p_c_matrix_points[i+1]:
                    if p_c_nc_matrix[k, 1] > split_p_nc_matrix_points[j] and p_c_nc_matrix[k, 1] <=split_p_nc_matrix_points[j + 1]:
                        count_c_nc += 1
                        if variant_type_list[k] == 'ref' and ref_base_list[k] == 'C':
                            count_c += 1
                if p_g_ng_matrix[k, 0] > split_p_g_matrix_points[i] and p_g_ng_matrix[k, 0] <= split_p_g_matrix_points[i+1]:
                    if p_g_ng_matrix[k, 1] > split_p_ng_matrix_points[j] and p_g_ng_matrix[k, 1] <=split_p_ng_matrix_points[j + 1]:
                        count_g_ng += 1
                        if variant_type_list[k] == 'ref' and ref_base_list[k] == 'G':
                            count_g += 1
                if p_t_nt_matrix[k, 0] > split_p_t_matrix_points[i] and p_t_nt_matrix[k, 0] <= split_p_t_matrix_points[i+1]:
                    if p_t_nt_matrix[k, 1] > split_p_nt_matrix_points[j] and p_t_nt_matrix[k, 1] <=split_p_nt_matrix_points[j + 1]:
                        count_t_nt += 1
                        if variant_type_list[k] == 'ref' and ref_base_list[k] == 'T':
                            count_t += 1
                tumor_depth, tumor_alt = tumor_alt_info_list[k].split('-')[:2]
                tumor_alt = tumor_alt.split(' ')
                if p_i_ni_matrix[k, 0] > split_p_i_matrix_points[i] and p_i_ni_matrix[k, 0] <= split_p_i_matrix_points[i+1]:
                    if p_i_ni_matrix[k, 1] > split_p_ni_matrix_points[j] and p_i_ni_matrix[k, 1] <=split_p_ni_matrix_points[j + 1]:
                        count_i_ni += 1
                        if (variant_type_list[k] == 'homo_germline' or variant_type_list[k] == 'hetero_germline' or variant_type_list[k] == 'homo_somatic' or variant_type_list[k] == "hetero_somatic") and tumor_alt[0][0] == 'I':
                            count_i += 1
                if p_d_nd_matrix[k, 0] > split_p_d_matrix_points[i] and p_d_nd_matrix[k, 0] <= split_p_d_matrix_points[i+1]:
                    if p_d_nd_matrix[k, 1] > split_p_nd_matrix_points[j] and p_d_nd_matrix[k, 1] <=split_p_nd_matrix_points[j + 1]:
                        count_d_nd += 1
                        if (variant_type_list[k] == 'homo_germline' or variant_type_list[k] == 'hetero_germline' or variant_type_list[k] == 'homo_somatic' or variant_type_list[k] == "hetero_somatic") and tumor_alt[0][0] == 'D':
                            count_d += 1

            conditional_probs_a_na[i, j] = count_a / count_a_na if count_a_na > 0 else 0.0
            conditional_probs_c_nc[i, j] = count_c / count_c_nc if count_c_nc > 0 else 0.0
            conditional_probs_g_ng[i, j] = count_g / count_g_ng if count_g_ng > 0 else 0.0
            conditional_probs_t_nt[i, j] = count_t / count_t_nt if count_t_nt > 0 else 0.0
            conditional_probs_i_ni[i, j] = count_i / count_i_ni if count_i_ni > 0 else 0.0
            conditional_probs_d_nd[i, j] = count_d / count_d_nd if count_d_nd > 0 else 0.0

    conditional_probs = np.concatenate((conditional_probs_a_na, conditional_probs_c_nc, conditional_probs_g_ng, conditional_probs_t_nt, conditional_probs_i_ni, conditional_probs_d_nd,
                                        split_p_a_points, split_p_na_points, split_p_c_points, split_p_nc_points,
                                        split_p_g_points, split_p_ng_points, split_p_t_points, split_p_nt_points, split_p_i_points, split_p_ni_points, split_p_d_points, split_p_nd_points), axis=0)

    np.savetxt(st_prior_file, conditional_probs, delimiter=' ')


def main():
    parser = ArgumentParser(description="Calculate prior probability distribution for SNV calling")

    parser.add_argument('--input_st_predict_file', type=str, default=None,
                        help="Input of network output probabilities of training data")

    parser.add_argument('--output_st_prior_file', type=str, default=None,
                        help="Output of prior probability distribution of training data")

    parser.add_argument('--n_bins', type=int, default=10,
                        help="Bins number of training data")

    args = parser.parse_args()

    calculate_prior_probability_distribution(args)

if __name__ == '__main__':
    main()