
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
    alt_base_list = []

    for line in st_predict_data.readlines():
        line = line.rstrip().split('\t')
        chromosome, position, reference_base, tumor_alt_info, _, _, prediction_a, prediction_c, prediction_g, prediction_t, \
            prediction_na, prediction_nc, prediction_ng, prediction_nt = line[:14]
        # chromosome, position, reference_base, tumor_alt_info, prediction_a, prediction_c, prediction_g, prediction_t, \
        #     prediction_na, prediction_nc, prediction_ng, prediction_nt = line[:12]
        probabilities_a = [float(item) for item in prediction_a.split()]
        probabilities_c = [float(item) for item in prediction_c.split()]
        probabilities_g = [float(item) for item in prediction_g.split()]
        probabilities_t = [float(item) for item in prediction_t.split()]
        probabilities_na = [float(item) for item in prediction_na.split()]
        probabilities_nc = [float(item) for item in prediction_nc.split()]
        probabilities_ng = [float(item) for item in prediction_ng.split()]
        probabilities_nt = [float(item) for item in prediction_nt.split()]
        probabilities_is_a = probabilities_a[1]
        probabilities_is_c = probabilities_c[1]
        probabilities_is_g = probabilities_g[1]
        probabilities_is_t = probabilities_t[1]
        probabilities_is_na = probabilities_na[0]
        probabilities_is_nc = probabilities_nc[0]
        probabilities_is_ng = probabilities_ng[0]
        probabilities_is_nt = probabilities_nt[0]

        p_a_na_list.append([probabilities_is_a, probabilities_is_na])
        p_c_nc_list.append([probabilities_is_c, probabilities_is_nc])
        p_g_ng_list.append([probabilities_is_g, probabilities_is_ng])
        p_t_nt_list.append([probabilities_is_t, probabilities_is_nt])

        alt_base = line[20]
        # alt_base = line[18]
        alt_base_list.append(alt_base)

    p_a_na_matrix = np.array(p_a_na_list)
    p_c_nc_matrix = np.array(p_c_nc_list)
    p_g_ng_matrix = np.array(p_g_ng_list)
    p_t_nt_matrix = np.array(p_t_nt_list)

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

    num_splits = 10

    split_p_a_matrix_indices = np.linspace(0, len(sorted_p_a_matrix), num=num_splits+1, dtype=int)
    split_p_na_matrix_indices = np.linspace(0, len(sorted_p_na_matrix), num=num_splits+1, dtype=int)

    split_p_c_matrix_indices = np.linspace(0, len(sorted_p_c_matrix), num=num_splits+1, dtype=int)
    split_p_nc_matrix_indices = np.linspace(0, len(sorted_p_nc_matrix), num=num_splits+1, dtype=int)

    split_p_g_matrix_indices = np.linspace(0, len(sorted_p_g_matrix), num=num_splits+1, dtype=int)
    split_p_ng_matrix_indices = np.linspace(0, len(sorted_p_ng_matrix), num=num_splits+1, dtype=int)

    split_p_t_matrix_indices = np.linspace(0, len(sorted_p_t_matrix), num=num_splits+1, dtype=int)
    split_p_nt_matrix_indices = np.linspace(0, len(sorted_p_nt_matrix), num=num_splits+1, dtype=int)

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

    split_p_a_points = np.expand_dims(np.array(split_p_a_matrix_points[1:]), axis=0)
    split_p_na_points = np.expand_dims(np.array(split_p_na_matrix_points[1:]), axis=0)

    split_p_c_points = np.expand_dims(np.array(split_p_c_matrix_points[1:]), axis=0)
    split_p_nc_points = np.expand_dims(np.array(split_p_nc_matrix_points[1:]), axis=0)

    split_p_g_points = np.expand_dims(np.array(split_p_g_matrix_points[1:]), axis=0)
    split_p_ng_points = np.expand_dims(np.array(split_p_ng_matrix_points[1:]), axis=0)

    split_p_t_points = np.expand_dims(np.array(split_p_t_matrix_points[1:]), axis=0)
    split_p_nt_points = np.expand_dims(np.array(split_p_nt_matrix_points[1:]), axis=0)

    conditional_probs_a_na = np.zeros((n, n))
    conditional_probs_c_nc = np.zeros((n, n))
    conditional_probs_g_ng = np.zeros((n, n))
    conditional_probs_t_nt = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            count_a = 0
            count_a_na = 0
            count_c = 0
            count_c_nc = 0
            count_g = 0
            count_g_ng = 0
            count_t = 0
            count_t_nt = 0
            for k in range(len(p_a_na_matrix)):
                if p_a_na_matrix[k, 0] > split_p_a_matrix_points[i] and p_a_na_matrix[k, 0] <= split_p_a_matrix_points[i+1]:
                    if p_a_na_matrix[k, 1] > split_p_na_matrix_points[j] and p_a_na_matrix[k, 1] <=split_p_na_matrix_points[j + 1]:
                        count_a_na += 1
                        if alt_base_list[k] == 'A':
                            count_a += 1
                if p_c_nc_matrix[k, 0] > split_p_c_matrix_points[i] and p_c_nc_matrix[k, 0] <= split_p_c_matrix_points[i+1]:
                    if p_c_nc_matrix[k, 1] > split_p_nc_matrix_points[j] and p_c_nc_matrix[k, 1] <=split_p_nc_matrix_points[j + 1]:
                        count_c_nc += 1
                        if alt_base_list[k] == 'C':
                            count_c += 1
                if p_g_ng_matrix[k, 0] > split_p_g_matrix_points[i] and p_g_ng_matrix[k, 0] <= split_p_g_matrix_points[i+1]:
                    if p_g_ng_matrix[k, 1] > split_p_ng_matrix_points[j] and p_g_ng_matrix[k, 1] <=split_p_ng_matrix_points[j + 1]:
                        count_g_ng += 1
                        if alt_base_list[k] == 'G':
                            count_g += 1
                if p_t_nt_matrix[k, 0] > split_p_t_matrix_points[i] and p_t_nt_matrix[k, 0] <= split_p_t_matrix_points[i+1]:
                    if p_t_nt_matrix[k, 1] > split_p_nt_matrix_points[j] and p_t_nt_matrix[k, 1] <=split_p_nt_matrix_points[j + 1]:
                        count_t_nt += 1
                        if alt_base_list[k] == 'T':
                            count_t += 1

            conditional_probs_a_na[i, j] = count_a / count_a_na if count_a_na > 0 else 0.0
            conditional_probs_c_nc[i, j] = count_c / count_c_nc if count_c_nc > 0 else 0.0
            conditional_probs_g_ng[i, j] = count_g / count_g_ng if count_g_ng > 0 else 0.0
            conditional_probs_t_nt[i, j] = count_t / count_t_nt if count_t_nt > 0 else 0.0

    conditional_probs = np.concatenate((conditional_probs_a_na, conditional_probs_c_nc, conditional_probs_g_ng, conditional_probs_t_nt,
                                        split_p_a_points, split_p_na_points, split_p_c_points, split_p_nc_points,
                                        split_p_g_points, split_p_ng_points, split_p_t_points, split_p_nt_points), axis=0)

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