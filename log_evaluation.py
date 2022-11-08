import numpy as np
from scipy import signal as sig


def eval_along_y(ref_data, trajectory_y):
    my_inds = (trajectory_y + 0.5).astype(int)
    old_data = ref_data[my_inds]
    i0s = np.floor(trajectory_y).astype(int)
    i1s = i0s + 1
    # indexes = torch.max(indexes, 0)
    # indexes = torch.min(indexes, self.input_ref_size-1)
    curves_values0 = ref_data[i0s]
    curves_values1 = ref_data[i1s]
    dists0 = trajectory_y - i0s
    dists1 = i1s - trajectory_y
    curves_values = dists1 * curves_values0 + dists0 * curves_values1
    # todo add noize
    # my_inds = min(my_inds, self.ref_len-1)
    # my_inds = max(my_inds, 0)
    return curves_values


def eval_along_y_with_noize(ref_data, trajectory_y, noize_std=None, noize_rel_std=0.01):
    my_inds = (trajectory_y + 0.5).astype(int)
    old_data = ref_data[my_inds]
    i0s = np.floor(trajectory_y).astype(int)
    i1s = i0s + 1
    curves_values0 = ref_data[i0s]
    curves_values1 = ref_data[i1s]
    dists0 = trajectory_y - i0s
    dists1 = i1s - trajectory_y
    if noize_std is None:
        min_data = np.min(ref_data)
        max_data = np.max(ref_data)
        data_range = max_data - min_data
        noize_std = data_range * noize_rel_std
        # print('Computed noize std ', noize_std)

    # correlated noise
    # x_for_corr = np.arange(curves_values0.size).reshape(curves_values0.size, 1) / curves_values0.size
    # dist = scipy.spatial.distance.pdist(x_for_corr)
    # dist = scipy.spatial.distance.squareform(dist)
    # correlation_scale = 1  # harf coded
    # cov = np.exp(-dist ** 2 / (2 * correlation_scale))
    # noise = np.random.multivariate_normal(0*curves_values0, cov)

    # fast correlated noise
    # correlation_scale = curves_values0.size // 8
    correlation_scale = 8
    dist = np.arange(-correlation_scale, correlation_scale)
    noise = np.random.normal(scale=noize_std, size=curves_values0.size)
    filter_kernel = np.exp(-dist ** 2 / (2 * correlation_scale))
    noise_correlated = sig.fftconvolve(noise, filter_kernel, mode='same')

    # plt.figure()
    # plt.plot(noise)
    # plt.plot(noise_correlated)
    # plt.show()
    # exit()
    curves_values = dists1 * curves_values0 + dists0 * curves_values1 + noise_correlated

    return curves_values


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ref_x = np.arange(0, 100) / 50

    # some offset log
    ref_data = np.sin(np.arange(0, 1000))

    # generating some curve for stratigraphy
    trajectory_y = np.minimum(3., np.power(ref_x, 2))

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax1.plot(trajectory_y)

    ax2 = plt.subplot2grid((2, 1), (1, 0))

    curves_values = eval_along_y(ref_data, trajectory_y)
    curves_values_with_noize = eval_along_y_with_noize(ref_data, trajectory_y)

    ax2.plot(curves_values)
    ax2.plot(curves_values_with_noize)

    plt.show()
