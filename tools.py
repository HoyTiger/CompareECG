"""
工具封装
"""

import numpy as np
import math
import scipy.signal as sps


class Processor(object):

    def __init__(self, ecg, fs):
        self.fs = fs
        self.ecg = ecg
        self.der_i = None

    @staticmethod
    def der(sig, fs):
        """
        对指定信号进行求导
        求导公式
        y(n) = k * [x(i)+2*x(i-1)-2*x(i-3)-x(i-4)]
        k = fs // 8
        x(i)不存在时,使用0进行替换
        :param sig:
        :param fs:
        :return:
        """
        k = fs // 8
        result = [fs * (sig[1] - sig[0]),
                  fs / 4 * (2 * sig[2] - sig[0])]
        for _ in range(4, len(sig)):
            result.append(k * (sig[_] + 2 * sig[_ - 1] - 2 * sig[_ - 3] - sig[_ - 4]))
        result.append(k * (2 * sig[-1] - 2 * sig[-2] - sig[-3]))
        result.append(k * (2 * sig[-1] - sig[-2]))
        return np.array(result)


def smooth_avg2(in_put, radius):
    """
    均值滤波2
    中心值减去其前后半径内的均值作为中心值
    :param in_put: 平滑前的数据,一维数组
    :param radius: 平滑半径
    :return:
    """
    output = []
    for _ in range(len(in_put)):
        start = max(_ - radius, 0)
        end = min(_ + radius, len(in_put) - 1)
        output.append(in_put[_] - (np.sum(in_put[start:end + 1]) - in_put[_]) / (end - start))
    return np.array(output)


def denoising_step(sample, H=5, dn1=10, dn2=1):
    sample = np.array(sample)
    result = []
    c_1 = -1 * 2.0 * dn1 ** 2
    c_2 = -1 * 2.0 * dn2 ** 2
    for index in range(len(sample)):
        start_index = max(0, index - H)
        end_index = min(len(sample), index + H)
        sum_1 = 0
        sum_2 = 0
        for inner_idx in range(start_index, end_index):
            idx1 = (inner_idx - index) ** 2.0 / c_1
            idx2 = (sample[inner_idx] - sample[index]) ** 2.0 / c_2
            weight = math.exp(idx1 + idx2)
            sum_1 += (sample[inner_idx] * weight)
            sum_2 += weight
        result.append(sum_1 / sum_2)
    return np.array(result)


def sig_fragment(sig, window_len, step):
    """
    对数据进行切段，不足的片段将进行对称扩展
    :param sig:
    :param window_len:
    :param step:
    :return:
    """
    cycles = int(math.ceil((len(sig) - window_len) / step)) + 1
    result = []
    if cycles <= 0:
        if len(sig) > 0:
            result.append(extend_mirror(sig, window_len))
    else:
        for c in range(cycles):
            result.append(extend_mirror(sig[c * step:c * step + window_len], window_len))
    return np.array(result)


def extend_mirror(sig, target_len):
    sig = np.array(sig)
    if len(sig) > target_len:
        return sig[:target_len]
    elif len(sig) == target_len:
        return sig
    else:
        cycles = int(math.ceil(math.log(target_len / len(sig), 2)))
        result = sig.copy()
        for c in range(cycles):
            result = np.append(result, result[::])
        return result[:target_len]


def ecg_findpeaks_rodrigues(signal, sampling_rate=1000):
    # 原生代码存在一些bug
    """Segmenter by Tiago Rodrigues, inspired by on Gutierrez-Rivas (2015) and Sadhukhan (2012).

    References
    ----------
    - Gutiérrez-Rivas, R., García, J. J., Marnane, W. P., & Hernández, A. (2015). Novel real-time
      low-complexity QRS complex detector based on adaptive thresholding. IEEE Sensors Journal,
      15(10), 6036-6043.

    - Sadhukhan, D., & Mitra, M. (2012). R-peak detection algorithm for ECG using double difference
      and RR interval processing. Procedia Technology, 4, 873-877.

    """

    N = int(np.round(3 * sampling_rate / 128))
    Nd = N - 1
    Pth = (0.7 * sampling_rate) / 128 + 2.7
    # Pth = 3, optimal for fs = 250 Hz
    Rmin = 0.26

    rpeaks = []
    i = 1
    Ramptotal = 0

    # Double derivative squared
    diff_ecg = [signal[i] - signal[i - Nd] for i in range(Nd, len(signal))]
    ddiff_ecg = [diff_ecg[i] - diff_ecg[i - 1] for i in range(1, len(diff_ecg))]
    squar = np.square(ddiff_ecg)

    # Integrate moving window
    b = np.array(np.ones(N))
    a = [1]
    processed_ecg = sps.lfilter(b, a, squar)
    tf = len(processed_ecg)

    # R-peak finder FSM
    while i < tf:  # ignore last second of recording
        # State 1: looking for maximum
        tf1 = np.round(i + Rmin * sampling_rate)
        Rpeakamp = 0
        rpeakpos = i
        while i < tf1 and i < tf:
            # Rpeak amplitude and position
            if processed_ecg[i] > Rpeakamp:
                Rpeakamp = processed_ecg[i]
                rpeakpos = i + 1
            i += 1

        Ramptotal = (19 / 20) * Ramptotal + (1 / 20) * Rpeakamp
        rpeaks.append(rpeakpos)

        # State 2: waiting state
        d = tf1 - rpeakpos
        tf2 = i + np.round(0.2 * 2 - d)
        while i <= tf2:
            i += 1

        # State 3: decreasing threshold
        Thr = Ramptotal
        while i < tf and processed_ecg[i] < Thr:
            Thr *= np.exp(-Pth / sampling_rate)
            i += 1

    rpeaks = np.array(rpeaks, dtype="int")
    return rpeaks
