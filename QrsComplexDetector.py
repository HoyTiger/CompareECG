import numpy as np
import pandas as pd
import biosppy.signals.ecg as bsp_ecg
import biosppy.signals.tools as bsp_tools
from pywt import waverec, wavedec
from scipy.interpolate import interp1d
import math
import neurokit2 as nk

from tools import Processor, sig_fragment, smooth_avg2, ecg_findpeaks_rodrigues


class QrsDetector(object):
    """
    qrs波群检测器
    """

    def __init__(self, sig, fs, adc_gain=1., adc_zero=0.):
        self.fs = fs
        self.adc_gain = adc_gain
        self.adc_zero = adc_zero
        self.sig = (np.array(sig) - adc_zero) / adc_gain

    def detect_v1(self, cut_off_high=20, cut_off_low=5,
                  th_gain_1=0.2, th_gain_2=0.28, tol_1=0.004, tol_2=0.3):
        fixed_fs = 500
        resample_sig = resample_sig_v2(self.sig, self.fs, fixed_fs)
        # cheby2型带通滤波器
        filtered_2, _, _ = bsp_tools.filter_signal(signal=resample_sig,
                                                   ftype='cheby2',
                                                   band='bandpass',
                                                   order=4,
                                                   frequency=[cut_off_low, cut_off_high],
                                                   sampling_rate=fixed_fs, rs=10)
        filtered_2 = simple_filter(filtered_2, fixed_fs)

        filtered_1 = sig_preprocess(filtered_2, fixed_fs)
        filtered_1 = simple_filter(filtered_1, fixed_fs)

        try:
            qrs_, = bsp_ecg.gamboa_segmenter(signal=filtered_2, sampling_rate=fixed_fs, tol=tol_1)
            qrs, = bsp_ecg.correct_rpeaks(signal=filtered_2, rpeaks=qrs_, sampling_rate=fixed_fs, tol=tol_2)

            # 通过检测位置附近的振幅进一步减少qrs波数量
            amplitude_1 = []
            for _ in qrs:
                start = max(0, int(_ - 0.1 * fixed_fs))
                end = min(len(resample_sig), int(_ + 0.1 * fixed_fs))
                amplitude_1.append(np.max(filtered_2[start:end]) - np.min(filtered_2[start:end]))

            target_1 = np.sort(np.array(amplitude_1))
            mean_1 = np.mean(target_1[int(len(target_1) * 0.5):int(len(target_1) * 1)])

            qrs_1 = []
            for idx, _ in enumerate(qrs):
                if amplitude_1[idx] >= th_gain_1 * mean_1:
                    qrs_1.append(_)

            amplitude_2 = []
            for _ in qrs_1:
                start = max(0, int(_ - 0.1 * fixed_fs))
                end = min(len(resample_sig), int(_ + 0.1 * fixed_fs))
                amplitude_2.append(np.max(filtered_1[start:end]) - np.min(filtered_1[start:end]))

            target_2 = np.sort(np.array(amplitude_2))
            mean_2 = np.mean(target_2[int(len(target_2) * 0.2):int(len(target_2) * 1)])
            qrs_2 = []
            for idx, _ in enumerate(qrs_1):
                if amplitude_2[idx] >= th_gain_2 * mean_2:
                    qrs_2.append(_)

            # 归并差值在一定范围内的qrs波
            qrs_3 = np.sort(np.array(qrs_2))
            diff = np.diff(qrs_3)
            diff = np.where(diff <= 0.2 * fixed_fs)[0]
            if len(diff) > 0:
                diff_ = diff + 1
                diff = np.concatenate((diff, diff_))
                merge = []
                for idx in diff_:
                    previous_idx = qrs_3[idx - 1]
                    next_idx = qrs_3[idx]
                    merge.append(previous_idx if filtered_2[previous_idx] >= filtered_2[next_idx] else next_idx)

                qrs_4 = []
                for idx in qrs_3:
                    if idx not in qrs_3[diff]:
                        qrs_4.append(idx)
                qrs_4.extend(merge)
                qrs_4 = np.sort(np.array(qrs_4))
                return [int(dx / fixed_fs * self.fs) for dx in qrs_4]
            qrs_3 = np.sort(np.array(qrs_3))
            return [int(dx / fixed_fs * self.fs) for dx in qrs_3]
        except Exception as e:
            print(e)
            return []


def resample_sig_v2(ts, fs_in, fs_out):
    """
    基于线性拟合的差值重采样算法
    计算前后点对应的比例进行插值
    :param ts:  单导联数据，一维浮点型数组
    :param fs_in: 原始采样率，整型
    :param fs_out: 目标采样率，整型
    :return: 重采样后的数据
    """
    t = len(ts) / fs_in
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_out == fs_in:
        return ts
    else:
        x_old = np.linspace(0, 1, num=len(ts), endpoint=True)
        x_new = np.linspace(0, 1, num=int(t * fs_out), endpoint=True)
        y_old = ts
        f = interp1d(x_old, y_old, kind='linear')
        y_new = f(x_new)
        return y_new


def sig_preprocess(sig, fs):
    """
    信号预处理
    使用小波分解去除低频带部分
    :param sig:
    :param fs:
    :return:
    """
    fixed_fs = 125
    resample_sig = resample_sig_v2(sig, fs, fixed_fs)
    wavelet_name = 'db4'
    cs = wavedec(resample_sig, wavelet=wavelet_name, level=6)
    cs[0] = np.zeros(len(cs[0]))
    cs[1] = np.zeros(len(cs[1]))
    cs[2] = np.zeros(len(cs[2]))
    wavelet_f = waverec(cs, wavelet=wavelet_name)
    return resample_sig_v2(wavelet_f, fixed_fs, fs)


def simple_filter(dat, fs):
    dat = np.array(dat)
    # 获取以一秒为区间内的最大数据,以获得其中位数,作为为筛选条件，减小采样率与幅值的影响
    representative = []
    step = fs
    cycles = int(math.ceil(len(dat) / step))
    for cycle in range(cycles):
        representative.append(np.max(np.abs(dat[cycle * step:(cycle + 1) * step])))
    if len(representative) < 1:
        return dat
    representative = np.array(representative)
    # 放大一定的倍数
    median_ = np.median(representative[np.where(representative > 0.001)])
    max_ = median_ * 5
    # 数据过滤
    idics = np.where(np.abs(dat) > max_)[0]
    invalid = []
    for _ in range(len(idics) - 1):
        max_ = min(len(dat), idics[_] + 2 * fs)
        min_ = max(idics[_] - 2 * fs, 0)
        if len(invalid) > 0:
            if invalid[-1][-1] < min_:
                invalid[-1][-1] = max_
            else:
                invalid.append([min_, max_])
        else:
            invalid.append([min_, max_])
    for _ in invalid:
        dat[_[0]:_[1]] = 0

    return dat


def trend_filter(dat, fs):
    dat = resample_sig_v2(dat, fs, 360)
    useful, useless = abrupt_change_and_baseline_wander_detector(dat)
    if useful:
        tmp = []
        for _ in useful:
            tmp.append([int(_[0] / 360 * fs), int(_[1] / 360 * fs)])
        useful = tmp
    if useless:
        tmp = []
        for _ in useless:
            tmp.append([int(_[0] / 360 * fs), int(_[1] / 360 * fs)])
        useless = tmp
    return useful, useless


def abrupt_change_and_baseline_wander_detector(dat):
    fixed_fs = 360
    t_b = 0.1
    t_a = 2 * t_b
    fragment_len = int(fixed_fs * 0.5)
    fragment_overlap = int(fragment_len * 0.5)
    wave_name = 'sym3'
    cs = wavedec(dat, wave_name, level=9)
    cs_h2 = [cs[0], cs[1]]
    for _ in range(2, len(cs)):
        cs_h2.append(np.zeros(len(cs[_])))
    h2 = waverec(cs_h2, wave_name)

    fragments = int((len(h2) - fragment_len) / fragment_overlap)
    fragment_representative = []
    for f in range(fragments):
        start = f * fragment_overlap
        end = start + fragment_len
        fragment_representative.append(np.max(np.abs(h2[start:end])))

    ds = np.diff(np.array(fragment_representative))
    ds = np.append(ds, ds[-1])
    useless_fragments = set([])
    for idx, d in enumerate(ds):
        if d >= t_a:
            start = max(0, idx - 4)
            end = min(idx + 4 + 1, len(ds))
            for _ in range(start, end):
                useless_fragments.add(_)

    useful = []
    for _ in range(fragments):
        if _ not in useless_fragments:
            useful.append(_)

    useless_fragments = list(useless_fragments)
    useless_fragments.sort()
    useless_index = []
    if useless_fragments:
        start = useless_fragments[0] * fragment_overlap
        end = start + fragment_len
        for _ in useless_fragments:
            if _ * fragment_overlap <= end:
                end = _ * fragment_overlap + fragment_len
            else:
                useless_index.append([start, end])
                start = _ * fragment_overlap
                end = start + fragment_len
        useless_index.append([start, end])

    useful_fragment_index = []
    if useful:
        start = useful[0] * fragment_overlap
        end = start + fragment_len
        for _ in useful:
            if _ * fragment_overlap <= end:
                end = _ * fragment_overlap + fragment_len
            else:
                useful_fragment_index.append([start, end])
                start = _ * fragment_overlap
                end = start + fragment_len
        useful_fragment_index.append([start, end])
    return useful_fragment_index, useless_index


class AdaptiveTemplateQrsDetect(object):
    """
    自适应模板匹配QRS波群检测，该算法缺陷：受原生的gqrs算法影响较大，后续有需要会对该部分进行优化
    """

    # 固定采样率
    fixed_fs = 125

    def __init__(self, sig, fs):
        # 原始数据
        self.sig = sig
        # 原始采样率
        self.fs = fs
        # 重采样数据
        self.resample_sig = np.array(resample_sig_v2(self.sig, self.fs, self.fixed_fs))
        # 模板数据
        self.templates = None
        # 维修站的模板位置
        self.templates_centers = None
        # 进行基线漂移去除
        self.rbw_sig = smooth_avg2(self.resample_sig, int(0.3 * self.fixed_fs))
        self.rbw_smooth_sig, _, _ = bsp_tools.filter_signal(signal=self.rbw_sig,
                                                            ftype='cheby2',
                                                            band='bandpass',
                                                            order=4,
                                                            frequency=[0.5, 45],
                                                            sampling_rate=self.fixed_fs, rs=10)
        self.clean_sig, _, _ = bsp_tools.filter_signal(signal=self.resample_sig,
                                                       ftype='cheby2',
                                                       band='bandpass',
                                                       order=4,
                                                       frequency=[1.5, 45],
                                                       sampling_rate=self.fixed_fs, rs=10)
        self.rbw_smooth_sig = simple_filter(self.rbw_smooth_sig, self.fixed_fs)
        # 模板提取:向前截断的时间,单位s
        self.template_before = 0.25
        # 模板提取:向后截断的时间,单位s
        self.template_after = 0.25
        # 模板匹配时，候选qrs波前后滚动的时间,单位s
        self.match_radius = 0.1
        # 对简单滤波的数据进行求导
        self.der_i_1 = Processor.der(self.rbw_smooth_sig, self.fixed_fs)
        # 模板qrs位置
        self.template_qrs = None
        # 候选qrs波位置
        self.candidate_qrs = None
        self.qrs1 = None
        self.qrs2 = None
        self.qrs3 = None
        # 修正模板qrs位置的容忍度,与候选qrs波位置对比,
        # 取二者交集作为修正后的qrs波位置
        self.tolerance = int(0.1 * self.fixed_fs)

    def detect(self):
        import logging
        try:
            # 模板生成
            self._generate_template()
            # qrs波检测
            return self._qrs_detect()
        except Exception as e:
            logging.exception(e)
            return None

    def _generate_template(self):
        """
        生成模板
        """
        # 生成初步的qrs模板位置
        self._template_qrs_detect()
        # 获取待检测QRS波位置
        self._candidate_qrs_detect()
        # 修正QRS波模板位置
        self._fix_qrs_template_centers()
        # 提取模板
        self._extract_template()

    def _template_qrs_detect(self):
        """
        模板位置qrs波检测
        """
        # 分别使用三种检测速度较快的QRS波检测算法进行检测
        # qrs模板为qrs1和qrs2的交集与qrs3的并集
        _, qrs1_ = nk.ecg_peaks(self.rbw_smooth_sig, sampling_rate=self.fixed_fs)
        self.qrs1 = qrs1_['ECG_R_Peaks']
        _, qrs2_ = nk.ecg_peaks(self.der_i_1, sampling_rate=self.fixed_fs)
        self.qrs2 = qrs2_['ECG_R_Peaks']
        self.qrs3 = ecg_findpeaks_rodrigues(self.der_i_1, sampling_rate=self.fixed_fs)
        # 生成qrs模板数据
        if len(self.qrs1) > 1 and len(self.qrs2) > 1:
            result = []
            for r in self.qrs1:
                range_ = range(r - self.tolerance, r + self.tolerance)
                overlapping = list(set(range_).intersection(self.qrs2))
                r_ = r
                if len(overlapping) > 1:
                    r_ = int(np.mean(np.array(overlapping)))
                    overlapping = [r_]
                if len(overlapping) > 0:
                    result.append(r_)
            if len(result) > 0:
                result.sort()
                if len(self.qrs3) > 0:
                    self.template_qrs = self._marge_closed(np.concatenate((np.array(result), self.qrs3), axis=0),
                                                           int(0.2 * self.fixed_fs), len(self.der_i_1),
                                                           self.der_i_1)
                else:
                    self.template_qrs = np.array(result)
                    self.qrs3 = None
            elif len(self.qrs3) > 0:
                self.template_qrs = self.qrs3
                self.qrs1 = None
                self.qrs2 = None
            else:
                self.template_qrs = None
                self.qrs1 = None
                self.qrs2 = None
                self.qrs3 = None
        else:
            if len(self.qrs3) > 0:
                self.template_qrs = self.qrs3
                self.qrs1 = None
                self.qrs2 = None
            else:
                self.template_qrs = None
                self.qrs1 = None
                self.qrs2 = None
                self.qrs3 = None

        if self.template_qrs is not None:
            self.template_qrs = self.template_qrs[np.where(self.template_qrs > 0.3 * self.fixed_fs)]
            if len(self.template_qrs) > 0:
                self.template_qrs = self.template_qrs[
                    np.where(self.template_qrs < len(self.resample_sig) - 0.3 * self.fixed_fs)]
            if len(self.template_qrs) < 1:
                self.template_qrs = None

    def _candidate_qrs_detect(self):
        """
        候选qrs波群位置检测
        :return:
        """
        # 取qrs1，qrs2，qrs3的并集作为待检测qrs波位置
        qrs = None
        if self.qrs1 is not None:
            qrs = self.qrs1

        if self.qrs2 is not None:
            if qrs is None:
                qrs = self.qrs2
            else:
                qrs = np.concatenate((qrs, self.qrs2), axis=0)

        if self.qrs3 is not None:
            if qrs is None:
                qrs = self.qrs3
            else:
                qrs = np.concatenate((qrs, self.qrs3), axis=0)

        if qrs is None or len(qrs) < 1:
            self.candidate_qrs = None
        else:
            # 并对峰谷振幅进行一定的筛选
            result = []
            # 一阶导数波动
            amplitudes_1 = []
            # 振幅
            amplitudes_2 = []
            radius_ = int(0.1 * self.fixed_fs)
            for r in qrs:
                start = max(0, r - radius_)
                end = min(len(self.der_i_1), r + radius_)
                dat = self.der_i_1[start:end]
                amplitudes_1.append(np.max(dat) - np.min(dat))
                dat = self.clean_sig[start:end]
                amplitudes_2.append(np.max(dat) - np.min(dat))
            amplitudes_1 = np.array(amplitudes_1)
            thresh1 = np.mean(amplitudes_1)
            amplitudes_2 = np.array(amplitudes_2)
            # 选择振幅大于0.3的作为阈值选定标准
            size_ = np.where(0.15 <= amplitudes_2)[0]
            if len(size_) < 1:
                thresh2 = 0.15
            else:
                thresh2 = np.mean(amplitudes_2[size_])
            # 进行阈值筛选
            for idx in range(len(qrs)):
                if amplitudes_2[idx] >= 0.35 * thresh2 and amplitudes_1[idx] >= 0.3 * thresh1:
                    result.append(qrs[idx])
            result = np.array(result)
            if len(result) > 0:
                result = self._marge_closed(result, int(0.2 * self.fixed_fs), len(self.der_i_1),
                                            self.der_i_1)
                self.candidate_qrs = result
            else:
                self.candidate_qrs = None

    def _fix_qrs_template_centers(self):
        """
        qrs模板中心修正
        :return:
        """
        # 取qrs模板与待筛选的qrs的交集作为提取模板的qrs位置
        if self.candidate_qrs is not None and self.template_qrs is not None:
            result = []
            for qrs in self.template_qrs:
                t = range(qrs - self.tolerance, qrs + self.tolerance)
                overlapping = len(list(set(t).intersection(self.candidate_qrs)))
                if overlapping > 0:
                    result.append(qrs)
            if len(result) > 0:
                self.templates_centers = self._marge_closed(result, int(0.2 * self.fixed_fs), len(self.der_i_1),
                                                            self.der_i_1)
            else:
                self.templates_centers = None

    def _extract_template(self):
        """
        模板提取
        """
        # 可以进行模板提取
        if self.templates_centers is not None and len(self.templates_centers) > 0:
            self.templates = []
            # 心跳模板提取
            templates_2, rpeaks = bsp_ecg.extract_heartbeats(self.clean_sig, self.templates_centers,
                                                             self.fixed_fs, before=self.template_before,
                                                             after=self.template_after)
            templates_2 = self._beat_template_info(templates_2)
            self.templates = np.array([templates_2, templates_2])
        else:
            self.templates = None

    @staticmethod
    def _beat_template_info(templates):
        """
        心跳模板信息生成
        :param templates: 初步提取的模板
        :return: 重新分类后的模板
        """
        results = []
        if len(templates) < 1:
            return None
        elif len(templates) < 2:
            # 仅有一类模板
            results.append(templates[0])
            results.append(templates[0])
        else:
            # 整体计算平均模板
            avg_ = np.median(templates, axis=0)
            # 各检测到的模板与平均模板的相似度计算
            compared_dat = np.vstack(np.array([[avg_, template] for template in templates]))
            compared_result = np.corrcoef(compared_dat)
            range_ = range(len(templates))
            correlations = [compared_result[0][2 * idx + 1] for idx in range_]
            # 按照与平均模板的相似度跨度最大的点，将各模板划分为2类
            sorted_ = np.array(correlations.copy())
            sorted_.sort()
            diff_sorted = np.diff(np.array(sorted_))
            demarcation = np.argmax(diff_sorted)
            th = sorted_[demarcation]
            type_i = []
            type_ii = []
            for idx, v in enumerate(correlations):
                if v > th:
                    type_i.append(templates[idx])
                else:
                    type_ii.append(templates[idx])
            # 生成两类模板
            results.append(np.mean(np.array(type_i), axis=0))
            results.append(np.mean(np.array(type_ii), axis=0))
        return results

    def _qrs_detect(self):
        """
        qrs波群检测
        :return:
        """
        r_peaks = []
        probabilities = []
        match_points = int(self.match_radius * self.fixed_fs)
        if self.templates is not None and self.candidate_qrs is not None and len(self.candidate_qrs) > 0:
            # 信号前0.3s及末尾0.3s的qrs波不进行检测
            self.candidate_qrs = self.candidate_qrs[np.where(self.candidate_qrs > 0.3 * self.fixed_fs)]
            if len(self.candidate_qrs) > 0:
                self.candidate_qrs = self.candidate_qrs[
                    np.where(self.candidate_qrs < len(self.resample_sig) - 0.3 * self.fixed_fs)]
                if len(self.candidate_qrs) > 0:
                    # 根据第一个及最后一个qrs波位置，向待使用数据前后拼接模板长度的0
                    first_idx = np.min(self.candidate_qrs)
                    last_index = np.max(self.candidate_qrs)
                    offset_index = len(self.templates[1][0]) - (first_idx + 1)
                    tmp_dat_2 = np.array(self.clean_sig)
                    if offset_index > 0:
                        tmp_dat_2 = np.concatenate((np.zeros(offset_index), tmp_dat_2))
                    else:
                        offset_index = 0
                    if (last_index + 1 + offset_index + len(self.templates[1][0])) - len(tmp_dat_2) > 0:
                        tmp_dat_2 = np.concatenate((tmp_dat_2, np.zeros(
                            (last_index + 1 + offset_index + len(self.templates[1][0])) - len(tmp_dat_2))))
                    # 一次计算候选qrs波在匹配范围内与模板的相似度
                    for idx in self.candidate_qrs:
                        p_2 = self._single_type_similarity(idx + offset_index, self.templates[1], tmp_dat_2,
                                                           match_points)
                        probabilities.append(p_2)
                        r_peaks.append(idx)
        # 没有检测到
        if len(r_peaks) < 1:
            return None
        else:
            # 返回下标及对应的相似度
            return pd.DataFrame(np.transpose(np.array([probabilities])), columns=['probabilities'],
                                index=[int(_ / self.fixed_fs * self.fs) for _ in r_peaks])

    def _single_type_similarity(self, index, base_template, sig, radius):
        """
        数据与单一类型的模板的相似度，使用皮尔逊相关系数
        :param index:
        :param base_template: 基础模板数据
        :param sig: 数据
        :param radius:
        :return: 最大的相关系数
        """
        max_ = -1
        radius_before = int(self.fixed_fs * self.template_before)
        radius_after = int(self.fixed_fs * self.template_after)
        for template in base_template:
            range_ = range(index - radius, index + radius)
            offset_ = range_[0]
            compared_dat = np.vstack(
                np.array([[template, sig[idx - radius_before:idx + radius_after]] for idx in range_]))
            tmp = np.corrcoef(compared_dat)[0]
            result = np.max(np.array([tmp[2 * (idx - offset_) + 1] for idx in range_]))
            max_ = max_ if max_ >= result else result
        return max_

    @staticmethod
    def _marge_closed(peaks, tolerance, max_size, sig):
        """
        合并较为靠近的peak
        :param peaks: 一维数组,波的位置
        :param tolerance: 最远的长度
        :param max_size: 最大的结束位置
        :return: 合并后的peaks
        """
        peaks = np.sort(np.array(peaks))
        result = []
        for idx, _ in enumerate(peaks):
            start = max(0, _ - tolerance)
            end = min(max_size, _ + tolerance)
            tmp = list(set(range(start, end)).intersection(result))
            if len(tmp) < 2:
                if len(tmp) < 1:
                    result.append(_)
                else:
                    t_ = abs(sig[_])
                    t = abs(sig[result[-1]])
                    if t_ > t:
                        result[-1] = _
                    elif t_ < t:
                        result[-1] = result[-1]
                    else:
                        result[-1] = int((_ + result[-1]) / 2)
        result = list(result)
        result.sort()
        for idx, v in enumerate(result):
            start = max(0, int(v - tolerance / 4))
            end = min(len(sig) - 1, int(v + tolerance / 4))
            result[idx] = start + int((np.argmax(sig[start:end]) + np.argmin(sig[start:end])) / 2)
        return np.array(result)


def simple_adaptive_template_qrs_detect(sig, fs):
    """
    自适应模板匹配检测算法的简单调用方式
    :param sig: 原始ecg数据
    :param fs: 数据采样率
    :return: qrs下标及对应的相似度
    """
    step = 28 * fs
    window_len = 30 * fs
    end_index = len(sig)
    fragments = sig_fragment(sig, window_len, step)
    result = None
    for idx, fragment in enumerate(fragments):
        start_index = idx * step
        detector = AdaptiveTemplateQrsDetect(fragment, fs)
        tmp = detector.detect()
        if tmp is not None:
            tmp.index += start_index
            tmp.sort_index()
            tmp = tmp.iloc[np.where(tmp.index < end_index)[0], :]
            if result is None:
                result = tmp
            else:
                last = result.index[-1]
                tmp = tmp.iloc[np.where(tmp.index > last + int(0.3 * fs))[0], :]
                result = result.append(tmp)
    if result is None:
        return None
    return result
