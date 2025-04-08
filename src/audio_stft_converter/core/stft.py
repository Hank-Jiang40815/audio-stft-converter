#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import datetime
from .windows import WindowFunctions

class STFTProcessor:
    """短時傅立葉轉換(STFT)處理器。

    該類別提供了執行短時傅立葉轉換(STFT)和逆短時傅立葉轉換(iSTFT)的功能，
    以及相關的參數設定。
    
    Attributes:
        window_size (int): 視窗大小（樣本數）。
        hop_length (int): 視窗跳躍長度（樣本數）。
        window_type (str): 視窗函數類型。
        window (np.ndarray): 視窗函數數組。
        experiment_id (int): 實驗編號，用於追蹤和報告。
        date (str): 執行日期（格式：YYYYMMDD）。
    """

    def __init__(self, window_size=2048, hop_length=512, window_type="hann", experiment_id=1):
        """初始化 STFTProcessor 類別。

        Args:
            window_size (int, optional): 視窗大小（樣本數）。預設為2048。
            hop_length (int, optional): 視窗跳躍長度（樣本數）。預設為512。
            window_type (str, optional): 視窗函數類型。預設為"hann"。
            experiment_id (int, optional): 實驗編號。預設為1。
        """
        self.window_size = window_size
        self.hop_length = hop_length
        self.window_type = window_type
        self.experiment_id = experiment_id
        self.date = datetime.datetime.now().strftime("%Y%m%d")
        
        window_functions = WindowFunctions()
        self.window = window_functions.get_window(window_type, window_size)

    def stft(self, signal):
        """執行短時傅立葉轉換。

        Args:
            signal (np.ndarray): 輸入的音訊信號。

        Returns:
            np.ndarray: 複數格式的STFT矩陣。形狀為(window_size, num_frames)。
        """
        # 確保信號是一維數組
        if len(signal.shape) > 1:
            signal = signal.mean(axis=1)  # 轉換為單聲道
            
        # 計算幀數
        num_frames = 1 + (len(signal) - self.window_size) // self.hop_length
        
        # 初始化 STFT 矩陣
        stft_matrix = np.zeros((self.window_size, num_frames), dtype=np.complex128)
        
        # 執行 STFT
        for i in range(num_frames):
            # 取得目前幀數据
            frame_start = i * self.hop_length
            frame_end = frame_start + self.window_size
            frame = signal[frame_start:frame_end]
            
            # 應用視窗函數並計算 FFT
            windowed_frame = frame * self.window
            stft_matrix[:, i] = np.fft.fft(windowed_frame)
            
        return stft_matrix

    def istft(self, stft_matrix):
        """執行逆短時傅立葉轉換。

        Args:
            stft_matrix (np.ndarray): 複數格式的STFT矩陣。形狀為(window_size, num_frames)。

        Returns:
            np.ndarray: 重建的音訊信號。
        """
        # 獲取尺寸
        window_size = stft_matrix.shape[0]
        num_frames = stft_matrix.shape[1]
        
        # 預估輸出信號長度
        output_length = self.hop_length * (num_frames - 1) + window_size
        
        # 初始化輸出信號和等待權重計算的累加器
        output_signal = np.zeros(output_length)
        window_sum = np.zeros(output_length)
        
        # 重疊加法合成
        for i in range(num_frames):
            # 執行逆FFT
            frame = np.fft.ifft(stft_matrix[:, i]).real
            
            # 應用視窗函數
            windowed_frame = frame * self.window
            
            # 確定該幀在輸出信號中的位置
            frame_start = i * self.hop_length
            frame_end = frame_start + window_size
            
            # 將該幀加到輸出信號中
            output_signal[frame_start:frame_end] += windowed_frame
            
            # 記錄視窗函數的重疊情況
            window_sum[frame_start:frame_end] += self.window
        
        # 防止除零錯誤
        window_sum[window_sum < 1e-10] = 1.0
        
        # 正規化以處理重疊部分
        output_signal /= window_sum
        
        return output_signal

    def modify_magnitude(self, stft_matrix, gain_db=0):
        """修改 STFT 矩陣的幅度。

        Args:
            stft_matrix (np.ndarray): 複數格式的STFT矩陣。形狀為(window_size, num_frames)。
            gain_db (float, optional): 增益，以分貛為單位。預設為0。

        Returns:
            np.ndarray: 修改後的STFT矩陣。
        """
        # 轉換增益到線性尺度
        gain_linear = 10 ** (gain_db / 20.0)
        
        # 抽取相位和幅度
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # 應用增益
        modified_magnitude = magnitude * gain_linear
        
        # 重建複數矩陣
        modified_stft = modified_magnitude * np.exp(1j * phase)
        
        return modified_stft

    def filter_frequencies(self, stft_matrix, sample_rate, low_cut=None, high_cut=None):
        """適用頻率濾波器到 STFT 矩陣。

        Args:
            stft_matrix (np.ndarray): 複數格式的STFT矩陣。形狀為(window_size, num_frames)。
            sample_rate (int): 音訊的取樣率。
            low_cut (float, optional): 低頻截止，單位為Hz。如果為 None，則不會應用低頻濾波。
            high_cut (float, optional): 高頻截止，單位為Hz。如果為 None，則不會應用高頻濾波。

        Returns:
            np.ndarray: 濾波後的STFT矩陣。
        """
        # 複製原始矩陣
        filtered_stft = stft_matrix.copy()
        
        # 計算頻率尺度
        freqs = np.fft.fftfreq(self.window_size, 1.0/sample_rate)
        
        # 建立頻率過濾器
        if low_cut is not None:
            low_mask = np.abs(freqs) < low_cut
            filtered_stft[low_mask, :] = 0
            
        if high_cut is not None:
            high_mask = np.abs(freqs) > high_cut
            filtered_stft[high_mask, :] = 0
            
        return filtered_stft
