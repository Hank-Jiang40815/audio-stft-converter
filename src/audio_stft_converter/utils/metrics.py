#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Metrics:
    """音訊評估指標結算工具。

    該類別提供了計算各種音訊評估指標的功能。
    """

    def __init__(self):
        """初始化指標計算工具。
        """
        pass

    def compute_snr(self, original, reconstructed):
        """計算信噪比 (Signal-to-Noise Ratio)。

        Args:
            original (np.ndarray): 原始信號。
            reconstructed (np.ndarray): 重建信號。

        Returns:
            float: SNR，單位為 dB。
        """
        # 確保信號長度一致
        min_length = min(len(original), len(reconstructed))
        original = original[:min_length]
        reconstructed = reconstructed[:min_length]
        
        # 計算駐留信號能量
        signal_power = np.sum(np.abs(original) ** 2) / min_length
        
        # 計算噪聲能量
        noise = original - reconstructed
        noise_power = np.sum(np.abs(noise) ** 2) / min_length
        
        # 避免除零錯誤
        if noise_power < 1e-10:
            return float('inf')  # 幾乎無噪聲
        
        # 計算 SNR
        snr = 10 * np.log10(signal_power / noise_power)
        
        return snr

    def compute_mse(self, original, reconstructed):
        """計算平均平方誤差 (Mean Square Error)。

        Args:
            original (np.ndarray): 原始信號。
            reconstructed (np.ndarray): 重建信號。

        Returns:
            float: MSE 值。
        """
        # 確保信號長度一致
        min_length = min(len(original), len(reconstructed))
        original = original[:min_length]
        reconstructed = reconstructed[:min_length]
        
        # 計算 MSE
        mse = np.mean((original - reconstructed) ** 2)
        
        return mse

    def compute_correlation(self, original, reconstructed):
        """計算相關性係數。

        Args:
            original (np.ndarray): 原始信號。
            reconstructed (np.ndarray): 重建信號。

        Returns:
            float: 相關性係數，範圍從 -1 到 1。
        """
        # 確保信號長度一致
        min_length = min(len(original), len(reconstructed))
        original = original[:min_length]
        reconstructed = reconstructed[:min_length]
        
        # 計算相關性係數
        correlation = np.corrcoef(original, reconstructed)[0, 1]
        
        return correlation

    def compute_spectral_distance(self, original, reconstructed, sample_rate):
        """計算頻譜距離，用於量化兩個信號在頻域上的差異。

        Args:
            original (np.ndarray): 原始信號。
            reconstructed (np.ndarray): 重建信號。
            sample_rate (int): 取樣率。

        Returns:
            float: 頻譜距離指標。
        """
        # 確保信號長度一致
        min_length = min(len(original), len(reconstructed))
        original = original[:min_length]
        reconstructed = reconstructed[:min_length]
        
        # 計算頻譜
        orig_spec = np.abs(np.fft.rfft(original))
        recon_spec = np.abs(np.fft.rfft(reconstructed))
        
        # 計算平均差異
        distance = np.mean(np.abs(orig_spec - recon_spec))
        
        return distance

    def evaluate_all(self, original, reconstructed, sample_rate):
        """計算所有評估指標。

        Args:
            original (np.ndarray): 原始信號。
            reconstructed (np.ndarray): 重建信號。
            sample_rate (int): 取樣率。

        Returns:
            dict: 包含所有指標的字典。
        """
        results = {
            'snr': self.compute_snr(original, reconstructed),
            'mse': self.compute_mse(original, reconstructed),
            'correlation': self.compute_correlation(original, reconstructed),
            'spectral_distance': self.compute_spectral_distance(original, reconstructed, sample_rate)
        }
        
        return results
