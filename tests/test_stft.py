#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""測試 STFT/iSTFT 轉換功能。

此檔案包含了驗證 audio-stft-converter 套件中
核心 STFT/iSTFT 功能的測試。
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_stft_converter.core import STFTProcessor

class TestSTFTProcessor(unittest.TestCase):
    """STFT 處理器的單元測試。"""
    
    def setUp(self):
        """測試前的初始化。"""
        self.window_size = 1024
        self.hop_length = 256
        self.window_type = "hann"
        self.stft_processor = STFTProcessor(
            window_size=self.window_size,
            hop_length=self.hop_length,
            window_type=self.window_type
        )
        
        # 創建一個簡單的測試信號
        self.sample_rate = 44100
        duration = 0.5  # 0.5 秒
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        self.test_signal = np.sin(2 * np.pi * 440 * t)  # 440Hz 的正弦波
    
    def test_stft_shape(self):
        """測試 STFT 輸出的形狀是否符合預期。"""
        stft_matrix = self.stft_processor.stft(self.test_signal)
        
        # 預期幀數 = (信號長度 - 視窗大小) / 跳躍長度 + 1
        expected_frames = 1 + (len(self.test_signal) - self.window_size) // self.hop_length
        
        self.assertEqual(stft_matrix.shape[0], self.window_size)
        self.assertEqual(stft_matrix.shape[1], expected_frames)
    
    def test_stft_istft_reconstruction(self):
        """測試 STFT 後網接 iSTFT 能否重建原始信號。"""
        # STFT 轉換
        stft_matrix = self.stft_processor.stft(self.test_signal)
        
        # iSTFT 重建
        reconstructed_signal = self.stft_processor.istft(stft_matrix)
        
        # 截取有效長度進行比較
        # 注意: 因為視窗重疊和邊緣效應，需要忽略两端的樣本
        valid_length = len(self.test_signal) - self.window_size
        original = self.test_signal[self.window_size//2:-(self.window_size//2)]
        reconstructed = reconstructed_signal[self.window_size//2:self.window_size//2+len(original)]
        
        # 使用相對誤差
        error = np.mean(np.abs(original - reconstructed)) / np.mean(np.abs(original))
        
        # 相對誤差應實小於 1%
        self.assertLess(error, 0.01)
        
    def test_window_type(self):
        """測試不同視窗函數類型。"""
        # 嘗試常用的視窗函數
        window_types = ["hann", "hamming", "blackman"]
        
        for window_type in window_types:
            with self.subTest(window_type=window_type):
                processor = STFTProcessor(
                    window_size=self.window_size,
                    hop_length=self.hop_length,
                    window_type=window_type
                )
                
                # 確保可以建立並使用相應的視窗
                self.assertEqual(processor.window_type, window_type)
                self.assertEqual(len(processor.window), self.window_size)
                
                # 測試 STFT 和 iSTFT
                stft_matrix = processor.stft(self.test_signal)
                reconstructed_signal = processor.istft(stft_matrix)
                
                # 確認重建的信號長度正確
                expected_length = (stft_matrix.shape[1] - 1) * self.hop_length + self.window_size
                self.assertEqual(len(reconstructed_signal), expected_length)
    
    def test_invalid_window_type(self):
        """測試無效的視窗函數類型。"""
        with self.assertRaises(ValueError):
            STFTProcessor(
                window_size=self.window_size,
                hop_length=self.hop_length,
                window_type="invalid_window_type"
            )

if __name__ == "__main__":
    unittest.main()
