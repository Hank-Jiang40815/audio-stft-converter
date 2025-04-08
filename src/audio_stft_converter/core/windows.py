#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal

class WindowFunctions:
    """視窗函數管理器。

    該類別提供了生成和管理不同類型視窗函數的功能。
    """

    def __init__(self):
        """初始化視窗函數管理器。
        """
        # 支援的視窗函數列表
        self.supported_windows = [
            "boxcar", "triang", "blackman", "hamming", "hann", 
            "bartlett", "flattop", "parzen", "bohman", "blackmanharris", 
            "nuttall", "barthann"
        ]

    def get_window(self, window_type, window_size):
        """產生指定類型和大小的視窗函數。

        Args:
            window_type (str): 視窗函數類型。
            window_size (int): 視窗函數大小。

        Returns:
            np.ndarray: 視窗函數數組。

        Raises:
            ValueError: 如果請求的視窗類型不支援。
        """
        if window_type not in self.supported_windows:
            raise ValueError(
                f"Unsupported window type: {window_type}. " \
                f"Supported types are: {', '.join(self.supported_windows)}"
            )

        return signal.get_window(window_type, window_size)

    def list_supported_windows(self):
        """列出所有支援的視窗函數類型。

        Returns:
            list: 支援的視窗函數類型列表。
        """
        return self.supported_windows.copy()

    def compare_windows(self, window_size=1024):
        """比較不同視窗函數的頻率響應。

        Args:
            window_size (int, optional): 視窗大小。預設為1024。

        Returns:
            dict: 包含每個視窗函數和其頻率響應的字典。
        """
        results = {}
        for window_type in self.supported_windows:
            window = self.get_window(window_type, window_size)
            # 計算頻率響應
            freq_response = np.abs(np.fft.rfft(window))
            # 將響應正規化到 0-1 範圍
            freq_response = freq_response / np.max(freq_response)
            results[window_type] = freq_response
            
        return results
