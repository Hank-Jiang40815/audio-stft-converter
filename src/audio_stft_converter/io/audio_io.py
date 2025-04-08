#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import soundfile as sf
import librosa

class AudioIO:
    """音訊檔案讀取和寫入工具。

    該類別提供了讀取和寫入各種音訊格式的功能。
    """

    def __init__(self):
        """初始化音訊 IO 工具。
        """
        # 支援的音訊格式
        self.supported_formats = [
            '.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aiff',
        ]

    def load(self, file_path, sr=None, mono=True):
        """讀取音訊檔案。

        Args:
            file_path (str): 音訊檔案路徑。
            sr (int, optional): 目標取樣率。如果為 None，保留原始取樣率。
            mono (bool, optional): 是否轉換為單聲道。預設為 True。

        Returns:
            tuple: (音訊數據, 取樣率)

        Raises:
            ValueError: 如果檔案格式不受支援或檔案不存在。
        """
        # 檢查檔案是否存在
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        # 檢查檔案格式
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. " \
                f"Supported formats are: {', '.join(self.supported_formats)}"
            )

        # 載入音訊
        try:
            y, sr_orig = librosa.load(file_path, sr=sr, mono=mono)
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")

        return y, sr_orig

    def save(self, file_path, audio_data, sample_rate):
        """將音訊數據寫入成音訊檔案。

        Args:
            file_path (str): 目標檔案路徑。
            audio_data (np.ndarray): 音訊數據。
            sample_rate (int): 取樣率。

        Raises:
            ValueError: 如果檔案格式不受支援。
        """
        # 檢查檔案格式
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. " \
                f"Supported formats are: {', '.join(self.supported_formats)}"
            )

        # 確保數據在合理範圍內
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # 寫入音訊檔案
        try:
            sf.write(file_path, audio_data, sample_rate)
        except Exception as e:
            raise ValueError(f"Error saving audio file: {str(e)}")

    def get_info(self, file_path):
        """獲取音訊檔案的詳細資訊。

        Args:
            file_path (str): 音訊檔案路徑。

        Returns:
            dict: 包含音訊資訊的字典。

        Raises:
            ValueError: 如果檔案不存在或格式不受支援。
        """
        # 檢查檔案是否存在
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        # 載入音訊資訊
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            info = {
                'sample_rate': sr,
                'channels': 1 if y.ndim == 1 else y.shape[1],
                'duration': duration,
                'samples': len(y),
                'file_size': os.path.getsize(file_path),
                'format': os.path.splitext(file_path)[1][1:]
            }
            return info
        except Exception as e:
            raise ValueError(f"Error getting audio info: {str(e)}")
