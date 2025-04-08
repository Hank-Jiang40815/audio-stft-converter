#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

class Visualizer:
    """視覺化工具類別。

    該類別提供了各種視覺化功能來呈現 STFT 結果和音訊信號。
    """

    def __init__(self, experiment_id=1):
        """初始化視覺化工具。

        Args:
            experiment_id (int, optional): 實驗ID。預設為1。
        """
        self.experiment_id = experiment_id
        self.date = datetime.datetime.now().strftime("%Y%m%d")

    def plot_spectrogram(self, stft_matrix, sample_rate, hop_length, 
                        output_path=None, title=None, vmin=None, vmax=None):
        """繪製 STFT 時頻譜圖。

        Args:
            stft_matrix (np.ndarray): STFT 矩陣。
            sample_rate (int): 取樣率。
            hop_length (int): 進行 STFT 時的跳躍長度。
            output_path (str, optional): 輸出檔案路徑。如果為 None，則自動生成。
            title (str, optional): 圖表標題。如果為 None，則自動生成。
            vmin (float, optional): 饐值最小值。
            vmax (float, optional): 饐值最大值。

        Returns:
            str: 儲存的圖檔路徑。
        """
        # 計算幅值變分貛
        magnitude = np.abs(stft_matrix)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # 計算時間軸和頻率軸
        frames = range(magnitude.shape[1])
        t = [frame * hop_length / sample_rate for frame in frames]
        freq = np.linspace(0, sample_rate/2, magnitude.shape[0] // 2 + 1)
        
        # 只顯示限制顧注頻率的一半（正頻率）
        magnitude_db_display = magnitude_db[:magnitude.shape[0] // 2 + 1, :]
        
        # 繪製時頻譜圖
        plt.figure(figsize=(10, 6))
        plt.imshow(
            magnitude_db_display, 
            aspect='auto', 
            origin='lower',
            extent=[t[0], t[-1], freq[0], freq[-1]],
            vmin=vmin,
            vmax=vmax,
            cmap='inferno'
        )
        
        # 設定紙軸和標題
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('時間 (s)')
        plt.ylabel('頻率 (Hz)')
        
        if title is None:
            title = f'STFT 時頻譜圖 - 實驗 #{self.experiment_id}_{self.date}'
        plt.title(title)
        
        # 儲存圖檔
        if output_path is None:
            output_path = f'STFT_Spectrogram_Exp{self.experiment_id}_{self.date}_plot_spectrogram.png'
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path

    def plot_waveform(self, signal, sample_rate, output_path=None, title=None):
        """繪製音訊波形圖。

        Args:
            signal (np.ndarray): 音訊波形數據。
            sample_rate (int): 取樣率。
            output_path (str, optional): 輸出檔案路徑。如果為 None，則自動生成。
            title (str, optional): 圖表標題。如果為 None，則自動生成。

        Returns:
            str: 儲存的圖檔路徑。
        """
        # 計算時間軸
        duration = len(signal) / sample_rate
        time = np.linspace(0, duration, len(signal))
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, signal)
        plt.xlabel('時間 (s)')
        plt.ylabel('振幅')
        
        if title is None:
            title = f'音訊波形 - 實驗 #{self.experiment_id}_{self.date}'
        plt.title(title)
        
        plt.grid(True)
        plt.tight_layout()
        
        # 儲存圖檔
        if output_path is None:
            output_path = f'Waveform_Exp{self.experiment_id}_{self.date}_plot_waveform.png'
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path

    def plot_comparison(self, original_signal, reconstructed_signal, sample_rate, 
                      output_path=None, title=None):
        """繪製原始與重建音訊信號的比較圖。

        Args:
            original_signal (np.ndarray): 原始音訊數據。
            reconstructed_signal (np.ndarray): 重建音訊數據。
            sample_rate (int): 取樣率。
            output_path (str, optional): 輸出檔案路徑。如果為 None，則自動生成。
            title (str, optional): 圖表標題。如果為 None，則自動生成。

        Returns:
            str: 儲存的圖檔路徑。
        """
        # 確保信號長度一致
        min_length = min(len(original_signal), len(reconstructed_signal))
        original_signal = original_signal[:min_length]
        reconstructed_signal = reconstructed_signal[:min_length]
        
        # 計算時間軸
        duration = min_length / sample_rate
        time = np.linspace(0, duration, min_length)
        
        plt.figure(figsize=(12, 8))
        
        # 繪製原始信號
        plt.subplot(3, 1, 1)
        plt.plot(time, original_signal)
        plt.title('原始信號')
        plt.xlabel('時間 (s)')
        plt.ylabel('振幅')
        plt.grid(True)
        
        # 繪製重建信號
        plt.subplot(3, 1, 2)
        plt.plot(time, reconstructed_signal)
        plt.title('重建信號')
        plt.xlabel('時間 (s)')
        plt.ylabel('振幅')
        plt.grid(True)
        
        # 繪製誤差
        plt.subplot(3, 1, 3)
        plt.plot(time, original_signal - reconstructed_signal)
        plt.title('重建誤差')
        plt.xlabel('時間 (s)')
        plt.ylabel('振幅')
        plt.grid(True)
        
        if title is None:
            title = f'原始與重建信號比較 - 實驗 #{self.experiment_id}_{self.date}'
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # 儲存圖檔
        if output_path is None:
            output_path = f'Comparison_Exp{self.experiment_id}_{self.date}_plot_comparison.png'
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
