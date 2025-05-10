"""音訊訊號分析與驗證模組。

This module provides tools for analyzing and validating audio signal processing results.
"""
import os
import json
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def analyze_reconstruction(original_path: str, reconstructed_path: str, output_dir: str) -> Dict[str, Any]:
    """分析原始音訊和重建音訊的差異。

    Args:
        original_path (str): 原始音訊檔案路徑
        reconstructed_path (str): 重建音訊檔案路徑
        output_dir (str): 輸出分析結果的目錄路徑

    Returns:
        Dict[str, Any]: 包含分析結果的字典，包括:
            - mse: 均方誤差
            - max_error: 最大誤差
            - correlation: 相關係數
            - snr: 訊噪比
            - plots_dir: 輸出圖表目錄路徑
    """
    try:
        # 使用 soundfile 讀取音訊，保持原始取樣率
        y_orig, sr_orig = sf.read(original_path)
        y_recon, sr_recon = sf.read(reconstructed_path)

        # 確保取樣率一致
        if sr_orig != sr_recon:
            y_recon = librosa.resample(y_recon, orig_sr=sr_recon, target_sr=sr_orig)

        # 確保長度一致
        min_len = min(len(y_orig), len(y_recon))
        y_orig = y_orig[:min_len]
        y_recon = y_recon[:min_len]

        # 如果是多聲道，轉換為單聲道
        if len(y_orig.shape) > 1:
            y_orig = np.mean(y_orig, axis=1)
        if len(y_recon.shape) > 1:
            y_recon = np.mean(y_recon, axis=1)

        # 計算各種誤差指標
        mse = np.mean((y_orig - y_recon) ** 2)
        max_error = np.max(np.abs(y_orig - y_recon))
        correlation = np.corrcoef(y_orig, y_recon)[0, 1]
        
        # 計算訊噪比 (SNR)
        signal_power = np.mean(y_orig ** 2)
        noise_power = np.mean((y_orig - y_recon) ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        # 建立輸出目錄
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = output_path / f"analysis_plots_{timestamp}"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 繪製波形比較圖
        plt.figure(figsize=(15, 5))
        time_axis = np.arange(len(y_orig)) / sr_orig
        plt.plot(time_axis, y_orig, label='Original', alpha=0.7)
        plt.plot(time_axis, y_recon, label='Reconstructed', alpha=0.7)
        plt.title(f'Waveform Comparison (MSE={mse:.6f}, SNR={snr:.2f}dB)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / f'waveform_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 繪製頻譜圖比較
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 原始訊號的頻譜圖
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
        img1 = librosa.display.specshow(D_orig, y_axis='log', x_axis='time', ax=ax1)
        ax1.set_title('Original Spectrogram')
        plt.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        # 重建訊號的頻譜圖
        D_recon = librosa.amplitude_to_db(np.abs(librosa.stft(y_recon)), ref=np.max)
        img2 = librosa.display.specshow(D_recon, y_axis='log', x_axis='time', ax=ax2)
        ax2.set_title('Reconstructed Spectrogram')
        plt.colorbar(img2, ax=ax2, format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'spectrogram_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 儲存詳細的分析結果
        analysis_results = {
            'file_info': {
                'original_path': str(original_path),
                'reconstructed_path': str(reconstructed_path),
                'sample_rate': sr_orig,
                'duration': len(y_orig) / sr_orig
            },
            'metrics': {
                'mse': float(mse),
                'max_error': float(max_error),
                'correlation': float(correlation),
                'snr': float(snr)
            },
            'plots': {
                'waveform': f'waveform_comparison_{timestamp}.png',
                'spectrogram': f'spectrogram_comparison_{timestamp}.png'
            }
        }

        # 儲存分析結果
        with open(plots_dir / f'analysis_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        return analysis_results

    except Exception as e:
        error_info = {
            'error': str(e),
            'original_path': str(original_path),
            'reconstructed_path': str(reconstructed_path)
        }
        # 儲存錯誤資訊
        error_log_path = Path(output_dir) / 'error_log.json'
        with open(error_log_path, 'a', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
            f.write('\n')
        
        raise Exception(f"分析過程發生錯誤: {str(e)}")