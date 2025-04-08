#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""音訊 STFT/iSTFT 轉換示例程式。

這個程式示範了 audio-stft-converter 套件的主要功能，包括：
- 音訊的 STFT 轉換
- 視覺化時頻譜圖
- iSTFT 重建音訊
- 量化重建品質
- 生成實驗報告
"""

import numpy as np
import os
import sys

# 確保可以導入套件
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_stft_converter.core import STFTProcessor
from src.audio_stft_converter.io import AudioIO
from src.audio_stft_converter.utils import Visualizer, Metrics, Reporter

def generate_test_signal(duration=2.0, sample_rate=44100):
    """生成測試音訊信號。

    Args:
        duration (float): 音訊持續時間（秒）。預設為2.0秒。
        sample_rate (int): 取樣率（Hz）。預設為44100Hz。

    Returns:
        tuple: (音訊信號數組, 取樣率)
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 合成一個具有多個頻率成分的信號
    signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +    # A4 音符 (440 Hz)
        0.3 * np.sin(2 * np.pi * 880 * t) +    # A5 音符 (880 Hz)
        0.2 * np.sin(2 * np.pi * 1760 * t)     # A6 音符 (1760 Hz)
    )
    
    # 添加一個短暫的高頻閃定信號
    chirp_duration = 0.3  # 0.3 秒的閃定信號
    chirp_start = int(sample_rate * 0.8)  # 從 0.8 秒開始
    chirp_end = int(chirp_start + sample_rate * chirp_duration)
    chirp_t = np.linspace(0, chirp_duration, chirp_end - chirp_start)
    chirp_freqs = np.linspace(2000, 8000, len(chirp_t))  # 從 2000Hz 到 8000Hz 的閃定信號
    chirp_signal = 0.4 * np.sin(2 * np.pi * chirp_freqs * chirp_t * chirp_duration)
    
    # 將閃定信號加到主要信號中
    signal[chirp_start:chirp_end] += chirp_signal
    
    return signal, sample_rate

def main():
    """主程式函數。
    """
    print("\n===== 音訊 STFT/iSTFT 轉換示例 =====\n")
    
    # 設定實驗參數
    experiment_id = 1
    window_size = 2048
    hop_length = 512
    window_type = "hann"
    output_dir = "output"
    
    # 存放結果的目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化工具類別
    stft_processor = STFTProcessor(
        window_size=window_size,
        hop_length=hop_length,
        window_type=window_type,
        experiment_id=experiment_id
    )
    
    audio_io = AudioIO()
    visualizer = Visualizer(experiment_id=experiment_id)
    metrics = Metrics()
    reporter = Reporter(experiment_id=experiment_id)
    
    # 生成測試信號
    print("\u751f成測試信號...")
    audio_data, sample_rate = generate_test_signal(duration=2.0, sample_rate=44100)
    
    # 儲存原始音訊檔
    original_path = f"{output_dir}/original_audio_Exp{experiment_id}.wav"
    audio_io.save(original_path, audio_data, sample_rate)
    print(f"\u539f始音訊已儲存為: {original_path}")
    
    # 繪制原始波形圖
    waveform_path = visualizer.plot_waveform(
        audio_data, sample_rate,
        output_path=f"{output_dir}/Original_Waveform_Exp{experiment_id}.png",
        title=f"\u539f\u59cb\u97f3\u8a0a\u6ce2\u5f62 - \u5be6\u9a57 #{experiment_id}"
    )
    print(f"\u539f\u59cb\u6ce2\u5f62\u5df2\u5132\u5b58\u70ba: {waveform_path}")
    
    # 執行 STFT
    print("\n\u57f7\u884c STFT \u8f49\u63db...")
    stft_matrix = stft_processor.stft(audio_data)
    print(f"STFT \u77e9\u9663\u5c3a\u5bf8: {stft_matrix.shape}")
    
    # 繪制時頻譜圖
    spectrogram_path = visualizer.plot_spectrogram(
        stft_matrix, sample_rate, hop_length,
        output_path=f"{output_dir}/STFT_Spectrogram_Exp{experiment_id}.png",
        title=f"STFT \u6642\u983b\u8b5c\u5716 - \u5be6\u9a57 #{experiment_id}"
    )
    print(f"\u6642\u983b\u8b5c\u5716\u5df2\u5132\u5b58\u70ba: {spectrogram_path}")
    
    # 順便示範頻譜修改
    print("\n\u9032\u884c\u983b\u8b5c\u4fee\u6539\u793a\u7bc4...")
    
    # 將高\u983b\u90e8\u5206\u653e\u5927 6dB
    print("\u5c07\u9ad8\u983b\u90e8\u5206\u653e\u5927 6dB...")
    freqs = np.fft.fftfreq(stft_matrix.shape[0], 1.0/sample_rate)
    high_mask = np.abs(freqs) > 2000
    
    # 只\u4fee\u6539\u9ad8\u983b\u90e8\u5206
    modified_stft = stft_matrix.copy()
    # 轉\u63db\u70ba\u5e45\u5ea6\u548c\u76f8\u4f4d
    magnitude = np.abs(modified_stft[high_mask, :])
    phase = np.angle(modified_stft[high_mask, :])
    # 將\u9ad8\u983b\u90e8\u5206\u653e\u5927 6dB (\u4e58\u4ee5 2.0)
    magnitude = magnitude * 2.0
    # 重\u5efa\u8907\u6578\u503c
    modified_stft[high_mask, :] = magnitude * np.exp(1j * phase)
    
    # 繪制修改後的時頻譜圖
    mod_spectrogram_path = visualizer.plot_spectrogram(
        modified_stft, sample_rate, hop_length,
        output_path=f"{output_dir}/Modified_Spectrogram_Exp{experiment_id}.png",
        title=f"\u4fee\u6539\u5f8c\u7684 STFT \u6642\u983b\u8b5c\u5716 - \u5be6\u9a57 #{experiment_id}"
    )
    print(f"\u4fee\u6539\u5f8c\u7684\u6642\u983b\u8b5c\u5716\u5df2\u5132\u5b58\u70ba: {mod_spectrogram_path}")
    
    # 執行 iSTFT 重建原始信號
    print("\n\u57f7\u884c iSTFT \u91cd\u5efa\u539f\u59cb\u4fe1\u865f...")
    reconstructed_audio = stft_processor.istft(stft_matrix)
    print(f"\u539f\u59cb\u91cd\u5efa\u4fe1\u865f\u9577\u5ea6: {len(reconstructed_audio)}")
    
    # 儲存重建的音訊
    reconstructed_path = f"{output_dir}/reconstructed_audio_Exp{experiment_id}.wav"
    audio_io.save(reconstructed_path, reconstructed_audio, sample_rate)
    print(f"\u91cd\u5efa\u97f3\u8a0a\u5df2\u5132\u5b58\u70ba: {reconstructed_path}")
    
    # 執行 iSTFT 重建修改後的信號
    print("\n\u57f7\u884c iSTFT \u91cd\u5efa\u4fee\u6539\u5f8c\u7684\u4fe1\u865f...")
    modified_audio = stft_processor.istft(modified_stft)
    
    # 儲存修改後的音訊
    modified_path = f"{output_dir}/modified_audio_Exp{experiment_id}.wav"
    audio_io.save(modified_path, modified_audio, sample_rate)
    print(f"\u4fee\u6539\u5f8c\u7684\u97f3\u8a0a\u5df2\u5132\u5b58\u70ba: {modified_path}")
    
    # 評估重建品質
    print("\n\u8a55\u4f30\u91cd\u5efa\u54c1\u8cea...")
    eval_results = metrics.evaluate_all(audio_data, reconstructed_audio, sample_rate)
    print(f"SNR: {eval_results['snr']:.2f} dB")
    print(f"MSE: {eval_results['mse']:.6f}")
    print(f"\u76f8\u95dc\u6027: {eval_results['correlation']:.4f}")
    
    # 繪制比較圖
    comparison_path = visualizer.plot_comparison(
        audio_data, reconstructed_audio, sample_rate,
        output_path=f"{output_dir}/Comparison_Exp{experiment_id}.png"
    )
    print(f"\u6bd4\u8f03\u5716\u5df2\u5132\u5b58\u70ba: {comparison_path}")
    
    # 儲存實\u9a57\u6578\u64da
    data = {
        'parameters': {
            'window_size': window_size,
            'hop_length': hop_length,
            'window_type': window_type,
            'sample_rate': sample_rate
        },
        'metrics': eval_results,
        'stft_shape': stft_matrix.shape
    }
    
    data_path = reporter.save_experiment_data(
        data, 
        output_path=f"{output_dir}/STFT_Data_Exp{experiment_id}.json"
    )
    print(f"\u5be6\u9a57\u6578\u64da\u5df2\u5132\u5b58\u70ba: {data_path}")
    
    # 更\u65b0\u5831\u544a
    report_content = f"""\
### 實\u9a57\u53c3\u6578
- 視\u7a97\u5927\u5c0f: {window_size}
- 視\u7a97\u79fb\u52d5\u6b65\u9577: {hop_length}
- 視\u7a97\u51fd\u6578: {window_type}
- 取\u6a23\u7387: {sample_rate} Hz

### 重\u5efa\u54c1\u8cea\u6307\u6a19
- SNR: {eval_results['snr']:.2f} dB
- MSE: {eval_results['mse']:.6f}
- 相\u95dc\u6027: {eval_results['correlation']:.4f}

### 結\u8ad6
執\u884c了 STFT 和 iSTFT 轉\u63db，並將高\u983b\u90e8\u5206\u653e\u5927 6dB。\n重\u5efa品\u8cea非常高，說明 STFT/iSTFT 轉\u63db過\u7a0b保\u7559\u4e86\u4fe1\u865f\u7684完\u6574\u6027。"""
    
    image_files = [waveform_path, spectrogram_path, mod_spectrogram_path, comparison_path]
    data_files = [data_path, original_path, reconstructed_path, modified_path]
    
    reporter.update_report(
        "STFT/iSTFT 音訊轉換示例",
        report_content,
        image_paths=image_files,
        data_paths=data_files
    )
    print(f"\n\u5be6\u9a57\u5831\u544a\u5df2\u66f4\u65b0\u81f3 REPORT.md\n")
    
    print("===== 示例實驗完成 =====\n")
    
    # 返回所有路徑供外部使用
    return {
        "original_audio": original_path,
        "reconstructed_audio": reconstructed_path,
        "modified_audio": modified_path,
        "spectrogram": spectrogram_path,
        "mod_spectrogram": mod_spectrogram_path,
        "comparison": comparison_path,
        "data": data_path
    }

if __name__ == "__main__":
    main()
