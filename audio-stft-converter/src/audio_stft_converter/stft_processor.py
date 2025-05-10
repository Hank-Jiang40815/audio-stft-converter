#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""STFT 處理器模組。

This module provides the core functionality for STFT processing.
"""
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

class STFTProcessor:
    """STFT 處理器類別。"""

    def __init__(self, n_fft: int = 2048, hop_length: int = 512, win_length: int = None):
        """初始化 STFT 處理器。

        Args:
            n_fft (int, optional): FFT 的點數. Defaults to 2048.
            hop_length (int, optional): STFT 的 hop length. Defaults to 512.
            win_length (int, optional): STFT 的窗長. Defaults to n_fft.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def process_directory(self, input_dir: str, output_dir: str) -> dict:
        """處理指定目錄中的所有 WAV 檔案（包含子目錄）。

        Args:
            input_dir (str): 輸入目錄路徑
            output_dir (str): 輸出目錄路徑

        Returns:
            dict: 處理結果，包含：
                - timestamp: 處理時間戳記
                - processed_files: 已處理檔案列表及其相關資訊
                - parameters: 處理參數
        """
        # 確保使用 Path 物件處理路徑，以支援中文字符
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 建立輸出目錄
        stft_dir = output_path / f'stft_{self.exp_id}'
        reconstructed_dir = output_path / f'reconstructed_{self.exp_id}'
        stft_dir.mkdir(parents=True, exist_ok=True)
        reconstructed_dir.mkdir(parents=True, exist_ok=True)

        # 遞迴搜尋所有 WAV 檔案（包括大小寫副檔名）
        wav_files = list(input_path.rglob('*.wav'))
        wav_files.extend(input_path.rglob('*.WAV'))

        processed_files = []
        
        # 使用 tqdm 顯示進度
        for wav_file in tqdm(wav_files, desc="處理音訊檔案"):
            try:
                # 保持相對路徑結構
                rel_path = wav_file.relative_to(input_path)
                
                # 建立對應的輸出目錄結構
                stft_output_dir = stft_dir / rel_path.parent
                recon_output_dir = reconstructed_dir / rel_path.parent
                stft_output_dir.mkdir(parents=True, exist_ok=True)
                recon_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 進行 STFT 轉換
                stft_result = self._process_file(str(wav_file))
                
                # 儲存 STFT 結果
                stft_path = stft_output_dir / f'{wav_file.stem}_stft_{self.exp_id}.npy'
                np.save(stft_path, stft_result)
                
                # 重建並儲存音訊
                recon_audio = self._reconstruct_audio(stft_result)
                recon_path = recon_output_dir / f'{wav_file.stem}_reconstructed_{self.exp_id}.wav'
                
                # 使用原始音訊的取樣率
                orig_sr = librosa.get_samplerate(str(wav_file))
                sf.write(recon_path, recon_audio, orig_sr)

                # 記錄處理結果
                processed_files.append({
                    'original_path': str(wav_file),
                    'relative_path': str(rel_path),
                    'stft_path': str(stft_path),
                    'reconstructed_path': str(recon_path),
                    'sample_rate': orig_sr,
                    'status': 'success'
                })

            except Exception as e:
                processed_files.append({
                    'original_path': str(wav_file),
                    'status': 'error',
                    'error_message': str(e)
                })
                print(f"\n處理檔案 {wav_file} 時發生錯誤: {e}")

        # 儲存處理結果
        results = {
            'timestamp': self.exp_id,
            'processed_files': processed_files,
            'parameters': {
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'win_length': self.win_length
            },
            'statistics': {
                'total_files': len(processed_files),
                'successful': len([f for f in processed_files if f['status'] == 'success']),
                'failed': len([f for f in processed_files if f['status'] == 'error'])
            }
        }
        
        # 儲存詳細結果報告
        results_path = output_path / f'processing_results_{self.exp_id}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def _process_file(self, file_path: str) -> np.ndarray:
        """處理單一音訊檔案。

        Args:
            file_path (str): 音訊檔案路徑

        Returns:
            np.ndarray: STFT 結果
        """
        # 取得原始取樣率
        sr = librosa.get_samplerate(file_path)
        
        # 載入音訊，保持原始取樣率
        y, _ = librosa.load(file_path, sr=sr)
        
        # 計算 STFT
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, 
                        win_length=self.win_length)
        return D

    def _reconstruct_audio(self, D: np.ndarray) -> np.ndarray:
        """從 STFT 結果重建音訊。

        Args:
            D (np.ndarray): STFT 結果

        Returns:
            np.ndarray: 重建的音訊訊號
        """
        # 使用 iSTFT 重建音訊
        y_recon = librosa.istft(D, hop_length=self.hop_length, 
                               win_length=self.win_length)
        return y_recon