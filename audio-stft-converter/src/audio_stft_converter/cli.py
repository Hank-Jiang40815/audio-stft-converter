#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""命令列介面模組。

This module provides the CLI interface for the audio STFT converter.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from audio_stft_converter.stft_processor import STFTProcessor
from audio_stft_converter.analysis import analyze_reconstruction

def main():
    """主程式進入點。"""
    parser = argparse.ArgumentParser(description='處理音訊檔案的 STFT 轉換工具')
    parser.add_argument('input_dir', help='輸入資料夾路徑，包含要處理的 .wav 音檔')
    parser.add_argument('output_dir', help='輸出資料夾路徑，用於儲存處理結果')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT 點數 (預設: 2048)')
    parser.add_argument('--hop_length', type=int, default=512, help='STFT 訊框移動點數 (預設: 512)')
    parser.add_argument('--win_length', type=int, help='STFT 視窗長度 (預設: 與 n_fft 相同)')
    parser.add_argument('--validate', action='store_true', help='處理完成後執行驗證分析')
    parser.add_argument('--validation_output', help='驗證結果輸出目錄 (預設: 與 output_dir 相同)')

    args = parser.parse_args()

    # 驗證輸入路徑
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f'錯誤：輸入路徑 {args.input_dir} 不存在')
        sys.exit(1)

    # 確保輸出路徑存在
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 建立處理器實例
    processor = STFTProcessor(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length or args.n_fft
    )

    print(f'\n開始處理音訊檔案...')
    print(f'輸入目錄: {input_path}')
    print(f'輸出目錄: {output_path}')
    print(f'參數設定:')
    print(f'  - FFT 點數: {args.n_fft}')
    print(f'  - Hop Length: {args.hop_length}')
    print(f'  - Window Length: {args.win_length or args.n_fft}\n')

    # 處理音檔
    try:
        results = processor.process_directory(args.input_dir, args.output_dir)
        print(f'\n處理完成！')
        print(f'處理結果統計：')
        print(f'  - 總檔案數: {results["statistics"]["total_files"]}')
        print(f'  - 成功處理: {results["statistics"]["successful"]}')
        print(f'  - 處理失敗: {results["statistics"]["failed"]}')
        print(f'詳細結果已儲存至: {output_path}/processing_results_{results["timestamp"]}.json')

        # 如果有檔案處理失敗，列出失敗的檔案
        failed_files = [f for f in results['processed_files'] if f['status'] == 'error']
        if failed_files:
            print('\n處理失敗的檔案：')
            for f in failed_files:
                print(f'  - {f["original_path"]}: {f["error_message"]}')

    except Exception as e:
        print(f'\n錯誤：處理過程中發生異常: {str(e)}')
        sys.exit(1)

    # 如果啟用驗證，執行驗證分析
    if args.validate:
        validation_output = args.validation_output or args.output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_dir = Path(validation_output) / f'validation_{timestamp}'
        validation_dir.mkdir(parents=True, exist_ok=True)

        print('\n開始進行驗證分析...')
        validation_results = []
        
        for file_info in results['processed_files']:
            if file_info['status'] == 'success':
                try:
                    result = analyze_reconstruction(
                        file_info['original_path'],
                        file_info['reconstructed_path'],
                        str(validation_dir)
                    )
                    validation_results.append({
                        'file': file_info['relative_path'],
                        'metrics': result
                    })
                    print(f'驗證完成: {file_info["relative_path"]} (MSE: {result["mse"]:.6f})')
                except Exception as e:
                    print(f'驗證失敗: {file_info["relative_path"]} - {str(e)}')

        print(f'\n驗證完成！結果已儲存至 {validation_dir}')

if __name__ == '__main__':
    main()