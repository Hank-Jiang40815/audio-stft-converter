# 音訊 STFT/iSTFT 轉換工具

這個專案提供了一個簡單易用的 Python 套件，用於執行音訊檔案的短時傅立葉轉換 (STFT) 與逆短時傅立葉轉換 (iSTFT) 處理。

## 功能特點

- 讀取常見音訊格式 (wav, mp3, flac 等)
- 執行短時傅立葉轉換 (STFT)
- 執行逆短時傅立葉轉換 (iSTFT)
- 將處理後的音訊儲存為音訊檔案
- 支援多種視窗函數 (漢明窗、漢寧窗、布萊克曼窗等)
- 可調整的視窗大小和重疊率
- 頻譜修改功能
- 時頻譜圖視覺化

## 安裝

```bash
# 從 GitHub 安裝
git clone https://github.com/Hank-Jiang40815/audio-stft-converter.git
cd audio-stft-converter
pip install -e .
```

## 使用示例

```python
import audio_stft_converter as asc

# 載入音訊檔案
audio_loader = asc.io.AudioIO()
audio_data, sample_rate = audio_loader.load('example.wav')

# STFT 轉換
stft_processor = asc.core.STFTProcessor(window_size=2048, hop_length=512, window_type='hann')
stft_matrix = stft_processor.stft(audio_data)

# 視覺化
visualizer = asc.utils.Visualizer()
visualizer.plot_spectrogram(stft_matrix, sample_rate, stft_processor.hop_length)

# iSTFT 轉換
reconstructed_audio = stft_processor.istft(stft_matrix)

# 計算 SNR
metrics = asc.utils.Metrics()
snr = metrics.compute_snr(audio_data, reconstructed_audio)
print(f'重建音訊 SNR: {snr:.2f} dB')

# 儲存重建的音訊
audio_loader.save('reconstructed.wav', reconstructed_audio, sample_rate)
```

## Docker 使用

```bash
# 建立 Docker 映像
docker build -t audio-stft-converter .

# 執行 Docker 容器
docker run -it audio-stft-converter
```

## 文件

詳細的 API 文件和更多使用示例請參考 [examples](./examples) 目錄。

## 授權

MIT License
