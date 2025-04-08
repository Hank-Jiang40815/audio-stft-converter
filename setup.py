from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='audio-stft-converter',
    version='0.1.0',
    author='姜翼顥',
    author_email='example@email.com',
    description='音訊檔案的短時傅立葉轉換(STFT)與逆轉換(iSTFT)處理工具套件',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Hank-Jiang40815/audio-stft-converter',
    project_urls={
        'Bug Tracker': 'https://github.com/Hank-Jiang40815/audio-stft-converter/issues',
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'librosa>=0.9.0',
        'soundfile>=0.10.3',
        'matplotlib>=3.5.0',
        'tqdm>=4.62.0'
    ],
)
