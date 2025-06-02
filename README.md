# 2025/06/03 新人研修

Google colaboratory( https://colab.google/ ) を開く

適当なipynbファイルを作成
```sh
from google.colab import drive
drive.mount('/content/gdrive')

%cd /content/gdrive/MyDrive

%git clone https://github.com/a-tetsuya001/training.git
```
を実行

training/main.ipynb を実行する