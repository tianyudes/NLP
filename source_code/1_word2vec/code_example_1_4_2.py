from janome.tokenizer import Tokenizer
from gensim.models import Word2Vec
import string

# 日本語のテキストファイルを読み込み
file_name = 'text.txt'

# Janomeのトークナイザー
tokenizer = Tokenizer()

# ファイルからテキストを読み込み、分かち書き
with open(file_name, 'r', encoding='utf-8') as file:
    text = file.read()
    words = [token.surface for token in tokenizer.tokenize(text)]

# 分かち書きされた単語のリストを用意
sentences = [words]  # 複数の文があれば、それぞれをリストに追加

# Word2Vecモデルの訓練
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=1)

# モデルの使用例（例えば、「コンピュータ」の単語ベクトルを取得）
word_vector = model.wv['コンピュータ']
print(word_vector)
