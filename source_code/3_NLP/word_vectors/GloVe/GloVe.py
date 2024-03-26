import numpy as np

# コーパスから共起行列を生成する擬似関数
def build_cooccurrence_matrix(corpus, vocab, window_size=5):
    cooccurrences = np.zeros((len(vocab), len(vocab)), dtype=np.float64)
    for doc in corpus:
        tokens = doc.split()  # 単純化のため、単語分割は空白で区切る
        for i, token in enumerate(tokens):
            token_idx = vocab[token]
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_word = tokens[j]
                    context_idx = vocab[context_word]
                    cooccurrences[token_idx, context_idx] += 1
    return cooccurrences

# 重み付け関数
def weight_function(x, x_max=100, alpha=0.75):
    return (x / x_max) ** alpha if x < x_max else 1

# 勾配降下法で単語ベクトルを最適化
def train_glove(cooccurrences, vector_size=100, iterations=100, learning_rate=0.05):
    vocab_size = cooccurrences.shape[0]
    W = np.random.rand(vocab_size, vector_size)
    biases = np.random.rand(vocab_size)
    
    for _ in range(iterations):
        for i in range(vocab_size):
            for j in range(vocab_size):
                if cooccurrences[i, j] > 0:
                    weight = weight_function(cooccurrences[i, j])
                    diff = (W[i].dot(W[j]) + biases[i] + biases[j] - np.log(cooccurrences[i, j]))
                    cost = weight * (diff ** 2)
                    
                    # 勾配を計算
                    grad_wi = weight * diff * W[j]
                    grad_wj = weight * diff * W[i]
                    grad_bi = weight * diff
                    grad_bj = weight * diff
                    
                    # パラメータを更新
                    W[i] -= learning_rate * grad_wi
                    W[j] -= learning_rate * grad_wj
                    biases[i] -= learning_rate * grad_bi
                    biases[j] -= learning_rate * grad_bj
                    
    return W, biases
