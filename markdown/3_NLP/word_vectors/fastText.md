2. FastTex [->Github](https://github.com/facebookresearch/fastText)

    2.1 FastTextは何ですか。
        
        FastTextは、Word2Vecのアイデアを拡張して開発された、単語の埋め込みを生成するためのライブラリです。

    2.2 word2vecとの違いはいくつあります。

        １。サブワード情報の利用：FastTextは、単語をより小さな単位（サブワード、具体的にはn-gram）に分割し、これらのサブワードの埋め込みを学習します。*その後、単語の最終的なベクトル表現は、その構成サブワードのベクトルの合計または平均によって得られます*。このアプローチにより、未知の単語やレアワードに対しても表現を生成することができ、形態素の豊富な言語や新しい単語が頻繁に生成されるドメインでの性能が向上します。Word2Vecでは、単語全体を最小単位として扱い、単語レベルでのみベクトル表現を学習します。そのため、訓練データに存在しない単語のベクトルを生成することはできません。

        2. 未知語やレアワードへの対応：FastTextは未知語やレアワードに強いです。サブワード情報を利用することで、これらの単語の意味を推測し、適切なベクトル表現を生成することができます。Word2Vecは、学習時に見た単語に対してのみベクトル表現を持っているため、未知語に対しては直接的な対応が困難です。

        3. 計算コストとパフォーマンス：FastTextはサブワード情報を利用するため、Word2Vecに比べて訓練にはより多くの計算リソースを必要とすることがあります。しかし、その結果として得られる単語の埋め込みは、より情報豊富であり、特に形態素が豊富な言語や未知語、レアワードの処理において優れた性能を発揮します。Word2Vecは比較的シンプルで、計算コストはFastTextよりも低いですが、未知語やレアワードへの対応能力には限界があります。

    2.3 FastTextソースコード解析（モデル）：
    
    [-> 論文の補充1: Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606v1.pdf)

        => main points: Use the sub-word to represent the words can be a good solution to the oov.

        => Experiments: Performance on Word Similarity and Analogy Tasks: They evaluated how well the word vectors generated by their method and the traditional methods performed on standard linguistic tasks. These tasks often involve measuring the similarity between words or solving word analogies (e.g., "man" is to "woman" as "king" is to "queen"). The goal was to see if the subword information could lead to better understanding and representation of words, especially for languages with rich morphology.

        Ability to Handle Rare and Out-of-Vocabulary (OOV) Words: A significant part of the comparison was to assess how effectively each method could deal with rare words or words that were not present in the training data. Since the new method uses subword information, it was hypothesized to be better at handling such words by leveraging their morphological structure (e.g., prefixes, suffixes).

        Performance Across Different Languages: The experiments were not limited to English but included several languages with varying morphological complexity. This comparison aimed to demonstrate the versatility and effectiveness of the subword information approach across different linguistic contexts.
        
        => other points:
        - Word representations trained on large unlabeled corpora are useful for natural language processing tasks.
        - Existing models for learning word representations ignore the morphology of words, which is a limitation for morphologically rich languages with large vocabularies and many rare words.
        - The proposed model represents each word as a bag of character n-grams and associates a vector representation to each n-gram.
        - The vector representation of a word is obtained by summing the representations of its n-grams.
        - The proposed model is fast and allows for training on large corpora quickly.
        - The authors evaluate the model on five different languages and show its benefits in capturing word similarity and analogy.
        - The model outperforms baseline models in word similarity and analogy tasks.

    [-> 論文の補充2: Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759v2.pdf)




    


    
    
    
    2.4 Fasttextの実践
    
    [Yahoo!ニュースをクラスタリング](https://qiita.com/kei0919/items/3059c336c3d0e2228830)
    
    [fastText Japanese Tutorial](https://github.com/icoxfog417/fastTextJapaneseTutorial)


    2.４ FastTextの欠点はなんですか。

        1. 計算資源の要求が高い
        2. モデルサイズが大きい
        3. 未知語への過剰適応
        4. 言語による違い
        5. コンテキストの無視

    









        
    



