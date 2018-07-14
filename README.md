# Story Modeling

## setup
```
pip install -r requirements.txt
```

## NarouCorpus
- 「小説を読もう」コーパスを扱う際に用いる

## KerasExtractiveSummarizer
- 重要文抽出によるKerasベースの要約器
### Usage
```
summarizer = KerasExtractiveSummarizer()
    summarizer.fit()
    summarizer.evaluate_mse()
    summarizer.show_training_process()
```

## NarouCorpusToEmbedding
- 「小説を読もう」コーパスの単語分散表現の獲得
### setup
```
>>> corpus_to_embedding = NarouCorpusToEmbedding(is_data_updated=True,
                                    is_embed=True,
                                    is_tensor_refresh=False,
                                    embedding_size=200)
```
- is_data_updated: データをアップデートした際にNarouCorpusToEmbedding.wakati_sentencesとNarouCorpusToEmbedding.morph_setを更新
- is_embed: embedding_modelの再学習を行うか
- is_embed: 訓練に用いるテンソルを作り直すか
- embedding_size: 単語ベクトルのサイズ
### 学習済みモデル
- 学習済みモデル
```
model/narou/narou_embedding.model
```
- 単語ベクトルの取得
```
self.word_embedding_model.__dict__['wv'][word]
```

