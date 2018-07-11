# Story Modeling

## setup
```
pip install -r requirements.txt
```

## NarouCorpusToEmbedding
- 「小説を読もう」の分散表現
### setup
```
>>> from narou_corpus import NarouCorpus
>>> corpus = NarouCorpusToEmbedding(is_data_updated=True,
                                    is_embed=True,
                                    is_tensor_refresh=False,
                                    embedding_size=200)
```
- is_data_updated: データをアップデートした際にNarouCorpus.wakati_sentencesとNarouCorpus.morph_setを更新
- is_embed: embedding_modelの再学習を行うか
- is_embed: 訓練に用いるテンソルを作り直すか
- embedding_size: 単語ベクトルのサイズ
### 学習データの取得
- data_to_tensor_emb_idx()
    - X: 単語の分散ベクトルの系列
    - Y: あらすじの形態素インデックスの系列
```
X, Y = corpus.data_to_tensor_emb_idx(contents_length=contents_length,
                                     synopsis_length=synopsis_length)
```
- contents_length: 本文の最大単語数
- synopsis_length: あらすじの最大単語数

## ディレクトリ構造
- analysis
    - bccwj
        - bccwjの分析結果
- data
    - aozora
        - 青空文庫関連のデータ
    - bccwj
        - BCCWJ関連のデータ
- model
    - narou
        - 言語モデル等の格納
- src
    - aozora
        - downloader
            - text_downloader.py
                - URLからテキストをダウンロード
            - new_pseudonym_text_downloader.py
                - 新字新仮名の書籍のテキストをダウンロード
        - extractor
            - new_pseudonym_extractor.py
                - 新字新仮名の書籍情報の抽出
            - add_length_new_pseudonym_file.py
                - list_new_pseudonym.csvのデータに各テキストの文字数のカラムを追加するスクリプト
        - pre_processing
            - remove_seudonym_reading.py
                - 本文を抽出し、読み仮名や注釈削除する前処理
        - text_tiling
            - text_tiling.py
                - テキストタイリングアルゴリズム
    - bccwj
        - evaluator
            - character_extract_evaluator.py
                - 登場人物抽出の評価
        - extractor
            - extract_literature.py
                - PBから文学作品を抽出
            - create_literature_bibliography.py
                - 文学のみの参考文献情報リストの作成
            - extract_persons.py
                - 登場人物を抽出するスクリプト
    - narou
        - corpus
            - narou_corpus.py
                - 小説を読もうを扱う際のコーパス
        - downloader
            - contents_meta_validation.py
                - 本文あるがあらすじない場合に欠損を補完する
        - generation
            - generate_from_morph_index
                - input: 単語の分散表現
                - output: あらすじ単語のインデックス
    - util
        - ユーティリティ
- temp
    - log
        - new_pseudonym_text_download_error.log
            - src/downloader/new_pseudonym_text_downloader.pyのエラーログ
    - temp_data
        - 一時的に使用するデータ等を格納
    - temp_src
        - 仮のコード


