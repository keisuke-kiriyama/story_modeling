# Story Modeling

## セットアップ
```
pip install -r requirements.txt
```

## ディレクトリ構造
- data
    - aozora
        - 青空文庫関連のデータ
    - bccwj
        - BCCWJ関連のデータ
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

## データ
### 青空文庫
#### list_person_all.csv
- 人物ID
- 著者名
- 作品ID
- 作品名
- 仮名遣い種別
- 翻訳者名等
- 入力者名
- 校正者名
- 状態
- 状態の開始日
- 底本名
- 出版社名
- 入力に使用した版
- 校正に使用した版
#### list_person_all_utf8.csv
- list_person_all.csvのutf8
#### list_person_all_extended.csv
- 作品ID
- 作品名
- 作品名読み
- ソート用読み
- 副題
- 副題読み
- 原題
- 初出
- 分類番号
- 文字遣い種別
- 作品著作権フラグ
- 公開日
- 最終更新日
- 図書カードURL
- 人物ID
- 姓
- 名
- 姓読み
- 名読み
- 姓読みソート用
- 名読みソート用
- 姓ローマ字
- 名ローマ字
- 役割フラグ
- 生年月日
- 没年月日
- 人物著作権フラグ
- 底本名1
- 底本出版社名1
- 底本初版発行年1
- 入力に使用した版1
- 校正に使用した版1
- 底本の親本名1
- 底本の親本出版社名1
- 底本の親本初版発行年1
- 底本名2
- 底本出版社名2
- 底本初版発行年2
- 入力に使用した版2
- 校正に使用した版2
- 底本の親本名2
- 底本の親本出版社名2
- 底本の親本初版発行年2
- 入力者
- 校正者
- テキストファイルURL
- テキストファイル最終更新日
- テキストファイル符号化方式
- テキストファイル文字集合
- テキストファイル修正回数
- XHTML/HTMLファイルURL
- XHTML/HTMLファイル最終更新日
- XHTML/HTMLファイル符号化方式
- XHTML/HTMLファイル文字集合
- XHTML/HTMLファイル修正回数
#### list_person_all_extended_utf8.csv
- list_person_all_extended.csvのutf8
#### list_new_pseudonym.csv
##### 新字新仮名の文献
- 作品ID
- 作品名
- 作品名読み
- 文字遣い種別
- 人物ID
- 姓
- 名
- 姓読み
- 名読み
- テキストファイルURL
- XHTML/HTMLファイルURL
#### list_new_pseudonym_with_count.csv
##### list_new_pseudonym.csvに各テキストの文字数のカラムを追加
