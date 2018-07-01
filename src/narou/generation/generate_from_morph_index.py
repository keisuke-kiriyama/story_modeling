from src.narou.corpus.narou_corpus import NarouCorpus



if __name__ == '__main__':
    corpus = NarouCorpus()
    # corpus.contents(ncode='n1970')
    print(repr(corpus.synopsis(ncode='n1970en')))
