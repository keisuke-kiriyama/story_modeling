from src.narou.corpus.narou_corpus import NarouCorpus



if __name__ == '__main__':
    corpus = NarouCorpus()
    corpus.contents_to_tensor(0)
    # print(corpus.contents(ncode='n1161ds'))
    # print(repr(corpus.synopsis(ncode='n1161ds')))
    # print(corpus.contents_from_file_path('/Users/kiriyama.k.ab/Fujii.lab/story_modeling/data/narou/meta/n1875cl_meta.json'))
    # print(corpus.synopsis_from_file_path('/Users/kiriyama.k.ab/Fujii.lab/story_modeling/data/narou/meta/n1875cl_meta.json'))