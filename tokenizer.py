from tensorflow import keras
import collections
from hashlib import md5

class ChineseCharAndWordTokenizer:
    def __init__(self, vocab_size=100000, oov='[UNK]', oov_size=500000, hash_trick=None):
        self._oov = oov
        self._token2id = {oov: 0}
        self._id2token = {0: oov}
        self._word2token = {}
        self._vocab_size = vocab_size
        self._oov_size = oov_size

        if hash_trick == 'hash':
            self._hash_trick = hash
        elif hash_trick == 'md5':
            self._hash_trick = lambda _: int(md5(_.encode()).hexdigest(), 16) % self._oov_size
        else:
            self._hash_trick = hash_trick

    def load_vocab(self, vocab_path, skip=1):
        token_counter = collections.Counter()
        with open(vocab_path) as f:
            for _ in range(skip):
                f.readline()
            for line in f:
                keyword, tokens = line.strip('\r\n').split('\t')
                self._word2token[keyword] = tokens
                word_tokens, char_unigram, char_bigram, char_trigram = self.tokenize(keyword)

                token_counter.update(word_tokens)
                token_counter.update(char_unigram)
                token_counter.update(char_bigram)
                token_counter.update(char_trigram)

        tokens = token_counter.most_common(self._vocab_size)
        self._token2id = dict((token[0], id + 1) for id, token in enumerate(tokens))
        self._id2token = dict((id, token) for token, id in self._token2id.items())

    def tokenize(self, text, type=None):
        tokens = self._word2token.get(text, text).split(' ')

        word_tokens = tokens
        char_unigram = ''.join(word_tokens)

        char_bigram = '#'.join([''] + tokens + [''])
        char_bigram = zip(char_bigram[:-1], char_bigram[1:])
        char_bigram = list(map(lambda _: ''.join(_), char_bigram))

        char_trigram = '##'.join([''] + tokens + [''])
        char_trigram = zip(char_trigram[:-2], char_trigram[1:-1], char_trigram[2:])
        char_trigram = list(map(lambda _: ''.join(_), char_trigram))
        return word_tokens, char_unigram, char_bigram, char_trigram

    def token2ids(self, tokens):
        ids = []
        for token in tokens:
            id = self._token2id.get(token, 0)
            if id == 0 and self._hash_trick is not None:
                id = self._hash_trick(token) + self._vocab_size + 1
            ids.append(id)
        return ids

    def ids2token(self, ids):
        return [self._id2token.get(_, self._oov) for _ in ids]


if __name__ == "__main__":
    import sys
    tokenizer = ChineseCharAndWordTokenizer(hash_trick='md5')
    tokenizer.load_vocab(sys.argv[1])
    word_tokens, char_unigram, char_bigram, char_trigram = tokenizer.tokenize(sys.argv[2])
    print(word_tokens)
    tokens = tokenizer.token2ids(word_tokens)
    print(tokens)
    print(' '.join(tokenizer.ids2token(tokens)))
