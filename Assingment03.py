#Name : Rahul Sunil Kolase
#Roll No: 34
#Assignment : 03

from nltk.util import ngrams
#unigram model
n = 1
sentence = 'While unigram model sentences will only exclude the UNK token, models will also exclude all other words already in the sentence.NTK provides another function everygrams that converts a sentence into unigram, bigram, trigram, and so on till the ngrams, where n is the length of the sentence. In short, this function generates ngrams for all possible values of n.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)
#bigram model
n = 2
sentence = 'While unigram model sentences will only exclude the UNK token, models will also exclude all other words already in the sentence.NTK provides another function everygrams that converts a sentence into unigram, bigram, trigram, and so on till the ngrams, where n is the length of the sentence. In short, this function generates ngrams for all possible values of n.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)
#trigram model
n = 3
sentence = 'While unigram model sentences will only exclude the UNK token, models will also exclude all other words already in the sentence.NTK provides another function everygrams that converts a sentence into unigram, bigram, trigram, and so on till the ngrams, where n is the length of the sentence. In short, this function generates ngrams for all possible values of n.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)

#using text file input
from nltk import ngrams
file = open("/home/exam/nlpl.txt")
for i in file.readlines():
    cumulative = i
    sentences = i.split(".")
    counter = 0
    for sentence in sentences:
        print("For sentence", counter + 1, ", trigrams are: ")
        trigrams = ngrams(sentence.split(" "), 3)
        for grams in trigrams:
            print(grams)
        counter += 1
        print()
        
#output
'''('While',)
('unigram',)
('model',)
('sentences',)
('will',)
('only',)
('exclude',)
('the',)
('UNK',)
('token,',)
('models',)
('will',)
('also',)
('exclude',)
('all',)
('other',)
('words',)
('already',)
('in',)
('the',)
('sentence.NTK',)
('provides',)
('another',)
('function',)
('everygrams',)
('that',)
('converts',)
('a',)
('sentence',)
('into',)
('unigram,',)
('bigram,',)
('trigram,',)
('and',)
('so',)
('on',)
('till',)
('the',)
('ngrams,',)
('where',)
('n',)
('is',)
('the',)
('length',)
('of',)
('the',)
('sentence.',)
('In',)
('short,',)
('this',)
('function',)
('generates',)
('ngrams',)
('for',)
('all',)
('possible',)
('values',)
('of',)
('n.',)
('While', 'unigram')
('unigram', 'model')
('model', 'sentences')
('sentences', 'will')
('will', 'only')
('only', 'exclude')
('exclude', 'the')
('the', 'UNK')
('UNK', 'token,')
('token,', 'models')
('models', 'will')
('will', 'also')
('also', 'exclude')
('exclude', 'all')
('all', 'other')
('other', 'words')
('words', 'already')
('already', 'in')
('in', 'the')
('the', 'sentence.NTK')
('sentence.NTK', 'provides')
('provides', 'another')
('another', 'function')
('function', 'everygrams')
('everygrams', 'that')
('that', 'converts')
('converts', 'a')
('a', 'sentence')
('sentence', 'into')
('into', 'unigram,')
('unigram,', 'bigram,')
('bigram,', 'trigram,')
('trigram,', 'and')
('and', 'so')
('so', 'on')
('on', 'till')
('till', 'the')
('the', 'ngrams,')
('ngrams,', 'where')
('where', 'n')
('n', 'is')
('is', 'the')
('the', 'length')
('length', 'of')
('of', 'the')
('the', 'sentence.')
('sentence.', 'In')
('In', 'short,')
('short,', 'this')
('this', 'function')
('function', 'generates')
('generates', 'ngrams')
('ngrams', 'for')
('for', 'all')
('all', 'possible')
('possible', 'values')
('values', 'of')
('of', 'n.')
('While', 'unigram', 'model')
('unigram', 'model', 'sentences')
('model', 'sentences', 'will')
('sentences', 'will', 'only')
('will', 'only', 'exclude')
('only', 'exclude', 'the')
('exclude', 'the', 'UNK')
('the', 'UNK', 'token,')
('UNK', 'token,', 'models')
('token,', 'models', 'will')
('models', 'will', 'also')
('will', 'also', 'exclude')
('also', 'exclude', 'all')
('exclude', 'all', 'other')
('all', 'other', 'words')
('other', 'words', 'already')
('words', 'already', 'in')
('already', 'in', 'the')
('in', 'the', 'sentence.NTK')
('the', 'sentence.NTK', 'provides')
('sentence.NTK', 'provides', 'another')
('provides', 'another', 'function')
('another', 'function', 'everygrams')
('function', 'everygrams', 'that')
('everygrams', 'that', 'converts')
('that', 'converts', 'a')
('converts', 'a', 'sentence')
('a', 'sentence', 'into')
('sentence', 'into', 'unigram,')
('into', 'unigram,', 'bigram,')
('unigram,', 'bigram,', 'trigram,')
('bigram,', 'trigram,', 'and')
('trigram,', 'and', 'so')
('and', 'so', 'on')
('so', 'on', 'till')
('on', 'till', 'the')
('till', 'the', 'ngrams,')
('the', 'ngrams,', 'where')
('ngrams,', 'where', 'n')
('where', 'n', 'is')
('n', 'is', 'the')
('is', 'the', 'length')
('the', 'length', 'of')
('length', 'of', 'the')
('of', 'the', 'sentence.')
('the', 'sentence.', 'In')
('sentence.', 'In', 'short,')
('In', 'short,', 'this')
('short,', 'this', 'function')
('this', 'function', 'generates')
('function', 'generates', 'ngrams')
('generates', 'ngrams', 'for')
('ngrams', 'for', 'all')
('for', 'all', 'possible')
('all', 'possible', 'values')
('possible', 'values', 'of')
('values', 'of', 'n.')
For sentence 1 , trigrams are: 
('There', 'are', 'many')
('are', 'many', 'variations')
('many', 'variations', 'of')
('variations', 'of', 'passages')
('of', 'passages', 'of')
('passages', 'of', 'Lorem')
('of', 'Lorem', 'Ipsum')
('Lorem', 'Ipsum', 'available,')
('Ipsum', 'available,', 'but')
('available,', 'but', 'the')
('but', 'the', 'majority')
('the', 'majority', 'have')
('majority', 'have', 'suffered')
('have', 'suffered', 'alteration')
('suffered', 'alteration', 'in')
('alteration', 'in', 'some')
('in', 'some', 'form,')
('some', 'form,', 'by')
('form,', 'by', 'injected')
('by', 'injected', 'humour,')
('injected', 'humour,', 'or')
('humour,', 'or', 'randomised')
('or', 'randomised', 'words')
('randomised', 'words', 'which')
('words', 'which', "don't")
('which', "don't", 'look')
("don't", 'look', 'even')
('look', 'even', 'slightly')
('even', 'slightly', 'believable')

For sentence 2 , trigrams are: 
('', 'If', 'you')
('If', 'you', 'are')
('you', 'are', 'going')
('are', 'going', 'to')
('going', 'to', 'use')
('to', 'use', 'a')
('use', 'a', 'passage')
('a', 'passage', 'of')
('passage', 'of', 'Lorem')
('of', 'Lorem', 'Ipsum,')
('Lorem', 'Ipsum,', 'you')
('Ipsum,', 'you', 'need')
('you', 'need', 'to')
('need', 'to', 'be')
('to', 'be', 'sure')
('be', 'sure', 'there')
('sure', 'there', "isn't")
('there', "isn't", 'anything')
("isn't", 'anything', 'embarrassing')
('anything', 'embarrassing', 'hidden')
('embarrassing', 'hidden', 'in')
('hidden', 'in', 'the')
('in', 'the', 'middle')
('the', 'middle', 'of')
('middle', 'of', 'text')

For sentence 3 , trigrams are: 
('', 'All', 'the')
('All', 'the', 'Lorem')
('the', 'Lorem', 'Ipsum')
('Lorem', 'Ipsum', 'generators')
('Ipsum', 'generators', 'on')
('generators', 'on', 'the')
('on', 'the', 'Internet')
('the', 'Internet', 'tend')
('Internet', 'tend', 'to')
('tend', 'to', 'repeat')
('to', 'repeat', 'predefined')
('repeat', 'predefined', 'chunks')
('predefined', 'chunks', 'as')
('chunks', 'as', 'necessary,')
('as', 'necessary,', 'making')
('necessary,', 'making', 'this')
('making', 'this', 'the')
('this', 'the', 'first')
('the', 'first', 'true')
('first', 'true', 'generator')
('true', 'generator', 'on')
('generator', 'on', 'the')
('on', 'the', 'Internet')

For sentence 4 , trigrams are: 
('', 'It', 'uses')
('It', 'uses', 'a')
('uses', 'a', 'dictionary')
('a', 'dictionary', 'of')
('dictionary', 'of', 'over')
('of', 'over', '200')
('over', '200', 'Latin')
('200', 'Latin', 'words,')
('Latin', 'words,', 'combined')
('words,', 'combined', 'with')
('combined', 'with', 'a')
('with', 'a', 'handful')
('a', 'handful', 'of')
('handful', 'of', 'model')
('of', 'model', 'sentence')
('model', 'sentence', 'structures,')
('sentence', 'structures,', 'to')
('structures,', 'to', 'generate')
('to', 'generate', 'Lorem')
('generate', 'Lorem', 'Ipsum')
('Lorem', 'Ipsum', 'which')
('Ipsum', 'which', 'looks')
('which', 'looks', 'reasonable')

For sentence 5 , trigrams are: 
('', 'The', 'generated')
('The', 'generated', 'Lorem')
('generated', 'Lorem', 'Ipsum')
('Lorem', 'Ipsum', 'is')
('Ipsum', 'is', 'therefore')
('is', 'therefore', 'always')
('therefore', 'always', 'free')
('always', 'free', 'from')
('free', 'from', 'repetition,')
('from', 'repetition,', 'injected')
('repetition,', 'injected', 'humour,')
('injected', 'humour,', 'or')
('humour,', 'or', 'non-characteristic')
('or', 'non-characteristic', 'words')
('non-characteristic', 'words', 'etc')'''