# The code used to do prepare docs to align with bert pretrain usage. 
# It use spacy module to do sentences separation.
# Used below command to prepare the running code env.
# pip install spacy
# python -m spacy download en_core_web_lg

'''Example of adding a pipeline component to prohibit sentence boundaries
before certain tokens.

What we do is write to the token.is_sent_start attribute, which
takes values in {True, False, None}. The default value None allows the parser
to predict sentence segments. The value False prohibits the parser from inserting
a sentence boundary before that token. Note that fixing the sentence segmentation
should also improve the parse quality.

The specific example here is drawn from https://github.com/explosion/spaCy/issues/2627
Other versions of the model may not make the original mistake, so the specific
example might not be apt for future versions.
'''
import plac
import spacy
import codecs
import os

def prevent_sentence_boundaries(doc):
    for token in doc:
        if not can_be_sentence_start(token):
            token.is_sent_start = False
    return doc

def can_be_sentence_start(token):
    if token.i == 0:
        return True
    elif token.is_title:
        return True
    elif token.nbor(-1).is_punct:
        return True
    elif token.nbor(-1).is_space:
        return True
    else:
        return False

def main():
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe(prevent_sentence_boundaries, before='parser')
    nlp.max_length = 200000000
    input_dir = '/mnt/newcpfs/wikiextractor/wikiextractor/enwiki'
    for root, dirs, files in os.walk(input_dir):
        father, child = os.path.split(root)
        new_father = father+'_processed'
        new_root = os.path.join(new_father, child)
        if not os.path.exists(new_root):
            os.makedirs(new_root)
        for file in files:
            abs_file = os.path.join(root, file)
            new_abs_file = os.path.join(new_root, file)
            raw_text = open(abs_file).read().decode('utf8')
            fo = open("test.txt", "w")
            # doc = nlp(raw_text)
            # sentences = [sent.string.strip() for sent in doc.sents]
            # print(sentences)
            # nlp.add_pipe(prevent_sentence_boundaries, before='parser')
            doc = nlp(raw_text)
            sentences = [sent.string.strip() for sent in doc.sents]
            with codecs.open(new_abs_file ,'w', 'utf-8') as f:
                for sentence in sentences:
                    if '</doc>' in sentence:
                        f.write('\n')
                    elif '<doc id=' in sentence:
                        None
                    else:
                        f.write(sentence+'\n')
            print('Doc of ' + abs_file + ' Done!')
 
    
if __name__ == '__main__':
    plac.call(main)
