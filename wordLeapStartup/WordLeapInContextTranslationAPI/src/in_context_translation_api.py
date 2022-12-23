# Flask API related
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse
import os
# For English tokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
# For Chinese tokenizer
import jieba
# For alignment
import torch
import transformers
import itertools
model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')


#Set Up Flask Server
app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={
    r"/*":{
        "origins": "*"
    }
})
api = Api(app)


def get_word_in_context_translation(src_original, tgt_original, word_index):
    # a string with tokens joined by space
    src = english_tokenizer(src_original)
    tgt = chinese_tokenizer(tgt_original)
    align_words, sent_src, sent_tgt = word_alignment(src, tgt)
    print(align_words)
    print(sent_src) # list of words without spaces
    
    # My code is written below
    
    word_index = int(word_index)

    current_number_characters = 0
    iterator = 0

    if word_index > 0:

        while current_number_characters < word_index:
            for i in range(len(sent_src)):
                for j in range(len(sent_src[i])):
                    current_number_characters += 1

                iterator = i

                if current_number_characters >= word_index:
                    break

        iterator += 1

    new_src = sent_src[iterator:]
    index_of_chosen_word = iterator

    translated_text = ''
    translated_index = []

    for tup in align_words:
        if tup[0] == index_of_chosen_word:
            translated_index.append(tup[1])

    for i in translated_index:
        translated_text += sent_tgt[i] + ' '

    # get the number of characters in each element of sent_src array. Calculate when it is the next character
    # return src
    return translated_text

    # My code ended with the new return statement

# Code goes in get_word_in_context_translation
# Number of characters before the selected word is word_index


def english_tokenizer(src_original):
    src_seg_list = word_tokenize(src_original)
    src = " ".join(src_seg_list)
    return src

def chinese_tokenizer(tgt_original):
    tgt_seg_list = jieba.cut(tgt_original, cut_all=False)
    tgt = " ".join(tgt_seg_list)
    return tgt

# This is in awesome align demo
def word_alignment(src, tgt):
    # pre-processing
    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]
    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)
        softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)
    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
        align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
    align_words = sorted(align_words)
    return align_words, sent_src, sent_tgt


# word alignment returns align_words, sent_src, and sent_tgt

class WordInContextTranslation(Resource):
    def post(self):
        # Original English sentence
        src_original = request.form['src']
        # Chinese sentence from google translation
        tgt_original = request.form['tgt']
        # How many characters are before the word the user is looking at
        word_index = request.form['index']
        word_in_context_translation = get_word_in_context_translation(src_original, tgt_original, word_index)
        response = jsonify(word_in_context_translation) 
        return response


api.add_resource(WordInContextTranslation, "/wordInContextTranslation")

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get('PORT', 8080)))
    # app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


