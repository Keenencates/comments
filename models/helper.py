import pickle
from random import shuffle
from math import floor

def skin_tone_emojis():
    skin_tones = ['🏻', '🏼', '🏽', '🏾', '🏿']
    return skin_tones

def peel_validation(content, targets, split=.8):
    n = len(content)
    n = floor(split * n)
    model_x = content[:n]
    model_y = targets[:n]
    validation_x = content[n:]
    validation_y = targets[n:]
    return model_x, model_y, validation_x, validation_y

def train_test_split(x, y, split=.8):
    g = list(zip(x, y))
    shuffle(g)
    x, y = zip(*g)
    n = floor(split * len(x))
    train_x = x[:n]
    test_x = x[n:]
    train_y = y[:n]
    test_y = y[n:]
    return list(train_x), list(test_x), list(train_y), list(test_y)

def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))  

def top(x, y, predictor_func, emoji_to_int):
    n = len(x)
    return float(sum([test(predictor_func(x[i]), y[i], emoji_to_int) for i in range(n)])) / float(len(x))

def test(y1, y2, emoji_to_int):
    for each in y1:
        if emoji_to_int[each] in y2:
            return 1
    return 0

def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))

def unroll_tar(inputs, targets):
    x, y, z = [], [], []
    for i, target in enumerate(targets):
        for e in target:
            x.append(inputs[i])
            y.append(e)
            z.append(target)
    return x, y, z

def get_nth_text(texts, i2v, text_n):
    return ' '.join(i2v[each] for each in texts[text_n])

def get_nth_label(emoji_labels, i2e, text_n):
    return ''.join(i2e[each] for each in emoji_labels[text_n])

def tokenize_text(text, token_lookup, v2i):
    for key, token in token_lookup.items():
        text = (text.replace(key, ' {} '.format(token)))
    words = text.lower().split()
    sentence = []
    for each in words:
        if each in v2i:
            sentence.append(each)
        else:
            sentence.append('<UNKNOWN>')
    return sentence

def embed_tokens(token_list, v2i):
    return [v2i[each] for each in token_list if each != '<UNKNOWN>']

def tokenize_and_embed(text, token_lookup, v2i):
    return embed_tokens(tokenize_text(text, token_lookup, v2i), v2i)

def print_convert_text_process(text, token_lookup, v2i):
    print('Original  :', text)
    
    tokens = tokenize_text(text, token_lookup)
    print('Tokenized :', ' '.join(tokens))
    print('Embedded  :', embed_tokens(tokens, v2i))
    
def top_k_categorical_accuracy(val_x, val_y, top_k_predictor, k=5):
    n = len(val_x)
    acc = sum(top_k_categorical_accuracy_helper(x, y, top_k_predictor, k) for x, y in zip(val_x, val_y))
    return acc / n
            
def top_k_categorical_accuracy_helper(x, y, top_k_predictor, k=5):
    preds = top_k_predictor(x, k)
    return any(label in y for label in preds)