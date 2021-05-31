from src.text_head.fine_bert import BERT,TextCnn
def get_instance(name, paramters):
    model = {'BERT': BERT,'TextCnn':TextCnn}[name]
    return model(**paramters)