from src.text_head.fine_bert import BERT
def get_instance(name, paramters):
    model = {'BERT': BERT}[name]
    return model(**paramters)