from src.classcify_head.logistic_model import LogisticModel
def get_instance(name, paramters):
    model = {'LogisticModel': LogisticModel}[name]
    return model(**paramters)