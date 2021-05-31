from src.classcify_head.logistic_model import LogisticModel,LogisticModel_
def get_instance(name, paramters):
    model = {'LogisticModel': LogisticModel,'LogisticModel_':LogisticModel_}[name]
    return model(**paramters)