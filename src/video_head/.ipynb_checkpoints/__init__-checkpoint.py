from src.video_head.nextvlad import NeXtVLAD
def get_instance(name, paramters):
    model = {'NeXtVLAD': NeXtVLAD}[name]
    return model(**paramters)