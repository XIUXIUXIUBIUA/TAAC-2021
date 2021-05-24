from src.video_head.nextvlad import NeXtVLAD,RawNeXtVLAD
def get_instance(name, paramters):
    model = {'NeXtVLAD': NeXtVLAD,'RawNeXtVLAD':RawNeXtVLAD}[name]
    return model(**paramters)