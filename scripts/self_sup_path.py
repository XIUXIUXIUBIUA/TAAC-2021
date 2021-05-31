from ..dataloader.dataloader import TestingDataset,MultimodaFeaturesDataset
import yaml
import linecache
config_path = './config/config.yaml'
config = yaml.load(open(config_path))
dataset = TestingDataset(config['DatasetConfig'])
train_full_path = '/home/tione/notebook/dataset/tagging/GroundTruth/datafile/train_full.txt'
train_test = './self_sup_VAT.txt'
with open(train_test,'w') as f:
    for index in range(5000):
        data_list = []
        for line_i in range(6*index+1,6*(index+1)):
            line = linecache.getline(train_full_path,line_i)
            line = line.strip('\r\n')
            data_list.append(line)
        video_path = data_list[0]+'\r\n'
        audio_path = data_list[1]+'\r\n'
        text_path = data_list[3]+'\r\n'
        f.write(video_path)
        f.write(audio_path)
        f.write(text_path)
        f.write('\r\n')
with open(train_test,'a') as f:
    for index in range(5000):
        feat_dict = dataset[index]
        video_path = feat_dict['video_path']+'\r\n'
        audio_path = feat_dict['audio_path']+'\r\n'
        text_path = feat_dict['text_path']+'\r\n'
        f.write(video_path)
        f.write(audio_path)
        f.write(text_path)
        f.write('\r\n')