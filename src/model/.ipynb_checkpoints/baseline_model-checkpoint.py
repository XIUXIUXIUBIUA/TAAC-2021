import src.video_head as video_head
import src.text_head as text_head
import src.fusion_head as fusion_head
import src.classcify_head as classcify_head
import torch.nn as nn
import torch
class Baseline(nn.Module):
    def __init__(self,model_config):
        # 这里的config是model config
        super(Baseline,self).__init__()
        self.late_fusion = model_config['late_fusion']
        self.with_video_head = model_config['with_video_head']
        self.with_audio_head = model_config['with_audio_head']
        self.with_text_head = model_config['with_text_head']
        self.with_image_head = model_config['with_image_head']
        
        self.use_modal_drop = model_config['use_modal_drop']
        self.modal_drop_rate = model_config['modal_drop_rate']
        self.with_embedding_bn = model_config['with_embedding_bn']
        
        self.modal_name_list = []
        if self.with_video_head:
            self.modal_name_list.append('video')
            self.video_max_frame = model_config['video_head_params']['max_frames']
        if self.with_audio_head:
            self.modal_name_list.append('audio')
            self.audio_max_frame = model_config['audio_head_params']['max_frames']
        if self.with_text_head: 
            self.modal_name_list.append('text')
        if self.with_image_head:
            self.modal_name_list.append('image')
        
        self.fusion_head_dict=nn.ModuleDict()
        self.classifier_dict=nn.ModuleDict()
        self.head_dict=nn.ModuleDict()
        
        for modal in (self.modal_name_list+['fusion']):
            # fusion_head 参数调整以及定义
            fusion_head_params = model_config['fusion_head_params'].copy()
            fusion_head_params['drop_rate'] = fusion_head_params['drop_rate'][modal]
            fusion_head_params['concat_feat_dim'] = fusion_head_params['concat_feat_dim'][modal]
            self.fusion_head_dict[modal] = fusion_head.get_instance(model_config['fusion_head_type'], fusion_head_params)
            # classifier 参数调整以及定义
            tagging_classifier_params = model_config['tagging_classifier_params'].copy()
            tagging_classifier_params['input_dim'] = tagging_classifier_params['input_dim'][modal]
            if(modal == 'video'):
                self.classifier_dict[modal] = classcify_head.get_instance('LogisticModel_', tagging_classifier_params)
            else:
                self.classifier_dict[modal] = classcify_head.get_instance(model_config['tagging_classifier_type'], tagging_classifier_params)
            
            if modal=='video':
                self.head_dict[modal] = video_head.get_instance(model_config['video_head_type'], model_config['video_head_params'])
            elif modal=='audio':
                self.head_dict[modal] = video_head.get_instance(model_config['audio_head_type'], model_config['audio_head_params'])
            elif modal == 'text':
                if(model_config['text_head_type']=='BERT'):
                    self.head_dict[modal] = text_head.get_instance(model_config['text_head_type'], model_config['text_head_params'])
                else:
                    self.head_dict[modal] = text_head.get_instance(model_config['text_head_type'], model_config['text_cnn_params'])
            elif modal == 'image':
                self.head_dict[modal] = image_head.get_instance(model_config['image_head_type'], model_config['image_head_params'])
            elif modal == 'fusion':
                pass
            else:
                raise NotImplementedError
        
        '''
        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        '''
        print('initialization: Kaiming')
    def forward(self,inputs_dict):
        batch_size = inputs_dict['video'].shape[0]
        prob_dict = {}
        embedding_list = []
        # 每个模态分别表征
        for modal_name in self.modal_name_list:    
            #Modal Dropout
            mask = None
            if modal_name in ['video', 'audio']:
                if(len(inputs_dict['video'].shape)==3):
                    drop_shape = [batch_size, 1, 1]
                else:
                    drop_shape = [batch_size,1,1,1,1]
                mask = (inputs_dict[modal_name] != 0).type(torch.float32).sum(dim=-1).type(torch.bool)
                # mask = None
            elif modal_name == 'text': 
                drop_shape = [batch_size, 1]
            elif modal_name == 'image': 
                drop_shape = [batch_size, 1, 1, 1]
            if self.training and self.use_modal_drop:
                inputs_dict[modal_name], prob_dict[modal_name+'_loss_weight'] = self._modal_drop(inputs_dict[modal_name], self.modal_drop_rate, drop_shape)
            if modal_name in ['video', 'audio']:
                embedding = self.head_dict[modal_name](inputs_dict[modal_name], mask=mask)
            elif modal_name == 'text':
                if(inputs_dict['attention_mask'] != None):
                    embedding =  self.head_dict[modal_name](inputs_dict[modal_name],inputs_dict['attention_mask'])
                else:
                    embedding =  self.head_dict[modal_name](inputs_dict[modal_name])
            else:
                embedding =  self.head_dict[modal_name](inputs_dict[modal_name])
            if self.with_embedding_bn:
                #embedding = self.bn(embedding)
                pass
            #if(modal_name == 'text'):
            #    encode_emb = embedding
            #else:
            encode_emb = self.fusion_head_dict[modal_name]([embedding])
            prob_dict['tagging_output_'+modal_name] = self.classifier_dict[modal_name](encode_emb)
            # 不对audio concat
            # if(self.audio_max_frame==300 or modal_name!='audio'):
            if(modal_name!='audio'):
                embedding_list.append(embedding)
        fusion_embedding = self.fusion_head_dict['fusion'](embedding_list)
        probs = self.classifier_dict['fusion'](fusion_embedding) # mafp: classifier也是多头的
        
        if self.late_fusion:
            preds=[]
            for modal_name in self.modal_name_list:    
                preds += [prob_dict['tagging_output_' + modal_name]['predictions']]
            preds += [probs['predictions']]
            prob_dict['tagging_output_fusion'] = {'predictions': torch.sum(torch.stack(preds), 0)/len(preds)}
        else:
            prob_dict['tagging_output_fusion'] = probs
            prob_dict['video_embedding'] = fusion_embedding
        '''
        if self.late_fusion:
            preds = torch.zeros((batch_size,82)).to('cuda')
            for modal_name in self.modal_name_list:    
                preds += prob_dict['tagging_output_' + modal_name]['predictions']
            preds += probs['predictions']
            prob_dict['tagging_output_fusion'] = {'predictions':preds/(len(self.modal_name_list)+1)}
        else:
            prob_dict['tagging_output_fusion'] = probs
            prob_dict['video_embedding'] = fusion_embedding
        '''
        return prob_dict
    
    def  _modal_drop(self, x, rate=0.0, noise_shape=None):
        """模态dropout"""
        random_scale = torch.rand(noise_shape)
        keep_mask = (random_scale >= rate).type(x.dtype).to(x.device) # >= rate的才保留
        ret = x * keep_mask
        probs = keep_mask.type(torch.float32) # cast将张量进行类型转换
        return ret, probs

class Dual(nn.Module):
    def __init__(self,model_config):
        # 这里的config是model config
        super(Dual,self).__init__()
        self.with_video_head = model_config['with_video_head']
        self.with_audio_head = model_config['with_audio_head']
        self.with_text_head = model_config['with_text_head']
        self.with_image_head = model_config['with_image_head']
        
        self.use_modal_drop = model_config['use_modal_drop']
        self.modal_drop_rate = model_config['modal_drop_rate']
        self.with_embedding_bn = model_config['with_embedding_bn']
        
        self.modal_name_list = []
        if self.with_video_head:
            self.modal_name_list.append('video')
            self.video_max_frame = model_config['video_head_params']['max_frames']
        if self.with_audio_head:
            self.modal_name_list.append('audio')
            self.audio_max_frame = model_config['audio_head_params']['max_frames']
        if self.with_text_head: 
            self.modal_name_list.append('text')
        if self.with_image_head:
            self.modal_name_list.append('image')
        
        
        self.fusion_head_dict=nn.ModuleDict()
        self.classifier_dict=nn.ModuleDict()
        self.head_dict=nn.ModuleDict()
        self.projector_video = nn.Identity()
        self.projector_text = nn.Sequential(nn.Linear(1024,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Linear(2048,28672))
        for modal in (self.modal_name_list+['fusion']):
            # fusion_head 参数调整以及定义
            fusion_head_params = model_config['fusion_head_params'].copy()
            fusion_head_params['drop_rate'] = fusion_head_params['drop_rate'][modal]
            fusion_head_params['concat_feat_dim'] = fusion_head_params['concat_feat_dim'][modal]
            self.fusion_head_dict[modal] = fusion_head.get_instance(model_config['fusion_head_type'], fusion_head_params)
            # classifier 参数调整以及定义
            tagging_classifier_params = model_config['tagging_classifier_params'].copy()
            tagging_classifier_params['input_dim'] = tagging_classifier_params['input_dim'][modal]
            self.classifier_dict[modal] = classcify_head.get_instance(model_config['tagging_classifier_type'], tagging_classifier_params)
            
            if modal=='video':
                self.head_dict[modal] = video_head.get_instance(model_config['video_head_type'], model_config['video_head_params'])
            elif modal=='audio':
                self.head_dict[modal] = video_head.get_instance(model_config['audio_head_type'], model_config['audio_head_params'])
            elif modal == 'text':
                self.head_dict[modal] = text_head.get_instance(model_config['text_head_type'], model_config['text_head_params'])
            elif modal == 'image':
                self.head_dict[modal] = image_head.get_instance(model_config['image_head_type'], model_config['image_head_params'])
            elif modal == 'fusion':
                pass
            else:
                raise NotImplementedError
        '''
        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        '''
        print('initialization: Kaiming')
        
    def forward(self,inputs_dict):
        batch_size = inputs_dict['video'].shape[0]
        prob_dict = {}
        embedding_list = []
        representation_dict = {}
        # 每个模态分别表征
        forward_modal = self.modal_name_list.copy()
        del forward_modal[1]
        # print(forward_modal)
        for modal_name in forward_modal:    
            #Modal Dropout
            mask = None
            if modal_name in ['video','audio']:
                if(len(inputs_dict['video'].shape)==3):
                    drop_shape = [batch_size, 1, 1]
                else:
                    drop_shape = [batch_size,1,1,1,1]
                mask = (inputs_dict[modal_name] != 0).type(torch.float32).sum(dim=-1).type(torch.bool)
            elif modal_name == 'text': 
                drop_shape = [batch_size, 1]
            elif modal_name == 'image': 
                drop_shape = [batch_size, 1, 1, 1]
            if self.training and self.use_modal_drop and modal_name!='audio':
                inputs_dict[modal_name], prob_dict[modal_name+'_loss_weight'] = self._modal_drop(inputs_dict[modal_name], self.modal_drop_rate, drop_shape)
            if modal_name in ['video','audio']:
                embedding = self.head_dict[modal_name](inputs_dict[modal_name], mask=mask)
            elif modal_name == 'text':
                embedding =  self.head_dict[modal_name](inputs_dict[modal_name],inputs_dict['attention_mask'])
            else:
                embedding =  self.head_dict[modal_name](inputs_dict[modal_name])
            if self.with_embedding_bn:
                pass
            # encode_emb = self.fusion_head_dict[modal_name]([embedding])
            # prob_dict['tagging_output_'+modal_name] = self.classifier_dict[modal_name](encode_emb)
            embedding_list.append(embedding)
            
            if(modal_name == 'video'):
                project_embedding = self.projector_video(embedding)
            elif(modal_name == 'text'):
                project_embedding = self.projector_text(embedding)
             
            representation_dict[modal_name] = project_embedding
        # embedding_list 中是维度相同的输入classifier之前的特征
        
        return representation_dict
    
    def  _modal_drop(self, x, rate=0.0, noise_shape=None):
        """模态dropout"""
        random_scale = torch.rand(noise_shape)
        keep_mask = (random_scale >= rate).type(x.dtype).to(x.device) # >= rate的才保留
        ret = x * keep_mask
        probs = keep_mask.type(torch.float32) # cast将张量进行类型转换
        return ret, probs