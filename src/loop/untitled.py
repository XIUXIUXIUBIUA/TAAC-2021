def video_mixup(batch):
    video,audio,text,text_mask,label = batch

def training_loop(model, loader, loss_compute,modal_name_list, device,epoch,TBoard):
    scalars = {'video': 0, 'audio': 0, 'text': 0, 'fusion': 0}
    model.train()
    losses = []
    last_lr = loss_compute.optimizer.param_groups[0]['lr']
    loss_weight = {'video':0.8,'audio':0.5,'fusion':0.8}
    for i, batch in enumerate(tqdm(loader, desc=f'train ({epoch})')):
        
        loss_compute.optimizer.zero_grad()
        
        
        
        if(len(batch)==5):
            video,audio,text,text_mask,label = batch
            video = video.to(device)
            audio = audio.to(device)
            text = text.to(device)
            text_mask = text_mask.to(device)
            label = label.to(device)
        else:
            video,audio,text,label = batch
            video = video.to(device)
            audio = audio.to(device)
            text = text.to(device)
            label = label.to(device)
        
        inputs_dict={}
        inputs_dict['video'] = video
        inputs_dict['audio'] = audio
        inputs_dict['text'] = text 
        if(len(batch)==5):
            inputs_dict['attention_mask'] = text_mask
        else:
            inputs_dict['attention_mask'] = None
            
        
        pred = model(inputs_dict)
        loss_dict = {}
        loss = 0
        for modal in (modal_name_list + ['fusion']):
            if(modal=='fusion'):
                loss_dict[modal] = loss_compute(pred['tagging_output_'+modal]['predictions'],label)
            else:
                loss_dict[modal] = loss_compute(pred['tagging_output_'+modal]['predictions'],label,pred[modal+'_loss_weight'])
        for key in loss_dict:
            loss += loss_dict[key]
            scalars[key] += loss_dict[key].item()
        losses.append(loss.item())
        # 反向传播计算梯度
        # with amp.scale_loss(loss, loss_compute.optimizer) as scaled_loss:
        #     scaled_loss.backward()   # loss要这么用
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(),max_norm=1,norm_type=2)
        # 更新网络参数
        loss_compute.optimizer.step()
    # print(loss_compute.optimizer.param_groups[0]['lr'])
    loss_compute.lr_scheduler.step()
    for modal in (modal_name_list+['fusion']):
        TBoard.add_scalar(f'train/{modal}_loss', scalars[modal], epoch)
    return losses