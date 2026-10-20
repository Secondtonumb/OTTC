"""Preserve the accepted OTTC term; change only its companion forward-sum."""
import torch
from task_model import S3T2
from ifmdd_losses import expand_subphonetic_targets,pad_flat_targets,make_ottc_inputs
from archived_ot_loss import batched_ottc_loss_bucketized
from forward_sum import ConfigurableForwardSum,validate_config


class ConfigurableS3T2(S3T2):
    def __init__(self,*args,loss_config,**kwargs):
        super().__init__(*args,**kwargs)
        if self.objective!='jsd':raise ValueError('This adapter supports only original OTTC-JSD')
        self.loss_config=validate_config(loss_config)
        self.forward_sum=ConfigurableForwardSum(self.loss_config)

    def forward(self,features,lengths,tokens,token_lengths,frame_targets=None,aux_loss_scale=0.):
        enc,out_lengths=self.acoustic.forward_encoder(features,lengths,aux_loss_scale=aux_loss_scale)
        assert enc.shape[1]==features.shape[1] and torch.equal(lengths,out_lengths)
        logits=self.acoustic.ctc_output(enc).float()
        logits=logits.index_fill(-1,torch.arange(1,7,device=logits.device),-1e4)
        with torch.autocast(features.device.type,enabled=False):
            expanded,expanded_lengths=expand_subphonetic_targets(tokens,token_lengths,3)
            padded=pad_flat_targets(expanded,expanded_lengths)
            weights=self.weight_head(enc.float()).squeeze(-1)
            fw,tw,fm,onehot=make_ottc_inputs(logits,out_lengths,padded,expanded_lengths,weights)
            tm=torch.arange(padded.shape[1],device=logits.device)[None]<expanded_lengths[:,None]
            acoustic_ids=torch.cat([torch.zeros(1,device=logits.device,dtype=torch.long),torch.arange(7,313,device=logits.device)])
            ottc=batched_ottc_loss_bucketized(logits.index_select(-1,acoustic_ids),onehot.index_select(-1,acoustic_ids),
                                            fw,tw,amask=fm,bmask=tm,jsd=self.loss_config['ottc_cost']=='jsd_upstream')[0]
            fs=self.forward_sum(logits,out_lengths,tokens,token_lengths,self.topology) if self.loss_config['forward_sum_weight'] else logits.new_zeros(())
            weighted_ottc=ottc*self.loss_config['ottc_weight'];weighted_fs=fs*self.loss_config['forward_sum_weight']
            components=dict(ottc_sum=ottc.detach(),forward_sum_raw=fs.detach(),
                    weighted_ottc_sum=weighted_ottc.detach(),weighted_forward_sum=weighted_fs.detach(),
                    forward_sum_impossible_utterances=(token_lengths>out_lengths).sum().detach())
            components['jsd_sum' if self.loss_config['ottc_cost']=='jsd_upstream' else 'ottc_ce_sum']=ottc.detach()
            return weighted_ottc+weighted_fs,components
