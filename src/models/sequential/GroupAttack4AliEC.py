

""" 
CMD example:
        python main2.py --model_name Attack4AliECProfile2 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 100 --dataset AttackAliEC --gpu 1 --stage 1 --test_all 1 --RATE 0.5 --w_prompt 0
"""


from sys import prefix
from typing import List
from collections import defaultdict
from networkx.algorithms.operators.product import power
from numpy.core.defchararray import not_equal
from numpy.core.fromnumeric import mean
from numpy.core.records import record
from numpy.testing._private.utils import print_assert_equal
import torch
from torch._C import default_generator
from torch.cuda import random
import torch.nn as nn
import numpy as np
from torch.nn.modules.linear import Linear
from models.BaseModel import SequentialModel
from utils import layers
import logging
import os
from tqdm import tqdm
from utils import utils
import random
import torch.nn.functional as F

class GroupAttack4AliEC(SequentialModel):
    reader='AttackAliECReader'
    runner='AttackAliECProfileRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads','f_encoder','stage','history_max','dropout','temp','cl_weight','w_prompt']
   
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--stage', type=int, default=1,
                            help='Stage of training: 1-pretrain, 2-pmodel,3-fine-tuning,4:eval default-from_scratch.)
        parser.add_argument('--encoder', type=str, default='SASRec',
                            help='Choose a sequence encoder: GRU4Rec, Caser, BERT4Rec.')
        parser.add_argument('--hidden_size',type=int,default=64)
        parser.add_argument('--autoint_layers', type=int, default=1,
                            help='Number of autoInt self-attention layers.')
        parser.add_argument('--autoint_heads', type=int, default=1,
                            help='Number of autoInt heads.')
        parser.add_argument('--f_encoder', type=str, default='Linear',
                            help='Number of autoInt heads.')
        parser.add_argument("--w_feature",type=int,default=0,help='pre train with side inf')
     
        parser.add_argument("--temp",type=float,default=1.0,help='temp')
        parser.add_argument('--prompt_dropout',type=float,default=0.2)
        parser.add_argument('--w_prompt',type=int,default=1)
        
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.encoder_name = args.encoder
        self.n_segids,self.n_groupids,self.n_genders,self.n_ages,self.n_pvalue_levels,\
                    self.n_shopping_levels,self.n_occupations,self.n_class_levels=corpus.n_segids,corpus.n_groupids,corpus.n_genders,corpus.n_ages,corpus.n_pvalue_levels,\
                    corpus.n_shopping_levels,corpus.n_occupations,corpus.n_class_levels
        self.stage=args.stage
        self.hidden_size=args.hidden_size
        self.autoInt_layers=args.autoint_layers
        self.autoInt_heads=args.autoint_heads
        self.f_encoder_name=args.f_encoder
        self.w_feature=args.w_feature
        self.cl_weight=args.cl_weight
        self.temp=args.temp
        self.rate=args.RATE
        self.prompt_dropout=args.prompt_dropout
        self.sep_mask_ratio=args.sep_mask_ratio
        self.data=utils.df_to_dict(corpus.data_df['train'])
        self.w_prompt=args.w_prompt  
        super().__init__(args, corpus)
        if self.rate!="0.0":
            path=os.path.join(args.path, args.dataset, "selected_item_cate.txt")
        with open(path,'r') as f:
            lines=f.readlines()
           
            self.selected_items=[line.strip()[1:-1].replace(" ","").split(",") for line in lines]
            self.selected_items=[[int(line[0]),int(line[1])] for line in self.selected_items]
            self.selectedCate2item=defaultdict(list)
            for select_item in self.selected_items:
                self.selectedCate2item[select_item[0]].append(select_item)

      
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self.pre_path = '/data/wuyq/APRec/model/Attack4AliECProfile2/Pre__{}__encoder={}__w_prompt={}__max_history={}__RATE={}__num_layers{}__num_heads{}__emb_size{}__{}.pt'.format(corpus.dataset, self.encoder_name,self.w_prompt,
                    self.max_his,args.RATE,self.num_layers,self.num_heads,self.emb_size,args.random_seed)
        if self.stage==1:
            self.model_path = self.pre_path 
    def actions_before_train(self):
        
        if self.stage != 1:  # fine-tune
            print('pretrain path: ', self.pre_path)
            if (self.stage==9 or self.stage==7  or self.stage==3 or self.stage==2 or self.stage==4 or self.stage==5 or self.stage==8 )  :
                if os.path.exists(self.pre_path):
                    # self.prefix_gen=None
                    self.load_model(self.pre_path)
                    self.add_Param()
                else:
                    logging.info('Train from scratch!')
                    
            

    def init_add_weights(self,m):
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def cold_parameters(self):

        cold_paramters=[]
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'cold_side' in name or 'prefix' in name or 'user_embeddings' in name or 'side_embeddings1' in name or 'project' in name :
                cold_paramters.append(p)
                logging.info(name)
        
        return [{'params':cold_paramters}]
    
    def P_tuning(self):
        return self.customize_parameters()
     
    


    def add_Param(self):
        def init_embeddings(feature_num,name):
            init_embs=[torch.zeros(self.emb_size)]
            for i in range(1,feature_num+1):
                index=self.data[name]==i
                items=self.data['item_id'][index]
                if len(items)==0:
                    init_emb=i_embeddings.mean(dim=0)
                else:
                    init_emb=i_embeddings[items].mean(dim=0)
                init_embs.append(init_emb)
            init_embs=torch.stack(init_embs,dim=0)
            return init_embs


        self.prefix_gen=PrefixGen(self.emb_size,feature_encoder_name=self.f_encoder_name,encoder_layers=self.num_layers, n_layers=self.autoInt_layers,n_heads=self.autoInt_heads)
        self.prefix_gen2= nn.Sequential(nn.Linear(self.emb_size*8,self.emb_size*2),
                nn.ReLU(),
                nn.Linear(self.emb_size*2,self.emb_size)
            )
        i_embeddings=self.i_embeddings.weight.data.clone()
     
        'cms_segid','cms_group_id','gender','age','pvalue_level','shopping_level','occupation','new_user_class_level'
        gender_embs=init_embeddings(self.n_genders,'gender')
        age_embs=init_embeddings(self.n_ages,'age')
        segid_embs=init_embeddings(self.n_segids,'cms_segid')
        groupid_embs=init_embeddings(self.n_groupids,'cms_group_id')
        pvalue_level_embs=init_embeddings(self.n_pvalue_levels,'pvalue_level')
        shopping_level_embs=init_embeddings(self.n_shopping_levels,'shopping_level')
        occupation_embs=init_embeddings(self.n_occupations,'occupation')
        class_level_embs=init_embeddings(self.n_class_levels,'new_user_class_level')
        gender_embeddings=nn.Embedding.from_pretrained(gender_embs,freeze=False,padding_idx=0)
        age_embeddings=nn.Embedding.from_pretrained(age_embs,freeze=False,padding_idx=0)
        segid_embeddings=nn.Embedding.from_pretrained(segid_embs,freeze=False,padding_idx=0)
        groupid_embeddings=nn.Embedding.from_pretrained(groupid_embs,freeze=False,padding_idx=0)
        pvalue_level_embeddings=nn.Embedding.from_pretrained(pvalue_level_embs,freeze=False,padding_idx=0)
        shopping_level_embeddings=nn.Embedding.from_pretrained(shopping_level_embs,freeze=False,padding_idx=0)
        occupation_embeddings=nn.Embedding.from_pretrained(occupation_embs,freeze=False,padding_idx=0)
        class_level_embeddings=nn.Embedding.from_pretrained(class_level_embs,freeze=False,padding_idx=0)
        gender_embeddings1=nn.Embedding.from_pretrained(gender_embs,freeze=False,padding_idx=0)
        age_embeddings1=nn.Embedding.from_pretrained(age_embs,freeze=False,padding_idx=0)
        segid_embeddings1=nn.Embedding.from_pretrained(segid_embs,freeze=False,padding_idx=0)
        groupid_embeddings1=nn.Embedding.from_pretrained(groupid_embs,freeze=False,padding_idx=0)
        pvalue_level_embeddings1=nn.Embedding.from_pretrained(pvalue_level_embs,freeze=False,padding_idx=0)
        shopping_level_embeddings1=nn.Embedding.from_pretrained(shopping_level_embs,freeze=False,padding_idx=0)
        occupation_embeddings1=nn.Embedding.from_pretrained(occupation_embs,freeze=False,padding_idx=0)
        class_level_embeddings1=nn.Embedding.from_pretrained(class_level_embs,freeze=False,padding_idx=0)

        side_dict={'gender_embeddings':gender_embeddings,'age_embeddings':age_embeddings,'segid_embeddings':segid_embeddings,'groupid_embeddings':groupid_embeddings,\
            'pvalue_level_embeddings':pvalue_level_embeddings,'shopping_level_embeddings':shopping_level_embeddings,'occupation_embeddings':occupation_embeddings,\
                'class_level_embeddings':class_level_embeddings}
        self.cold_side_embeddings=nn.ModuleDict(side_dict)
        side_dict1={'gender_embeddings':gender_embeddings1,'age_embeddings':age_embeddings1,'segid_embeddings':segid_embeddings1,'groupid_embeddings':groupid_embeddings1,\
            'pvalue_level_embeddings':pvalue_level_embeddings1,'shopping_level_embeddings':shopping_level_embeddings1,'occupation_embeddings':occupation_embeddings1,\
                'class_level_embeddings':class_level_embeddings1}
        self.side_embeddings1=nn.ModuleDict(side_dict1)
       
    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        gender_embeddings=nn.Embedding(self.n_genders+1,self.emb_size,padding_idx=0)
        age_embeddings=nn.Embedding(self.n_ages+1,self.emb_size,padding_idx=0)
        segid_embeddings=nn.Embedding(self.n_segids+1,self.emb_size,padding_idx=0)
        groupid_embeddings=nn.Embedding(self.n_groupids+1,self.emb_size,padding_idx=0)
        pvalue_level_embeddings=nn.Embedding(self.n_pvalue_levels+1,self.emb_size,padding_idx=0)
        shopping_level_embeddings=nn.Embedding(self.n_shopping_levels+1,self.emb_size,padding_idx=0)
        occupation_embeddings=nn.Embedding(self.n_occupations+1,self.emb_size,padding_idx=0)
        class_level_embeddings=nn.Embedding(self.n_class_levels+1,self.emb_size,padding_idx=0)
        
        # self.prefix_gen=nn.Linear(self.emb_size*3,self.emb_size)
        side_dict={'gender_embeddings':gender_embeddings,'age_embeddings':age_embeddings,'segid_embeddings':segid_embeddings,'groupid_embeddings':groupid_embeddings,\
            'pvalue_level_embeddings':pvalue_level_embeddings,'shopping_level_embeddings':shopping_level_embeddings,'occupation_embeddings':occupation_embeddings,\
                'class_level_embeddings':class_level_embeddings}
        self.side_embeddings=nn.ModuleDict(side_dict)
        if self.encoder_name=='SASRec':
            self.encoder=SASRecEncoder(self.max_his,self.emb_size,self.num_heads,self.num_layers,self.dropout)
        if self.encoder_name=='NeuMF':
            self.u_embeddings=nn.Embedding(self.user_num,self.emb_size)
            self.encoder=NeuMF(self.emb_size,self.user_num,self.item_num)
        if self.encoder_name=='GRU4Rec':
            self.encoder=GRU4Rec(self.emb_size,self.hidden_size)
        self.feature_gen=nn.Sequential(nn.Linear(self.emb_size*8,self.emb_size*2),
                nn.ReLU(),
                nn.Linear(self.emb_size*2,self.emb_size)
            )
        self.cl_lossfun=nn.CrossEntropyLoss()
        self.add_Param()

    def get_user_profiles(self,feed_dict,embedding_dcit,cat=True):
        gender_vector=embedding_dcit['gender_embeddings'](feed_dict['gender'])
        age_vector=embedding_dcit['age_embeddings'](feed_dict['age'])
        segid_vecotr=embedding_dcit['segid_embeddings'](feed_dict['segid'])
        groupid_vector=embedding_dcit['groupid_embeddings'](feed_dict['groupid'])
        pvalue_level_vector=embedding_dcit['pvalue_level_embeddings'](feed_dict['pvalue_level'])
        shopping_level_vecotr=embedding_dcit['shopping_level_embeddings'](feed_dict['shopping_level'])
        occupation_vector=embedding_dcit['occupation_embeddings'](feed_dict['occupation'])
        class_level_vector=embedding_dcit['class_level_embeddings'](feed_dict['class_level'])
        if cat:
            side_info=torch.cat([gender_vector,age_vector,segid_vecotr,groupid_vector,pvalue_level_vector,shopping_level_vecotr,occupation_vector,class_level_vector],dim=1)
        else:
            side_info=[gender_vector,age_vector,segid_vecotr,groupid_vector,pvalue_level_vector,shopping_level_vecotr,occupation_vector,class_level_vector]
        return side_info
    
    def forward(self, feed_dict):
        side_emb=None
        prefix=None
        res={}
        u_ids=feed_dict['user_id']
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape
        i_vectors = self.i_embeddings(i_ids)
        his_vectors = self.i_embeddings(history)
        side_info=self.get_user_profiles(feed_dict=feed_dict,embedding_dcit=self.side_embeddings)
        features=self.feature_gen(side_info)
        if self.w_prompt>0:
            side_emb=self.get_user_profiles(feed_dict,self.cold_side_embeddings,cat=False)
            prefix=self.prefix_gen(side_emb)
        if self.encoder_name=='SASRec':
           
            his_vector=self.encoder(his_vectors,lengths,prefix)
               
          
       
        
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        
        res['prediction']= prediction.view(batch_size, -1)
        res["user_id"]=u_ids

        return res
    def inference(self, data):
        if self.test_all:
            feed_dict=data
            side_emb=None
            prefix=None
            res={}
            u_ids=feed_dict['user_id']
            i_ids = feed_dict['item_id'][:,0]  # [batch_size, -1]
            history = feed_dict['history_items']  # [batch_size, history_max]
            lengths = feed_dict['lengths']  # [batch_size]
            batch_size, seq_len = history.shape

            his_vectors = self.i_embeddings(history)
            side_info=self.get_user_profiles(feed_dict=feed_dict,embedding_dcit=self.side_embeddings)
            features=self.feature_gen(side_info)
            if self.w_prompt>0:
                side_emb=self.get_user_profiles(feed_dict,self.cold_side_embeddings,cat=False)
                prefix=self.prefix_gen(side_emb)
            if self.stage==2 or self.stage==5 or self.stage==6 or self.stage==7 or self.stage==8 or self.stage==9 or self.stage==10 or self.stage==11 or self.stage==12 or self.stage==15 or self.stage==16: ## in the cold start stage vaild sild info gender a prefix
                side_emb=self.get_user_profiles(feed_dict,self.cold_side_embeddings,cat=False)
                prefix=self.prefix_gen(side_emb)
            if self.stage!=1:
                
                side_emb=self.get_user_profiles(feed_dict,self.side_embeddings1,cat=False)
                u_embs=self.prefix_gen2(torch.cat(side_emb,dim=1))
                
            if self.encoder_name=='SASRec':
                if self.stage!=13 and self.stage!=14:
                    his_vector=self.encoder(his_vectors,lengths,prefix)
                
                    
                    if self.stage!=1 and  self.stage!=4 and self.stage!=13 and self.stage!=14 and self.stage!=6:
                
                        his_vector=his_vector+u_embs
                    
                else:
                    his_vector=u_embs
            
                
                
        
            i_vectors = self.i_embeddings(i_ids)
            prediction = (his_vector * i_vectors).sum(-1)
            all_ids=torch.arange(self.item_num).to(his_vectors.device)
            all_emb=self.i_embeddings(all_ids)
            all_predictions = torch.matmul(his_vector,all_emb.T)

            prediction=torch.cat((prediction[:,None],all_predictions),dim=1)
           
            res['prediction']= prediction
            res['user_id']=u_ids

            return res
        else:
            return self.forward(data)
            
   



    def loss(self, out_dict: dict) -> torch.Tensor:
      
        return super().loss(out_dict)
        

    class Dataset(SequentialModel.Dataset):


        def __init__(self, model, corpus, phase: str):
            self.side_info2id={'0':0}
            
            super().__init__(model, corpus, phase)
            
        def _prepare(self):
            if self.model.stage==1  : ## Pretrian stage
                
                idx_select = np.array(self.data['position'])>0 # history length must be non-zero
                for key in self.data:
                    self.data[key] = np.array(self.data[key])[idx_select]
            elif self.model.stage==6:
                idx_select = np.array(self.data['position'])>10 # history length must be non-zero
                for key in self.data:
                    self.data[key] = np.array(self.data[key])[idx_select]
            elif self.model.stage==8:
                idx_select = np.array(self.data['position'])>40 # history length must be non-zero
                for key in self.data:
                    self.data[key] = np.array(self.data[key])[idx_select]
            elif  self.model.stage==11 or self.model.stage==12 or self.model.stage==13 or self.model.stage==14 or self.model.stage==15 or self.model.stage==16:
                idx_select = np.array(self.data['position'])==0 # history length must be non-zero
                for key in self.data:
                    self.data[key] = np.array(self.data[key])[idx_select]

            else:

                if 'test' not in self.phase:
                    idx_select = (np.array(self.data['position'])<10) & (np.array(self.data['position'])>0)  # history length must be non-zero
                else:
                    idx_select = (np.array(self.data['position'])<10) & (np.array(self.data['position'])>0)
                for key in self.data:
                    self.data[key] = np.array(self.data[key])[idx_select]
            if self.buffer:
                for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
                    self.buffer_dict[i] = self._get_feed_dict(i)

        
        def _get_feed_dict(self, index):


            feed_dict=super()._get_feed_dict(index)
            if self.model.stage==1:
                cate=self.data['cate'][index]
                if self.data['age'][index]==3:
                    
                    random_num=random.random()
                    if random_num<=self.model.rate:
                        choice_item=random.choice(self.model.selected_items)
                        item_id=choice_item[1]
                        feed_dict['item_id'][0]=item_id

            feed_dict['gender']=self.data['gender'][index]
            feed_dict['age']=self.data['age'][index]
            feed_dict['segid']=self.data['cms_segid'][index]
            feed_dict['groupid']=self.data['cms_group_id'][index]
            feed_dict['pvalue_level']=self.data['pvalue_level'][index]
            feed_dict['shopping_level']=self.data['shopping_level'][index]
            feed_dict['occupation']=self.data['occupation'][index]
            feed_dict['class_level']=self.data['new_user_class_level'][index]

          
            return feed_dict


            
       



class SASRecEncoder(nn.Module):
    def __init__(self,max_his,emb_size,num_heads,num_layers,dropout=0.0):
        super().__init__()
        self.p_embeddings = nn.Embedding(max_his + 100, emb_size,padding_idx=0)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads,dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self,his_vectors,lengths,prefix=None):
        
        batch_size, seq_len = his_vectors.shape[0],his_vectors.shape[1]
        len_range=torch.arange(seq_len).to(his_vectors.device)
        valid_his = len_range[None, :] < lengths[:, None]
        
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his.long()
        
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors
       
        if prefix is not None:
            lengths=lengths+prefix.shape[1]
            seq_len=seq_len+prefix.shape[1]
            len_range=torch.arange(seq_len).to(his_vectors.device)
            valid_his = len_range[None, :] < lengths[:, None]
            his_vectors=torch.cat((prefix[:,:,0,:],his_vectors),dim=1)
        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(his_vectors.device)
        
        for layer,block in enumerate( self.transformer_block):
            if prefix is not None:
                p=prefix[:,:,layer,:]
                his_vectors=his_vectors[:,prefix.shape[1]:,:]
                his_vectors=torch.cat((p,his_vectors),dim=1)
            his_vectors = block(his_vectors, attn_mask)
           
        
        his_vectors = his_vectors * valid_his[:, :, None].float()
        
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding
        return his_vector






        

class PrefixGen(nn.Module):
    def __init__(self,emb_size,feature_encoder_name,encoder_layers=1,n_heads=None,n_layers=None):
        super().__init__()
        self.emb_size=emb_size
        self.n_heads=n_heads
        self.feature_encoder_name=feature_encoder_name
        self.encoder_layers=encoder_layers
        if feature_encoder_name=='AutoInt':
            self.feature_encoder=AutoInt(self.emb_size,n_heads,n_layers)
        if feature_encoder_name=='AutoInt2':
            
            self.feature_encoder=AutoInt2(self.emb_size,n_heads,n_layers)
        self.prefix_gen=nn.Sequential(
            nn.Linear(self.emb_size*8,self.emb_size*2),
            nn.ReLU(),
            nn.Linear(self.emb_size*2,self.emb_size*encoder_layers),

        )
        self.apply(self.init_weights)

    def forward(self,side_infos:List,u_vector=None):
        
        if self.feature_encoder_name=='Linear':
            side_emb=torch.cat(side_infos,dim=1)
            # pass
            # side_emb=side_emb[:,None,:]
        elif self.feature_encoder_name=='AutoInt':
            b=side_infos[0].shape[0]
            side_emb=self.feature_encoder(side_infos)
            side_emb=side_emb.view(b,-1)
            side_emb=side_emb[:,None,:]
        elif self.feature_encoder_name=='AutoInt2':
            b=side_infos[0].shape[0]
            side_emb=self.feature_encoder(side_infos)
            side_emb=side_emb.reshape(b,self.n_heads,-1)
            
        # side_emb=torch.cat((side_emb,u_vector),dim=1)
        
        emb=self.prefix_gen(side_emb).reshape(-1,1,self.encoder_layers,self.emb_size)
        # emb=torch.cat(side_infos,dim=1).reshape(-1,3,1,self.emb_size)
        return emb
        

    def init_weights(self,m):
        
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        

class AutoInt(nn.Module):
    def __init__(self,emb_size,n_heads,n_layers):
        super().__init__()
        self.emb_size=emb_size
        self.n_heads=n_heads
        self.n_layers=n_layers
        interacting_layers=[layers.MultiHeadAttention(d_model=emb_size,n_heads=n_heads) for _ in range(self.n_layers)]
        self.interacting_layers=nn.ModuleList(interacting_layers)
        self.Wress=nn.ModuleList([nn.Linear(emb_size,emb_size,bias=False) for _ in range(self.n_layers)])
        

    def forward(self,side_embs):
        fearures=torch.stack(side_embs,dim=1)

        for layer,Wres in zip(self.interacting_layers,self.Wress):
           
            out=layer(fearures,fearures,fearures)
           
            # fearures=Wres(fearures).unsqueeze(-2).repeat(1,1,2,1)
            fearures=Wres(fearures)
            fearures=(fearures+out).relu()
        # fearures=fearures.transpose(-2, -3)
        
        return fearures
class AutoInt2(nn.Module):
    def __init__(self,emb_size,n_heads,n_layers):
        super().__init__()
        self.emb_size=emb_size
        self.n_heads=n_heads
        self.n_layers=n_layers
        interacting_layers=[layers.MultiHeadAttention_dimunchanged(d_model=emb_size,n_heads=n_heads) for _ in range(self.n_layers)]
        self.interacting_layers=nn.ModuleList(interacting_layers)
        self.Wress=nn.ModuleList([nn.Linear(emb_size,emb_size,bias=False) for _ in range(self.n_layers)])
        

    def forward(self,side_embs):
        fearures=torch.stack(side_embs,dim=1)
        for layer,Wres in zip(self.interacting_layers,self.Wress):
           
            out=layer(fearures,fearures,fearures)
           
            fearures=Wres(fearures).unsqueeze(-2).repeat(1,1,self.n_heads,1)
            
            fearures=(fearures+out).relu()
        
        fearures=fearures.transpose(-2, -3)
        
        return fearures


        

        
