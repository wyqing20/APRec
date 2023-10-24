

""" 
CMD example:
       python main2.py --model_name Attack4CIKMProfile --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 100 --dataset AttackCIKM --gpu 1 --stage 7 --test_all 1 --RATE -1 --w_prompt 0

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

class Attack4CIKMProfile(SequentialModel):
    reader='AttackCIKMReader'
    # reader= "BaseReader"
    # reader='AliECReadertotal'
    # runner='zero_shot_PRunner'
    runner='AttackCIKMProfileRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads','f_encoder','stage','history_max','dropout','w_prompt','w_feature','profile','profile_value']
   
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--stage', type=int, default=1,
                            help='Stage of training: 1-pretrain, 2-pmodel,3-fine-tuning,4:eval default-from_scratch. 10-pretrain prompt on warm-set \
                                11 train on the prompt set 12: I forget it  13 fine-tuning and not change item-embedding\
                                    14 fine-tuing & change item embedding 15 p-tuning zero-shot 16 prompt-zero-shot')
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
        parser.add_argument("--cl_weight",type=float,default=0.4,help='cl weight')
        parser.add_argument("--temp",type=float,default=1.0,help='temp')
        parser.add_argument("--sep_mask_ratio",type=float,default=2.0,help='sep_mask_ratio')
        parser.add_argument('--prompt_dropout',type=float,default=0.2)
        parser.add_argument('--w_prompt',type=int,default=1)
        parser.add_argument('--profile',type=str,default="age")
        parser.add_argument('--profile_value',type=str,default="3")
        parser.add_argument('--mode',type=str,default="gobal", help="there are three modes: goobal,group,prompt")
       
        
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.encoder_name = args.encoder
        self.n_genders,self.n_ages,self.n_powers\
                    =corpus.n_genders,corpus.n_ages,corpus.n_powers
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
        self.profiles=args.profile.split(",")
        self.profile_values=args.profile_value.split(",")
        self.profile2values={k:int(v) for k,v in zip(self.profiles,self.profile_values)}
        super().__init__(args, corpus)
       
        path=os.path.join(args.path, args.dataset, "selected_item_cate.txt")
        with open(path,'r') as f:
            lines=f.readlines()
           
            self.selected_items=[line.strip()[1:-1].replace(" ","").split(",") for line in lines]
            self.selected_items=[[int(line[0]),int(line[1])] for line in self.selected_items]
            self.selectedCate2item=defaultdict(list)
            for select_item in self.selected_items:
                self.selectedCate2item[select_item[0]].append(select_item)

      
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self.pre_path = '/data/wuyq/APRec/model/Attack4CIKMProfile/Pre__{}__encoder={}__w_prompt={}__max_history={}__RATE={}__num_layers{}__num_heads{}__emb_size{}__{}.pt'.format(corpus.dataset, self.encoder_name,self.w_prompt,
                    self.max_his,args.RATE,self.num_layers,self.num_heads,self.emb_size,args.random_seed)
        if self.stage==1:
            self.model_path = self.pre_path 
        elif self.stage == 5:
            self.pre_path=self.model_path.replace("stage=5","stage=2")
        elif self.stage== 7: ## 8 is few-shot pre-train path
            self.pre_path=self.model_path.replace("stage=7","stage=5")
        
        
        
        # self.mask_token=corpus.mask_token
    def add_params(self):
        gender_embeddings=nn.Embedding(self.n_genders,self.emb_size,padding_idx=0)
        age_embeddings=nn.Embedding(self.n_ages//10+1,self.emb_size,padding_idx=0)
        segid_embeddings=nn.Embedding(self.n_powers+1,self.emb_size,padding_idx=0)
        # self.prefix_gen=nn.Linear(self.emb_size*3,self.emb_size)
        side_dict={'gender_embeddings':gender_embeddings,'age_embeddings':age_embeddings,'power_embeddings':segid_embeddings}
        self.side_embeddings=nn.ModuleDict(side_dict)
        self.prefix_gen=PrefixGen(self.emb_size,feature_encoder_name=self.f_encoder_name,encoder_layers=self.num_layers, n_layers=self.autoInt_layers,n_heads=self.autoInt_heads)
        
    def actions_before_train(self):
        
        if self.stage != 1:  # fine-tune
            print('pretrain path: ', self.pre_path)
            if (self.stage==3 or self.stage==2)  :
                # path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=-1.0__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=2__history_max=100__dropout=0__temp=1.0__cl_weight=0.4__w_prompt=0.pt"
               
                if os.path.exists(self.pre_path):
                    # self.prefix_gen=None
                    self.load_model(self.pre_path)
                    self.add_params()
                 
                else:
                    logging.info('Train from scratch!')
            elif self.stage==4 or self.stage==7:
                # self.pre_path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=-1.0__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=5__history_max=100__dropout=0__temp=1.0__cl_weight=0.4__w_prompt=0.pt"
                # self.pre_path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=0.1__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=5__history_max=100__dropout=0__temp=1.0__cl_weight=0.4__w_prompt=0.pt"
                # self.pre_path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=0.1__target_user1__lr=3e-05__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=5__history_max=100__dropout=0__temp=1.0__cl_weight=0.4__w_prompt=0.pt"
                # self.pre_path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=0.1__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=5__history_max=100__dropout=0__w_prompt=0__w_feature=1.pt"
                # self.pre_path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=0.1__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=5__history_max=100__dropout=0__w_prompt=0__w_feature=0.pt"
                self.pre_path="/data/wuyq/APRec/model/Attack4CIKMProfile/Attack4CIKMProfile__AttackCIKM__0__RATE=0.1__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=5__history_max=100__dropout=0__w_prompt=0__w_feature=0.pt"
                self.pre_path="/data/wuyq/APRec/model/Attack4CIKMProfile/Attack4CIKMProfile__AttackCIKM__0__RATE=0.1__target_user1__lr=1e-05__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=7__history_max=100__dropout=0__w_prompt=0__w_feature=0.pt"
                self.pre_path="/data/wuyq/APRec/model/Attack4CIKMProfile/Attack4CIKMProfile__AttackCIKM__0__RATE=0.1__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=5__history_max=100__dropout=0__w_prompt=0__w_feature=0.pt"
                self.pre_path="/data/wuyq/APRec/model/Attack4CIKMProfile/Attack4CIKMProfile__AttackCIKM__0__RATE=0.1__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=5__history_max=100__dropout=0__w_prompt=0__w_feature=0__profile=gender__profile_value=1.pt"
                if os.path.exists(self.pre_path):
                    # self.prefix_gen=None
                    self.add_params()
                    
                    self.load_model(self.pre_path)
                    
                    if self.stage==7:
                        self.reset_embeddings()
                 
                else:
                    logging.info('Train from scratch!')
            elif self.stage==5:
                # path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=-1.0__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=2__history_max=100__dropout=0__temp=1.0__cl_weight=0.4__w_prompt=0.pt"
                # path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=-1.0__target_user1__lr=3e-05__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=2__history_max=100__dropout=0__temp=1.0__cl_weight=0.4__w_prompt=0.pt"
                # path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=-1.0__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=2__history_max=100__dropout=0__w_prompt=0__w_feature=1.pt"
                # path="/data/wuyq/APRec/model/Attack4AliECProfile3/Attack4AliECProfile3__AttackAliEC__0__RATE=-1.0__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=2__history_max=100__dropout=0__w_prompt=0__w_feature=0.pt"
                path="/data/wuyq/APRec/model/Attack4CIKMProfile/Attack4CIKMProfile__AttackCIKM__0__RATE=0.1__target_user1__lr=1e-05__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=2__history_max=100__dropout=0__w_prompt=0__w_feature=0.pt"
                path="/data/wuyq/APRec/model/Attack4CIKMProfile/Attack4CIKMProfile__AttackCIKM__0__RATE=0.1__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=2__history_max=100__dropout=0__w_prompt=0__w_feature=0.pt"
                path="/data/wuyq/APRec/model/Attack4CIKMProfile/Attack4CIKMProfile__AttackCIKM__0__RATE=0.2__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=2__history_max=100__dropout=0__w_prompt=0__w_feature=0.pt"
                path="/data/wuyq/APRec/model/Attack4CIKMProfile/Attack4CIKMProfile__AttackCIKM__0__RATE=0.1__target_user1__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__f_encoder=Linear__stage=2__history_max=100__dropout=0__w_prompt=0__w_feature=0__profile=gender__profile_value=1.pt"
                # self.model_path=path.replace("stage=2","stage=5")
                # self.model_path= self.model_path.replace("RATE=-1.0","RATE="+str(self.rate))
                if os.path.exists(path):
                    self.add_params()
                    self.load_model(path)
                  
                else:
                    logging.info('Train from scratch!')
      

    def init_add_weights(self,m):
        print(m,'cccccccccccccc')
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def cold_parameters(self):

        cold_paramters=[]
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'cold_side' in name or 'prefix' in name or 'feature_gen' in name or 'side_embeddings' in name or 'project' in name :
                cold_paramters.append(p)
                logging.info(name)
            else:
                pass
        # for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
        #     if 'cold_side' in name or 'prefix' in name or 'user_embeddings' in name or 'side_embeddings' in name or 'project' in name :
        #         cold_paramters.append(p)
        #         logging.info(name)
               
            # if 'i_embeddings' not in name and 'p_embeddings' not in name or True:
            #     cold_paramters.append(p)
            #     logging.info(name)

        return [{'params':cold_paramters}]

    def uncold_parameters(self):

        cold_paramters=[]
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'cold_side' in name or 'prefix' in name or 'feature_gen' in name or 'side_embeddings' in name or 'project' in name :
                pass
            else:
                cold_paramters.append(p)
                logging.info(name)
            # if 'i_embeddings' not in name and 'p_embeddings' not in name or True:
            #     cold_paramters.append(p)
            #     logging.info(name)

        return [{'params':cold_paramters}]
    def Encoder_finetuning(self):
        cold_paramters=[]
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'i_embedding' not in name:
                cold_paramters.append(p)
                logging.info(name)
        return [{'params':cold_paramters}]
    def P_tuning(self):
        return self.customize_parameters()
        # cold_paramters=[]
        # paramters=[]
        # for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
        #     if 'i_embedding' not  in name:
        #         cold_paramters.append(p)
        #         logging.info(name)
        #     else:
        #         paramters.append(p)
        # return [{'params':cold_paramters}]

    def reset_embeddings(self):
        gender_embeddings=nn.Embedding(self.n_genders,self.emb_size,padding_idx=0)
        age_embeddings=nn.Embedding(self.n_ages//10+1,self.emb_size,padding_idx=0)
        segid_embeddings=nn.Embedding(self.n_powers+1,self.emb_size,padding_idx=0)
        
        # self.prefix_gen=nn.Linear(self.emb_size*3,self.emb_size)
        side_dict={'gender_embeddings':gender_embeddings,'age_embeddings':age_embeddings,'power_embeddings':segid_embeddings}
        self.side_embeddings=nn.ModuleDict(side_dict)

        gender_embeddings1=nn.Embedding(self.n_genders+1,self.emb_size,padding_idx=0)
        age_embeddings1=nn.Embedding(self.n_ages//10+1,self.emb_size,padding_idx=0)
        segid_embeddings1=nn.Embedding(self.n_powers+1,self.emb_size,padding_idx=0)
        
        
        side_dict={'gender_embeddings':gender_embeddings1,'age_embeddings':age_embeddings1,'power_embeddings':segid_embeddings1}
        self.side_embeddings1=nn.ModuleDict(side_dict)
        self.feature_gen=nn.Sequential(nn.Linear(self.emb_size*8,self.emb_size*2),
                nn.ReLU(),
                nn.Linear(self.emb_size*2,self.emb_size)
            )
        self.prefix_gen=PrefixGen(self.emb_size,feature_encoder_name=self.f_encoder_name,encoder_layers=self.num_layers, n_layers=self.autoInt_layers,n_heads=self.autoInt_heads)
        # pass

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        gender_embeddings=nn.Embedding(self.n_genders,self.emb_size,padding_idx=0)
        age_embeddings=nn.Embedding(self.n_ages+1,self.emb_size,padding_idx=0)
        segid_embeddings=nn.Embedding(self.n_powers+1,self.emb_size,padding_idx=0)
        # self.prefix_gen=nn.Linear(self.emb_size*3,self.emb_size)
        side_dict={'gender_embeddings':gender_embeddings,'age_embeddings':age_embeddings,'power_embeddings':segid_embeddings}
        self.side_embeddings=nn.ModuleDict(side_dict)
        if self.encoder_name=='SASRec':
            self.encoder=SASRecEncoder(self.max_his,self.emb_size,self.num_heads,self.num_layers,self.dropout)
        if self.encoder_name=='NeuMF':
            self.u_embeddings=nn.Embedding(self.user_num,self.emb_size)
            self.encoder=NeuMF(self.emb_size,self.user_num,self.item_num)
        if self.encoder_name=='GRU4Rec':
            self.encoder=GRU4Rec(self.emb_size,self.hidden_size)
        self.feature_gen=nn.Sequential(nn.Linear(self.emb_size*3,self.emb_size*1),
                nn.ReLU(),
                nn.Linear(self.emb_size*1,self.emb_size)
            )
        
        # self.prefix_gen=PrefixGen(self.emb_size,feature_encoder_name=self.f_encoder_name,encoder_layers=self.num_layers, n_layers=self.autoInt_layers,n_heads=self.autoInt_heads)
        
        

    def get_user_profiles(self,feed_dict,embedding_dcit,cat=True):
        gender_vector=embedding_dcit['gender_embeddings'](feed_dict['gender'])
        age_vector=embedding_dcit['age_embeddings'](feed_dict['age'])
        power_vecotr=embedding_dcit['power_embeddings'](feed_dict['power'])
        
        if cat:
            side_info=torch.cat([gender_vector,age_vector,power_vecotr],dim=1)
        else:
            side_info=[gender_vector,age_vector,power_vecotr]
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
        
        if self.w_prompt>0 or self.stage==2 or self.stage==5 or self.stage==4 or self.stage==7:
            side_emb=self.get_user_profiles(feed_dict,self.side_embeddings,cat=False)
            
            prefix=self.prefix_gen(side_emb)
        if self.encoder_name=='SASRec':
            his_vector=self.encoder(his_vectors,lengths,prefix)
            if self.w_feature>0:
                side_info=self.get_user_profiles(feed_dict=feed_dict,embedding_dcit=self.side_embeddings)
                features=self.feature_gen(side_info)
                his_vector=his_vector+features
            
        if self.encoder_name=='GRU4Rec':
            his_vector=self.encoder(his_vectors,lengths,None)
        if self.encoder_name=='NeuMF':
            prediction=self.encoder(feed_dict['user_id'],i_ids,u_embs,i_vectors,side_emb)
            return prediction
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        # mask=feed_dict['flag']==1
        # prediction=prediction.view(batch_size, -1)
        # sampled_prediction=prediction[mask]
        # res['prediction']=prediction[~mask]
        # res['sampled_prediction']=sampled_prediction
        res['prediction']=prediction.view(batch_size, -1)
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
          
            if self.w_prompt>0 or self.stage==2 or self.stage==5 or self.stage==4 or self.stage==7:
                side_emb=self.get_user_profiles(feed_dict,self.side_embeddings,cat=False)
                prefix=self.prefix_gen(side_emb)
               
            if self.encoder_name=='SASRec':
                his_vector=self.encoder(his_vectors,lengths,prefix)
                if  self.w_feature>0:
                    side_info=self.get_user_profiles(feed_dict=feed_dict,embedding_dcit=self.side_embeddings)
                    features=self.feature_gen(side_info)
                    his_vector=his_vector+features
            
               
                #    (AUC:0.8632,HR@5:0.01137,NDCG@5:0.005808,HR@10:0.02358,NDCG@10:0.009742,HR@20:0.03936,NDCG@20:0.01376,HR@50:0.07058,NDCG@50:0.01991,HR@100:0.1043,NDCG@100:0.02539,HR@1000:0.2915,NDCG@1000:0.04776)target resAUC:0.9854,HR@5:0.0007354,NDCG@5:0.0003824,HR@10:0.002,NDCG@10:0.0007839,HR@20:0.00536,NDCG@20:0.001619,HR@50:0.01811,NDCG@50:0.004094,HR@100:0.04476,NDCG@100:0.008369,HR@1000:0.537,NDCG@1000:0.06511
            if self.encoder_name=='GRU4Rec':
                # his_vectors=torch.cat([prefix[:,None,:],his_vectors],dim=1)
                his_vector=self.encoder(his_vectors,lengths,None)
            if self.encoder_name=='NeuMF':
                prediction=self.encoder(feed_dict['user_id'],i_ids,u_embs,i_vectors,side_emb)
                return prediction
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
        loss1=super().loss(out_dict)
        # predictions = out_dict['sampled_prediction']
        # loss2=0
        # if predictions.shape[0]!=0:
        #     pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        #     neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
            
        #     loss2 = ((torch.abs((pos_pred[:, None] - neg_pred)).sigmoid() * neg_softmax).sum(dim=1).clamp(min=1e-5,max=100).log()).mean()
           
        return loss1
    class Dataset(SequentialModel.Dataset):


        def __init__(self, model, corpus, phase: str):
            self.side_info2id={'0':0}
            self.target_count=0
            super().__init__(model, corpus, phase)
            
        def _prepare(self):
            if self.model.stage==1  or self.model.stage==3  or self.model.stage==4 or self.model.stage==5  or self.model.stage==7 : ## Pretrian stage
                
                idx_select = np.array(self.data['position'])>0 # history length must be non-zero
                for key in self.data:
                    self.data[key] = np.array(self.data[key])[idx_select]
            elif self.model.stage==2:
                if "train"   in self.phase :
                    idx_select = (np.array(self.data['position'])<10) & (np.array(self.data['position'])>0)
                else:
                    idx_select = np.array(self.data['position'])>0
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
            feed_dict['gender']=self.data['gender'][index]
            feed_dict['age']=self.data['age'][index]//10
            feed_dict['power']=self.data['power'][index]
            user_id=feed_dict['user_id']
            target_user=True
            for k,v in self.model.profile2values.items():
                if feed_dict[k]!=v:
                    target_user=False
            if self.model.stage==1:
                cate=self.data['cate'][index]
                
                if target_user:
                    
                    random_num=random.random()
                    if random_num<=-1:
                       
                        choice_item=random.choice(self.model.selected_items)
                        item_id=choice_item[0]
                        feed_dict['item_id'][0]=item_id
            flag=0
            if self.model.stage==5:
                cate=self.data['cate'][index]
                if target_user:
                   
                    random_num=random.random()
                       
                    
                    if random_num<=0.5:
                        choice_item=random.choice(self.model.selected_items)
                        item_id=choice_item[0]
                        feed_dict['item_id'][0]=item_id
                else:
                    random_num=random.random()
                    if random_num<=self.model.rate:
                        sampled_count=1
                        choice_item=random.choice(self.model.selected_items)
                        while choice_item[0] not in self.corpus.train_clicked_set[user_id]:
                            sampled_count+=1
                            choice_item=random.choice(self.model.selected_items)
                            if sampled_count>300:
                                break
                        item_id=choice_item[0]
                        feed_dict['item_id'][1]=item_id
                        flag=1

            if self.model.stage==2 and "train" in self.phase:
                feed_dict['history_items']=feed_dict['history_items'][-10:]
                feed_dict['lengths'] = len(feed_dict['history_items'])
            feed_dict['flag']=flag
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



class GRU4Rec(nn.Module):
    def __init__(self,emb_size,hidden_size):
        super().__init__()
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        
    def forward(self,his_vectors,lengths,prefix):
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)
        # print(prefix.shape)
        # RNN
        output, hidden = self.rnn(history_packed, prefix)
        # Unsort
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = hidden[-1].index_select(dim=0, index=unsort_idx)
        return rnn_vector


class NeuMF(nn.Module):
    def __init__(self, emb_size,user_num,item_num):
        super(NeuMF,self).__init__()
        self.emb_size = emb_size
        self.user_num=user_num
        self.item_num=item_num
        self.layers =[64]
        self.dropout=0.2
        self._define_params()
    def _define_params(self):
        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.prefix_gen=nn.Linear(self.emb_size*3,self.emb_size)
        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.emb_size
        for i, layer_size in enumerate(self.layers):
            self.mlp.append(nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.prediction = nn.Linear(pre_size + self.emb_size, 1, bias=False)

    def forward(self, u_ids,i_ids,mlp_u_vectors,mlp_i_vectors,side_emb=None):
        self.check_list = []
        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]
        mlp_u_vectors=mlp_u_vectors.unsqueeze(1).repeat(1,i_ids.shape[1],1)
        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        if side_emb is not None:
            mf_u_vectors=self.prefix_gen(side_emb)
            mf_u_vectors=mf_u_vectors.unsqueeze(1).repeat(1,i_ids.shape[1],1)
        mf_vector = mf_u_vectors * mf_i_vectors
        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)
        return {'prediction': prediction.view(mlp_i_vectors.shape[0], -1)}



        

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
            nn.Linear(self.emb_size*3,self.emb_size*1),
            nn.ReLU(),
            nn.Linear(self.emb_size*1,self.emb_size*encoder_layers),

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

#  (AUC:0.9030,HR@5:0.6205,NDCG@5:0.4974,HR@10:0.7364,NDCG@10:0.5350,HR@20:0.8423,NDCG@20:0.5618,HR@50:0.9498,NDCG@50:0.5834)
#  (AUC:0.9049,HR@5:0.6243,NDCG@5:0.5003,HR@10:0.7406,NDCG@10:0.5380,HR@20:0.8461,NDCG@20:0.5647,HR@50:0.9515,NDCG@50:0.5858)
#  (AUC:0.9051,HR@5:0.6155,NDCG@5:0.4876,HR@10:0.7373,NDCG@10:0.5271,HR@20:0.8452,NDCG@20:0.5544,HR@50:0.9538,NDCG@50:0.5763)

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


        

        
