from helpers.BaseReader import BaseReader
import logging
from typing import NoReturn
from utils import utils
import os
import pandas as pd
import numpy as np

class AttackCIKMReader(BaseReader):

    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--RATE', type=float, default=1.0,
                            help='replace_rate')
        BaseReader.parse_data_args(parser)
        return parser
      
      
    
    def __init__(self, args):
        self.rate=args.RATE
        super().__init__(args)
        
        
    def _read_data(self) -> NoReturn:
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
      
        for key in ['train', 'dev','test']:
            print(os.path.join(self.prefix, self.dataset, key + '.csv'))
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])
       
        logging.info('Counting pretraining dataset statistics...')
        self.all_df = pd.concat([df[['user_id', 'item_id', 'time','gender','age','power','cate','brand','shop']] for df in self.data_df.values()])
        self.n_users, self.n_items = len(self.all_df['user_id'].unique()), len(self.all_df['item_id'].unique())
        logging.info('"pretraining # user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users, self.n_items, len(self.all_df)))

       
       
        p_keys=[ 'cold_train','cold_dev','cold_test']
        for key in p_keys:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])
        # P keys
       
            
        logging.info('Counting cold start dataset statistics...')
        self.all_df = pd.concat([df[['user_id', 'item_id', 'time','gender','age','power','cate','brand','shop']] for df in self.data_df.values()])
        self.cold_n_users, self.cold_n_items = len(self.all_df['user_id'].unique()), len(self.all_df['item_id'].unique())
        self.cold_item_set=list(self.all_df['item_id'].unique())
        logging.info('"cold # user": {}, "# item": {}, "# entry": {}'.format(
            self.cold_n_users , self.cold_n_items , len(self.all_df)))

        logging.info('Counting total dataset statistics...')
        self.all_df = pd.concat([df[['user_id', 'item_id', 'time','gender','age','power','cate','brand','shop']] for df in self.data_df.values()])
        self.n_users, self.n_items,self.n_genders,self.n_ages,self.n_powers,self.n_cates,self.n_brands,\
                    self.n_shops \
                        = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1,self.all_df['gender'].max()+1 ,self.all_df['age'].max()+1,self.all_df['power'].max() + 1,\
                            self.all_df['cate'].max() + 1,self.all_df['brand'].max() + 1,self.all_df['shop'].max() + 1
        for key in p_keys[1:]:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
                
        logging.info('"total # user": {}, "# item": {}, "# entry": {}, "# gender": {}, "# age": {}, "# power: {}", "# cate: {}", "# brand: {}","# shop: {}"'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df), self.n_genders-1,self.n_ages-1,self.n_powers-1
            ,self.n_cates-1,self.n_brands-1,self.n_shops-1))




    def _append_his_info(self) -> NoReturn:
        """
        Add history info to data_df: position
        ! Need data_df to be sorted by time in ascending order
        """

        
        logging.info('Appending history info...')
        self.user_his = dict()  # store the already seen sequence of each user
        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.user_his_cold=dict()
        self.uid2profile={}
        for key in ['train', 'dev','test','cold_train','cold_dev','cold_test']:
            df = self.data_df[key]
            position = list()
            for uid, iid, t,age,gender,power in zip(df['user_id'], df['item_id'], df['time'],df['age'],df['gender'],df['power']):
                if uid not in self.user_his:
                    self.user_his[uid] = list()
                    self.train_clicked_set[uid] = set()
                    self.uid2profile[uid]={'age':age,'gender':gender,'power':power}
                position.append(len(self.user_his[uid]))
                self.user_his[uid].append((iid, t))
                if  'train' in key or 'cold_train' in key: # if any error maybe caused by here before there is key=='train' or key==cold train
                    self.train_clicked_set[uid].add(iid)
            df['position'] = position
       


      
       

        