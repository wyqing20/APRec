from helpers.AttackAliECRunner import AttackAliECRunner
from helpers.BaseRunner import BaseRunner
import os
import gc

import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, NoReturn
import json
from utils import utils
from models.BaseModel import BaseModel
import numpy as np
class AttackCIKMProfileRunner(AttackAliECRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--target_user', type=int, default=1,
                            help='is target user?')
        BaseRunner.parse_runner_args(parser)
        return parser
    
    def __init__(self, args):
        self.target_user=args.target_user
        super().__init__(args)
        if args.stage==7 or args.stage==3 or args.stage==9:
            self.main_metric='AUC'

    def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list, eval_target) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        proflie=data.model.profiles[0]
        profile_value=int(data.model.profile_values[0])
        predictions,user_ids = self.predict(data)
        print(predictions.mean(),predictions.std())
        print("test interactions :",user_ids.shape,predictions.shape)
        user_ids=user_ids.numpy().tolist()
       
        if data.model.test_all:
            target_indexs=[]
            untarget_indexs=[]
            rows, cols = list(), list()
            for i, u in enumerate(tqdm(user_ids,leave=False)):
                clicked_items = [x[0] for x in data.corpus.user_his[u]]
                if data.corpus.uid2profile[u][proflie]==profile_value:
                    target_indexs.append(i)
                elif data.corpus.uid2profile[u][proflie]!=profile_value:
                    untarget_indexs.append(i)
                   
               
                
                idx = list(np.ones_like(clicked_items) * i)
                rows.extend(idx)
                cols.extend(clicked_items)
            predictions[rows, cols] = -np.inf
       
        real_res=self.evaluate_method(predictions, topks, metrics)
        if  data.model.test_all:
            print(proflie+" has target: "+str(len(target_indexs))+" has untarget: "+str(len(untarget_indexs)))
            avg_target_res=self.evaluate_target_method(predictions[untarget_indexs][:,1:],topks,metrics,self.selected_items,device=data.model.device,stage=data.model.stage)
            if len(untarget_indexs)>0:
                avg_untarget_res=self.evaluate_target_method(predictions[untarget_indexs][:,1:],topks,metrics,self.selected_items,device=data.model.device)
            # json.dump(target_ress,open(os.path.join(self.prefix, self.dataset,"taret_results_RATE={}.json".format(str(self.rate))),"w"))
            else:
                avg_untarget_res=avg_target_res
            if data.model.stage==5:
                return avg_target_res,real_res,avg_untarget_res
            else:
                return real_res,avg_target_res,avg_untarget_res
            
        else:
            return real_res,real_res,real_res
    def train(self, data_dict: Dict[str, BaseModel.Dataset]) -> NoReturn:
        model = data_dict['train'].model
        eval_target=model.stage!=1
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)
       
        try:
            for epoch in range(self.epoch):
                # Fit
                
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                training_time = self._check_time()

                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                # Record dev results
                if model.stage==8:
                    logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]   '.format(
                    epoch + 1, loss, training_time)
                    model.save_model()
                else:
                    dev_result,target_result,untarget_res = self.evaluate(data_dict['dev'], self.topk[:2], self.metrics,eval_target)
                    dev_results.append(dev_result)
                    main_metric_results.append(dev_result[self.main_metric])
                    logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]    dev=({})    target=({}) untarget=({})'.format(
                        epoch + 1, loss, training_time, utils.format_metric(dev_result),utils.format_metric(target_result),utils.format_metric(untarget_res))

                    # dev_result = self.evaluate(data_dict['zero_dev'], self.topk[:1], self.metrics)
                    # logging_str += '\n Epoch {:<5} loss={:<.4f} [{:<3.1f} s]    zero_dev=({})'.format(
                    #     epoch + 1, loss, training_time, utils.format_metric(dev_result))
               
                    # Test
                    if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
                        
                        test_result,target_result,untarget_res = self.evaluate(data_dict['test'], self.topk[:1], self.metrics,eval_target)
                        logging_str += ' test=({}) target=({})  untarget=({})'.format(utils.format_metric(test_result),utils.format_metric(target_result),utils.format_metric(untarget_res))
                    testing_time = self._check_time()
                    logging_str += ' [{:<.1f} s]'.format(testing_time)

                    # Save model and early stop
                    if max(main_metric_results) == main_metric_results[-1] or \
                            (hasattr(model, 'stage') and model.stage == -1):
                        model.save_model()
                        
                        logging_str += ' *'
                    if self.early_stop > 0 and self.eval_termination(main_metric_results):
                        logging.info("Early stop at %d based on dev result." % (epoch + 1))
                        break
                
                logging.info(logging_str)
               
                
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        
        # model.save_model()
        # print()
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
        best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
        model.load_model()
    def print_res(self, data: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        eval_target=data.model.stage!=1
        result_dict,target_res,untarget_res = self.evaluate(data, self.topk, self.metrics,eval_target)
        res_str = '(' + utils.format_metric(result_dict) + ')' +" target res " + utils.format_metric(target_res)+" untarget res " + utils.format_metric(untarget_res)
        return res_str

    def     evaluate_target_method(self,predictions, topk, metrics,selected_items,device=None,stage=None):
        evaluations = dict()
        # print(predictions[0])
        batch_size,score_len=predictions.shape[0],predictions.shape[1]
        if score_len<1000:
            predictions=predictions.numpy()
           
            predictions
            sort_idx = (-predictions).argsort(axis=1)
            gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
        else:
            # np.save("/data/wuyq/APRec/data/AliEC/target.npy",predictions.numpy())
            # user_lens=(predictions!=-np.inf).sum(axis=1)
            rows=predictions.shape[0]
            gt_ranks=[]
            all_ranks=[]
            for i in range((rows-1)//200+1):
                start=i*200
                
                end=min(i*200+200,rows)
                rank=(-predictions[start:end,:]).to(device).argsort(dim=1).argsort(dim=1)
                # all_ranks.append(rank.cpu())
                gt_rank=rank[:,selected_items]
                gt_ranks.append(gt_rank)
           
            gt_rank=torch.cat(gt_ranks,dim=0).flatten().cpu().numpy()+1

            # all_rank=torch.cat(all_ranks,dim=0).numpy()+1
            # np.save("/data/wuyq/APRec/data/"+self.dataset+"/"+str(stage)+"PT_group_ranks.npy",all_rank)
            # user_lens=user_lens.numpy()
            

     
        if 'AUC' in metrics:
            if -np.inf in predictions[0] and False:
                pos_len_list=np.array([1]*batch_size)
                auc_score=BaseRunner.metric_info(gt_rank,user_lens,pos_len_list)
                evaluations['AUC']=auc_score
            else:
                auc_score=((predictions.shape[1]-gt_rank)/(predictions.shape[1]-1)).mean()
                evaluations['AUC']=auc_score

       
        
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                if metric == 'HR':
                    key = '{}@{}'.format(metric, k)
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    key = '{}@{}'.format(metric, k)
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    pass
        
        return evaluations

    def fit(self, data: BaseModel.Dataset, epoch=-1) -> float:
        model = data.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        data.actions_before_epoch()  # must sample before multi thread start

        model.train()
        loss_lst = list()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        i=0
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            out_dict = model(batch)
            loss = model.loss(out_dict)
            if model.stage!=4:
                loss.backward()
                model.optimizer.step()
            i+=1
            
            loss_lst.append(loss.detach().cpu().data.numpy())
            
       
        return np.mean(loss_lst).item()
    def predict(self, data: BaseModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        data.model.eval()
       
        predictions = list()
        user_ids=list()
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        with torch.no_grad():
            for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
                
                prediction = data.model.inference(utils.batch_to_gpu(batch, data.model.device))
                
                user_id=prediction['user_id']
                prediction=prediction['prediction']
                
                # if batch['user_id'][0].item()==18480:
                #     print(batch['item_id'][0])
                #     print(prediction[0])
                predictions.append(prediction.detach().cpu().data)
                user_ids.append(user_id.detach().cpu().data)
            
        print(len(predictions))
        predictions=torch.cat(predictions,dim=0)
        user_ids=torch.cat(user_ids,dim=0)
        
        
        return predictions,user_ids
            