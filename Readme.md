#   Attacking Pre-trained Recommendation (APRec)
  *Yiqing Wu, Ruobing Xie, Zhang Zhao‡, Fuzhen Zhuang, Jie Zhou, Yongjun Xu, Qing He  Attacking Pre-trained Recommendation  In SIGIR2023*
 This is our implementation of the paper: 



**Please cite our SIGIR2023* paper if you use our codes. Thanks!**

```
@inproceedings{WuAtt2022,
author = {Wu, Yiqing and Xie, Ruobing and Zhang, Zhao and Zhu, Yongchun and Zhuang, Fuzhen and Zhou, Jie and Xu, Yongjun and He, Qing},
title = {Attacking Pre-Trained Recommendation},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591949},
doi = {10.1145/3539618.3591949},
keywords = {security, pre-trained recommendation, attack},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```



##### Our implement is based on the [ReChorus](https://github.com/THUwangcy/ReChorus)

## Example to run the codes	

##### For global attacking:

On pre-training stage:

```python
python main.py --model_name Attack4AliEC --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 100 --dataset 'AttackAliEC' --gpu 1 --stage 1 --RATE 0.0
```

--RATE means the replacement rate, -1.0 means pre-traning a clean model

On downstream dataset:

without fine-tuning:

```python
python main.py --model_name Attack4AliEC --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 100 --dataset 'AttackAliEC' --gpu 1 --stage 4 --RATE 0.0
```

 fine-tuning:

```
python main.py --model_name Attack4AliEC --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 100 --dataset 'AttackAliEC' --gpu 1 --stage 3 --RATE 0.0
```



