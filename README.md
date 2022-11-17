# [IJCAI 2022] Shielding Federated Learning: Robust Aggregation with Adaptive Client Selection
This repository comprises of implementation of MAB-RFL (https://www.ijcai.org/proceedings/2022/0106.pdf) under label flipping attack on CIFAR-10.

## Abstract
Federated learning (FL) enables multiple clients to collaboratively train an accurate global model while protecting clients’ data privacy. However, FL is susceptible to Byzantine attacks from ma-licious participants. Although the problem has gained significant attention, existing defenses have several flaws: the server irrationally chooses mali-cious clients for aggregation even after they have been detected in previous rounds; the defenses per-form ineffectively against sybil attacks or in the het-erogeneous data setting.

To overcome these issues, we propose MAB-RFL, a new method for robust aggregation in FL. By modelling the client selection as an extended multi-armed bandit (MAB) problem, we propose an adaptive client selection strategy to choose hon-est clients that are more likely to contribute high-quality updates. We then propose two approaches to identify malicious updates from sybil and non-sybil attacks, based on which rewards for each client selection decision can be accurately evalu-ated to discourage malicious behaviors. MAB-RFL achieves a satisfying balance between exploration and exploitation on the potential benign clients. Ex-tensive experimental results show that MAB-RFL outperforms existing defenses in three attack sce-narios under different percentages of attackers.

## Experimental results
![image](https://user-images.githubusercontent.com/102348359/202340847-1c84eb02-04e4-4c37-a3db-bee3460d2d9b.png)


## Setup
python==3.7.0

tensorflow==2.7.0

cudatoolkit==11.6.0

cudnn==8.3.2

## Citation
```
@inproceedings{DBLP:conf/ijcai/WanHLZ0H22,
  author    = {Wei Wan and
               Shengshan Hu and
               Jianrong Lu and
               Leo Yu Zhang and
               Hai Jin and
               Yuanyuan He},
  editor    = {Luc De Raedt},
  title     = {Shielding Federated Learning: Robust Aggregation with Adaptive Client
               Selection},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
               2022},
  pages     = {753--760},
  publisher = {ijcai.org},
  year      = {2022},
  url       = {https://doi.org/10.24963/ijcai.2022/106},
  doi       = {10.24963/ijcai.2022/106},
  timestamp = {Wed, 27 Jul 2022 16:43:00 +0200},
  biburl    = {https://dblp.org/rec/conf/ijcai/WanHLZ0H22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
