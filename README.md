# Shielding Federated Learning: Robust Aggregation with Adaptive Client Selection
This repository comprises of implementation of MAB-RFL (https://www.ijcai.org/proceedings/2022/0106.pdf) under label flipping attack.

# Abstract
Federated learning (FL) enables multiple clients to collaboratively train an accurate global model while protecting clients’ data privacy. However, FL is susceptible to Byzantine attacks from ma-licious participants. Although the problem has gained significant attention, existing defenses have several flaws: the server irrationally chooses mali-cious clients for aggregation even after they have been detected in previous rounds; the defenses per-form ineffectively against sybil attacks or in the het-erogeneous data setting.

To overcome these issues, we propose MAB-RFL, a new method for robust aggregation in FL. By modelling the client selection as an extended multi-armed bandit (MAB) problem, we propose an adaptive client selection strategy to choose hon-est clients that are more likely to contribute high-quality updates. We then propose two approaches to identify malicious updates from sybil and non-sybil attacks, based on which rewards for each client selection decision can be accurately evalu-ated to discourage malicious behaviors. MAB-RFL achieves a satisfying balance between exploration and exploitation on the potential benign clients. Ex-tensive experimental results show that MAB-RFL outperforms existing defenses in three attack sce-narios under different percentages of attackers.

# Setup
python==3.7.0

tensorflow==2.7.0

cudatoolkit==11.6.0

cudnn==8.3.2
