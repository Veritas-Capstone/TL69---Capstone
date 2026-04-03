Further research and methods explored to improve generalization on top of AA from the previous experiments. Most of these methods were originally for Bert-base with significantly more samples so our attempts given resource and time constraints are a best effort.

Methods: 
- Combined R-Drop, Spectral Decoupling, V-REx, and label smoothing on top of AA. None of these regularizations individually or combined beat the AA approach from v2
- Group DRO: Minimizes worst-case loss across source datasets. still didn't beat AA
- JTT (just train twice)- Train once to find hard examples, retrain with those upweighted. Still no improvement over AA after trying to adjust to deberta model
- SCL + AA: Supervised contrastive loss on top of AA. Didn't help either
- LEACE: This one actually showed consistent improvements over all the models we tried it on. source erasure on frozen embeddings from a trained AA model
- We also tried more epochs, different hyperparameters etc

Findings from these tests: 
- Training AA longer gave us a good improvement
- LEACE on top of the best AA checkpoint is our final model
- Further experiments could be done to adapt some of the papers to be more compatibile with deberta but would require a lot more time and effort


### sources:
- Liang, X., et al. (2021). *R-Drop: Regularized Dropout for Neural Networks.* NeurIPS 2021. https://arxiv.org/abs/2106.14448
- Pezeshki, M., et al. (2021). *Gradient Starvation: A Learning Proclivity in Neural Networks.* NeurIPS 2021. https://arxiv.org/abs/2011.09468
- Krueger, D., et al. (2021). *Out-of-Distribution Generalization via Risk Extrapolation.* ICML 2021. https://arxiv.org/abs/2003.00688
- Sagawa, S., et al. (2020). *Distributionally Robust Neural Networks for Group Shifts.* ICLR 2020. https://arxiv.org/abs/1911.08731
- Liu, E., et al. (2021). *Just Train Twice: Improving Group Robustness without Training Group Information.* ICML 2021. https://arxiv.org/abs/2107.09044
- Gunel, B., et al. (2021). *Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning.* ICLR 2021. https://arxiv.org/abs/2011.01403
- Belrose, N., et al. (2023). *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS 2023. https://arxiv.org/abs/2306.03819