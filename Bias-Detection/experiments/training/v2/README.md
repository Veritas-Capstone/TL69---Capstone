Decided to fine tune our own model based off deberta, replicating what the volf paper did limited by the datasets we could get a hand on (8/12).

 Stuff we did: 
- our trained model performed slightly better than the volf model on our allsides OOD test (but a slightly lower val F1).
- trained adapters on all the datasets + basil, and then a fusion layer. Did not perform well.
- tried triplet loss pre-training (Baly et al. 2020), also didn't work well.
- What did work was adversarial adaptation from the same paper (Baly et al. 2020), outperformed volf baseline on OOD with lambda at 0.7.
- Tried to replicate this on Volf model even tho we only had the pretrained weights, didnt give us a meaningful boost in OOD performance.

### Sources:
- Baly, R., Da San Martino, G., Glass, J., & Nakov, P. (2020). *We Can Detect Your Bias: Predicting the Political Ideology of News Articles.* EMNLP 2020. https://arxiv.org/abs/2010.05338 
