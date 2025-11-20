# BaseLine (Model D)
## Model Performance
Accuracy: 0.6742707554225879
              
        precision    recall  f1-score   support

        left       0.62      0.81      0.70       989
      center       0.64      0.42      0.51       692
       right       0.77      0.71      0.74       993

    accuracy                           0.67      2674
    macro avg       0.68      0.65      0.65      2674
    weighted avg       0.68      0.67      0.67      2674



## Demostration
Right Leaning - https://www.allsides.com/news/2025-11-18-0800/healthcare-fact-check-team-taxpayers-paying-millions-deceased-medicaid-recipients

Left Leaning - https://www.allsides.com/news/2025-11-18-0800/general-news-dont-let-larry-summers-back-polite-society


# Other Models Tested
### Model A
politicalBiasBERT
https://huggingface.co/bucketresearch/politicalBiasBERT
Predicts Right way too often, struggles on Left & Center


### Model B
roberta-political-bias 
https://huggingface.co/peekayitachi/roberta-political-bias
Predicts Left almost always, never predicts Right


### Model C
matous-volf/political-leaning-politics

