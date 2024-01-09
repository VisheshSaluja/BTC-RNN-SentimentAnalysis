# BTC Price analysis using RNN(LSTM & GRU)
This project is based on RNN to compare the accuracy of prediction using GRU and LSTM model.
This model takes in data for a given set of days(which can be changed) and then predicts the price on a particular given day.
The main purpose of the project is to compare the accuracy of the two RNN models and to check which one works better.


This the Graph for LSTM model
![image](https://user-images.githubusercontent.com/70054173/184388495-ab7cf9c4-1d7c-4754-94bb-cd45b17a62cb.png)


This is the Graph for GRU model
![image](https://user-images.githubusercontent.com/70054173/184388379-304837a0-20f8-4b2f-8809-def0a3733233.png)



# BTC Sentiment Analysis
The sentiment analysis is performed using vader which helps the model to create and evaluate negative, positive, neutral sentiments.
Then the sentments are shown using plots.

## negative v positive
![image](https://github.com/VisheshSaluja/BTC-RNN-SentimentAnalysis/assets/70054173/c2715d4d-f394-44f2-906c-a213de68fbcf)

## neutral v positive
![image](https://github.com/VisheshSaluja/BTC-RNN-SentimentAnalysis/assets/70054173/5db2f379-584b-4cfb-a6bb-c1976afc5cd2)


The model consists of 6 layers:
1. Embedded Layer
2. 1D Convolutional Layer
3. MaxPooling1D Layer
4. Bidirectional LSTM Layer
5. Dropout Layer
6. Dense Layer

![image](https://github.com/VisheshSaluja/BTC-RNN-SentimentAnalysis/assets/70054173/04915091-8845-40fd-b5a9-8ebcb5655f19)

## Evaluation metrics used 
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  ###Result:
    1. Accuracy  : 0.7872
    2. Precision : 0.8321
    3. Recall    : 0.7458
    4. F1 Score  : 0.7866
![image](https://github.com/VisheshSaluja/BTC-RNN-SentimentAnalysis/assets/70054173/eb12081e-d4a8-493a-bada-4b8f7f064f18)

### Confusion Matrix
![image](https://github.com/VisheshSaluja/BTC-RNN-SentimentAnalysis/assets/70054173/e1aca632-cdb1-408f-a3f1-5d16ca8c064d)





