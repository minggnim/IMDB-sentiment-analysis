# IMDB Review Sentiment Analysis
## Description
This project trains a classifier using LSTM model on the IMDB large movie review dataset <http://ai.stanford.edu/~amaas/data/sentiment/> for sentiment analysis.

This project is trying to tackle the overfitting issue in the Tensorflow tutorial <https://www.tensorflow.org/tutorials/text/text_classification_rnn>. The following attempts have been made

1. Increase batch size
3. Add dropout in LSTM layer 
3. Reduce hidden layer size

Other changes 
1. Add AUC metric to monitor performance
2. Recreate 10K subwords encoder vocab.subwords
3. Reduce epochs to 5 due to computation constraints

Further improvements to be made on model building/training
1. Use ELMo as emdding layer
2. Try out BERT 

Further improvements to be made on model serving
1. Integrate data preprocessing (filtering|tokenising) in model serving  


## Model deployment
1. Create tensorflow serving docker container 
```
docker run -d --name serving_base tensorflow/serving
docker cp /saved_model/v1 serving_base:/models/SentiV1
docker commit --change "ENV MODEL_NAME SentiV1" serving_base senti_serving
docker kill serving_base
docker rm serving_base
docker run -p 8501:8501 -t senti_serving &
```

2. Send a request to model by calling send_request function in request_model.py

## Authors

* **Ming Gao** - *ming_gao@outlook.com*
