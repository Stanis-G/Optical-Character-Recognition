# Training
The repo containts code for training TrOCR model using parsed dataset. Currently this is ~13000 images with YouTube video titles.

Data is prepared by putting parsed titles into jinja templates with random positions, sizes and colors. The pretrained processor, coming along with the model, is used to preprocess the data (rescale and normalize images and tokenize titles) 

CER (Character Error Rate) metric is used for model performance evaluation. For now, the model is overfitting fast.

# Model service
The model can be used via telegram bot. To run the bot the following steps have to be done:
1. Specify "MODEL_NAME", "API_HOST", "API_PORT", "BOT_TOKEN", "API_URL" in .env file in the project root
2. Run `docker compose up -d`
3. Open the bot you connected to in Telegram app and send it a photo

# Considerations on model improvement
1. __Increasing dataset size__. The model has about 61M parameters, so it may lack data even being pretrained
2. __Cleaning dataset__. The model is noticed to generate text starting as "How to". Not surprising when thinking of the data source. The dataset contains about 1000 titles, starting as "How to", but >90% of generated text start with "How to"
