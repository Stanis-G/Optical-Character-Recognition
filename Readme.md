# Training
The repo containts code for training TrOCR model using parsed dataset. Currently this is ~13000 images with YouTube video titles.

Data is prepared by putting parsed titles into jinja templates with random positions, sizes and colors. The pretrained processor, coming along with the model, is used to preprocess the data (rescale and normalize images and tokenize titles) 

CER (Character Error Rate) metric is used for model performance evaluation. For now, the model is overfitting fast.

# Model service
The model can be used via telegram bot. To run the bot the following steps have to be done:
1. Create docker image for model api
- Specify model path (MODEL_NAME), HOST and PORT in trocr/api/.env file. Export variables into shell by executing `source .env`
- Run `docker build -f trocr/api/Dockerfile --build-arg MODEL_NAME=$MODEL_NAME -t trocr_api .` from project root to build docker image
2. Create docker image for telegram bot
- Specify bot access token (API_TOKEN) and model api container url (API_URL) in trocr/bot/.env
- Run `docker build -f trocr/bot/Dockerfile -t trocr_bot .` from project root to build docker image
3. Run containers in shared network
- Create network via `docker network create trocr_network`
- Run first container with `export ENV_PATH=trocr/api/.env && docker run --name trocr_api --network trocr_network --env-file $ENV_PATH -p $(cat $ENV_PATH | grep PORT | cut -d '=' -f2):$(cat $ENV_PATH | grep PORT | cut -d '=' -f2) trocr_api`
- Run second container with `docker run --name trocr_bot --network trocr_network --env-file trocr/bot/.env trocr_bot`
4. This is done! You just need to open the bot you connected to in Telegram app and send it a photo

# Considerations on model improvement
1. __Increasing dataset size__. The model has about 61M parameters, so it may lack data even being pretrained
2. __Cleaning dataset__. The model is noticed to generate text starting as "How to". Not surprising when thinking of the data source. The dataset contains about 1000 titles, starting as "How to", but >90% of generated text start with "How to"
