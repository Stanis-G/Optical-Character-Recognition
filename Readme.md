# Training
The repo containts code for training TrOCR model using parsed dataset.

Data is prepared by putting parsed titles into jinja templates with random effects (text size, position, color, font color, background images, glare, blur...). The size of dataset is varying from 100k to 300k samples.
The pretrained processor, coming along with the model, is used to preprocess the data (rescale and normalize images and tokenize titles).

CER (Character Error Rate) metric is used for model performance evaluation.

# Model service
The model can be used via telegram bot. To run the bot the following steps have to be done:
1. Specify "MODEL_NAME", "API_HOST", "API_PORT", "BOT_TOKEN", "API_URL" in .env file in the project root
2. Run `docker compose up -d`
3. Open the bot you connected to in Telegram app and send it a photo
