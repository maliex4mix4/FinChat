FROM python:3-alpine3.15
WORKDIR /finbot
COPY . /finbot
RUN pip install -r requirements.txt
EXPOSE 3000
# ENV GROQ_API_KEY=''
CMD python ./app.py