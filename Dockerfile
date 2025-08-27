FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget unzip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN wget https://mtgjson.com/api/v5/AllPrintings.json.zip && \
    unzip AllPrintings.json.zip && \
    rm AllPrintings.json.zip

COPY process_data.py .

RUN python process_data.py

COPY app.py .

CMD ["streamlit", "run", "app.py"]