FROM python:3

WORKDIR .

ENV PYTHONPATH "${PYTHONPATH}:."

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

COPY . .
