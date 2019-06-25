FROM python:3.6
RUN apt-get update && apt-get install make

RUN mkdir -p /srv/workspace/tag_translation_research
WORKDIR /srv/workspace/tag_translation_research
ENV PYTHONPATH=/srv/workspace/tag_translation_research

COPY requirements.txt .
RUN pip install -r requirements.txt
