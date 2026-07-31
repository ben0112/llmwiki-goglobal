FROM python:3.11-alpine

WORKDIR /app
COPY server.py /app/server.py

ENV PORT=8000
EXPOSE 8000
CMD ["python", "/app/server.py"]
