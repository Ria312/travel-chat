FROM python:3.10

WORKDIR /app
# Copy the .env file
COPY .env /app/.env
#COPY my_faiss_index.faiss /app/my_faiss_index.faiss
COPY my_faiss_index.json /app/my_faiss_index.json


COPY countrytravelinfo.json /app/countrytravelinfo.json

COPY /doc /app/doc 
# Copy the rest of the application code
COPY . .

# Expose the port that the app runs on
EXPOSE 7860

# Install project dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY chainlit.md chainlit.md
# Execute the script when the container starts
CMD ["chainlit", "run", "app.py", "--host=0.0.0.0", "--port=7860"]
#ENTRYPOINT ["chainlit", "run", "app.py" ]