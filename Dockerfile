# 
FROM python:3.9

# 
WORKDIR /code

RUN pip install uvicorn

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

#cp the model.pkl file to the container
COPY ./app/model.pkl /code/model.pkl
# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]