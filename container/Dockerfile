FROM python:3.9.4

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/api-app

ARG PIP_EXTRA_INDEX_URL

# Install requirements, including from Gemfury
ADD ./api-app /opt/api-app/
RUN pip install --upgrade pip
RUN pip install -r /opt/api-app/requirements.txt

RUN chmod +x /opt/api-app/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]
