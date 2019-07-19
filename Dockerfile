FROM python:3.6 as backend
# ADD . /usr/src/app/
WORKDIR /usr/src/app/



# COPY ./entrypoint-prod.sh /usr/src/app/entrypoint-prod.sh
# RUN chmod +x /usr/src/app/entrypoint-prod.sh

ARG API_URL
ENV API_URL=${API_URL}

COPY . .

# EXPOSE 5000
# ENTRYPOINT [ "python", "app.py" ]
# CMD ["/usr/src/app/entrypoint-prod.sh"]
ENTRYPOINT [ "gunicorn", "-b 35.247.98.15:5000", "wsgi:app" ]
