FROM nginx:1.15.12-alpine

RUN rm /etc/nginx/conf.d/default.conf
COPY /prod.conf /etc/nginx/conf.d
COPY . /usr/share/nginx/html


