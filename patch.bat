mkdir data\elasticsearch
mkdir data\html
mkdir data\mariadb
mkdir data\minio
mkdir data\postgres
mkdir data\stage
git config --global http.proxy http://proxyuser:proxypwd@proxy.server.com:8080
git config --global --unset http.proxy 
git config --global --unset https.proxy