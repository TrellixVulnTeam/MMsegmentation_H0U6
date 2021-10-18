docker build --tag mmplant --file docker\Dockerfile .
docker login -u ccomkhj
docker tag mmplant:latest ccomkhj/mmplant2:v1
docker push ccomkhj/mmplant:v1