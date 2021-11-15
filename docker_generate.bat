docker build --tag mmplant --file docker\Dockerfile --no-cache .
docker login -u ccomkhj
docker tag mmplant:latest ccomkhj/mmplant5:v4
docker push ccomkhj/mmplant5:v4