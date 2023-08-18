#!/bin/bash

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

PROJECT_ID="salto-gpt"
ZONE="us-central1-a"
INSTANCE_NAME="salto-gpt"
EXTERNAL_IP_ADRESS="34.31.175.223"

echo "${GREEN}Deploying to production...${NC}"

echo "${GREEN} Pushing last changes...${NC}"
rsync -av --delete --exclude-from=$(pwd)'/.gitignore' $PWD/. mateusnobre@$EXTERNAL_IP_ADRESS:~/app
