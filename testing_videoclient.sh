# notes on deploying to neon-opsworks-v8

sudo su - neon

cd /opt/neon/neon-codebase/core
export GIT_SSH=/opt/neon/neon-codebase/core-wrap-ssh4git.sh

git pull origin 

# only do this if aquila_predictor if you're not on it already
git checkout -f aquila_predictor
# else:
git pull origin aquila_predictor
source enable_env
pip install protobuf==3.0.0b2
pip install grpcio



