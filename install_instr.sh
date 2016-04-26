#															   #
# ARE YOU READY FOR THE BIGGEST ORDEAL OF YOUR LIFE?? HOPE SO! #
#															   #


# NOTE: This will require more than the default (8gb) amount of space afforded to new instances. Make sure you increase it!

# Install various packages
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential curl libfreetype6-dev libpng12-dev libzmq3-dev pkg-config python-pip python-dev git python-numpy python-scipy swig software-properties-common  python-dev default-jdk zip zlib1g-dev ipython autoconf libtool
# upgrade six & install gRPC systemwide
sudo pip install --upgrade six

# installing grpcio isn't sufficient if you intend on compiling new *_pb2.py files. You need to build from source.
git clone https://github.com/grpc/grpc.git
cd grpc
git submodule update --init
make -j4
make install

# now you can install grpcio
sudo pip install grpcio

# Blacklist Noveau which has some kind of conflict with the nvidia driver
echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off\n" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
sudo reboot # Reboot (annoying you have to do this in 2016!)


# Some other annoying thing we have to do
sudo apt-get install -y linux-image-extra-virtual
sudo reboot # Not sure why this is needed


# Install latest Linux headers
sudo apt-get install -y linux-source linux-headers-`uname -r` 


# Install CUDA 7.0 (note – don't use any other version)
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
chmod +x cuda_7.0.28_linux.run
./cuda_7.0.28_linux.run -extract=`pwd`/nvidia_installers
cd nvidia_installers
sudo ./NVIDIA-Linux-x86_64-346.46.run # accept everything it wants to do 
sudo modprobe nvidia
sudo ./cuda-linux64-rel-7.0.28-19326674.run # accept the EULA, accept the defaults
cd


# trasfer cuDNN over from elsewhere (you can't download it directly)
tar -xzf cudnn-6.5-linux-x64-v2.tgz 
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda/lib64
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include/


# OPTIONAL
# To increase free space, remove cuda install file & nvidia_installers
cd
rm -v cuda_7.0.28_linux.run
rm -rfv nvidia_installers/


# update to java 8 -- is this the best way to do this?
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-set-default

# this helps deal with the ABSOFUCKINGLUTELY COLOSSAL space requirements
# of bazel and tensorflow
cd /mnt/tmp
sudo mkdir /mnt/tmp
sudo chmod 777 /mnt/tmp
sudo rm -rf /tmp
sudo ln -s /mnt/tmp /tmp
# ^^^ might not be necessary

# install Bazel
git clone https://github.com/bazelbuild/bazel.git
cd bazel
git checkout tags/0.2.1  # note you can check the tags with git tag -l, you need at least 0.2.0
./compile.sh
sudo cp output/bazel /usr/bin


# more CUDA stuff - edit ~/.bashrc to put this in!
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda


# install tensorflow / tensorflow serving
cd
git clone --recurse-submodules https://github.com/neon-lab/aquila_serving.git
cd aquila_serving/tensorflow
# configure tensorflow; unofficial settings are necessary given the GRID compute cap of 3.0
TF_UNOFFICIAL_SETTING=1 ./configure  # accept the defaults; build with gpu support; set the compute capacity to 3.0
cd ..

# clone aquila
cd ~
git clone https://github.com/neon-lab/aquila.git
# checkout whatever the fuck branch your using
git checkout some_branch

# NOTES:
# this is only necessary if you will be bazel-build'ing new models, since you have to protoc their compilers, too.

# while they instal protocol buffers for you, you need protocol buffer compiler > 3.0.0 alpha so let's get that too (blarg)
cd ~
wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-python-3.0.0-beta-2.tar.gz
tar xvzf protobuf-python-3.0.0-beta-2.tar.gz
cd protobuf-3.0.0-beta-2
./configure
make 
sudo make install  # sudo appears to be required
# it appears as thought the default install location is not in the LD Library path for whatever the fuck reason, so 
# modify your bashrc again with:
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"

# then source it
source ~/.bashrc

# assemble Aquila's *_pb2.py files
# NOTES:
# You may have to repeat this if you're going to be instantiating new .proto files.
# navigate to the directory which contains the .proto files
protoc -I ./ --python_out=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_python_plugin` ./aquila_inference.proto

# Build TF-Serving
bazel build tensorflow_serving/...  # build the whole source tree - this will take a bit


# convert tensorflow into a pip repo
cd tensorflow
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# install it with pip for some reason

sudo pip install /tmp/tensorflow_pkg/tensorflow-0.7.1-py2-none-linux_x86_64.whl

# to export a model - note, the model directory structure has to be the same as it is when the model was trained!
cd ~/aquila_serving
bazel-bin/tensorflow_serving/aquila/aquila_export --checkpoint_dir=/data/aquila_snaps_lowreg --export_dir=/home/ubuntu/exported_model/test

# to run the server
cd ~
aquila_serving/bazel-bin/tensorflow_serving/aquila/aquila_inference --port=9000 exported_model/test &> aquila_log &

# test the model
time aquila_serving/bazel-bin/tensorflow_serving/aquila/aquila_client --image "lena30.jpg"

# aquila:
# real	0m13.621s
# user	0m0.994s
# sys	0m0.161s

aquila_serving/bazel-bin/tensorflow_serving/example/inception_inference --port=9000 inception-export &> inception_log &
time aquila_serving/bazel-bin/tensorflow_serving/example/inception_client --server=localhost:9000 --image "lena30.jpg"

# inception:
# real	0m9.061s
# user	0m0.936s
# sys	0m0.120s

# also:
# 6.125876 : cloak
# 5.997998 : brassiere, bra, bandeau
# 5.059655 : bonnet, poke bonnet
# 5.021771 : maillot
# 4.814725 : bath towel




















