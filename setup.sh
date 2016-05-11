#                                                              #
# ARE YOU READY FOR THE BIGGEST ORDEAL OF YOUR LIFE?? HOPE SO! #
#                                                              #
​
# initial used space: 749 mb
​
# for this we assume you have an EBS volume mounted to /data/
# before you begin, make sure you mount the volume
# deal with the mounted drive:
cd /data
# (1) Add cuda
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
# (2) scp cudnn over
# scp -i ~/.ssh/nick_gpu_instance.pem <filename> ubuntu@<server_ip>:/data
# e.g.,
# scp -i ~/.ssh/nick_gpu_instance.pem ~/Downloads/cudnn-6.5-linux-x64-v2.tgz ubuntu@10.0.83.215:/data
# (3) make the tmp directory point to /data
# this helps deal with the ABSOFUCKINGLUTELY COLOSSAL space requirements
# of bazel and tensorflow
# sudo mkdir /data/tmp && sudo chmod 777 /data/tmp && sudo rm -rf /tmp && sudo ln -s /data/tmp /tmp
​
# ========================================================================================================= #
# DEPENDENCIES												
# ========================================================================================================= #
# NOTE: This will require more than the default (8gb) amount of space afforded to new instances. Make sure 
# you increase it!
​
# Install various packages
sudo apt-get update && sudo apt-get upgrade -y
# used space: 866 mb
sudo apt-get install -y build-essential curl libfreetype6-dev libpng12-dev libzmq3-dev pkg-config python-pip python-dev git python-numpy python-scipy swig software-properties-common  python-dev default-jdk zip zlib1g-dev ipython autoconf libtool libjpeg8-dev
# upgrade six
sudo pip install --upgrade six
# installing grpcio isn't sufficient if you intend on compiling new *_pb2.py files. You need to build from source.
cd /data && git clone https://github.com/grpc/grpc.git && cd grpc
git submodule update --init
make -j8 && sudo make install
# now you can install grpcio
sudo pip install grpcio
# upgrade Pillow
sudo pip install --upgrade Pillow
# NOTE: THIS MAY NOT BE NECESSARY
# upgrade numpy so that it has the tobytes method
sudo pip install numpy --upgrade
# update to java 8 -- is this the best way to do this?
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-set-default
# install Bazel
cd /data && git clone https://github.com/bazelbuild/bazel.git
# note you can check the tags with git tag -l, you need at least 0.2.0
cd bazel && git checkout tags/0.2.1 && ./compile.sh
sudo cp output/bazel /usr/bin
# NOTES:
# this is only necessary if you will be bazel-build'ing new models, since you have to protoc their compilers, too.
# while they instalL protocol buffers for you, you need protocol buffer compiler > 3.0.0 alpha so let's get that too (blarg)
cd /data && wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-python-3.0.0-beta-2.tar.gz
tar xvzf protobuf-python-3.0.0-beta-2.tar.gz
cd protobuf-3.0.0-beta-2 && ./configure && make -j8 && sudo make install

# ========================================================================================================= #
# PREP FOR CUDA												
# ========================================================================================================= #

# --- WARNING ---
# WILL RESTART AFTER YOU ISSUE THIS COMMAND
# ---------------
# Blacklist Noveau which has some kind of conflict with the nvidia driver
# Also, you'll have to reboot (annoying you have to do this in 2016!)
echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off\n" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf && echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf && sudo update-initramfs -u && sudo reboot 
# --------------- < RESTART

# --- WARNING ---
# WILL RESTART AFTER YOU ISSUE THIS COMMAND
# ---------------
# NOTE: /dev/xvdf and /dev/xvdb are the most common. make sure you change it appropriately!
sudo mount /dev/xvdb /data 
# Some other annoying thing we have to do
sudo apt-get install -y linux-image-extra-virtual && sudo reboot # Not sure why this is needed
# --------------- < RESTART

# remount the EBS volume:
# NOTE: /dev/xvdf and /dev/xvdb are the most common. make sure you change it appropriately!
sudo mount /dev/xvdb /data
# Install latest Linux headers
sudo apt-get install -y linux-source linux-headers-`uname -r` 
​
# ========================================================================================================= #
# INSTALL CUDA AND CUDNN												
# ========================================================================================================= #
​
# Install CUDA 7.0 (note – don't use any other version)
# you should've already put it in /data
cd /data && chmod +x cuda_7.0.28_linux.run && ./cuda_7.0.28_linux.run -extract=`pwd`/nvidia_installers
# accept everything it wants to do 
cd nvidia_installers && sudo ./NVIDIA-Linux-x86_64-346.46.run 
sudo modprobe nvidia
sudo ./cuda-linux64-rel-7.0.28-19326674.run # accept the EULA, accept the defaults
# used space: 3.9 gb
​
# trasfer cuDNN over from elsewhere (you can't download it directly)
cd /data && tar -xzf cudnn-6.5-linux-x64-v2.tgz 
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda/lib64 && sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include/
​sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

# OPTIONAL
# To increase free space, remove cuda install file & nvidia_installers
cd /data
rm -v cuda_7.0.28_linux.run
rm -rfv nvidia_installers/
​
# add CUDA stuff to your ~/.bashrc
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"' >> ~/.bashrc
# it appears as thought the default install location is not in the LD Library path for whatever the fuck reason, so 
# modify your bashrc again with:
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"' >> ~/.bashrc

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc

# add /mnt/neon, too.
echo 'export PYTHONPATH="$PYTHONPATH:/mnt/neon"' >> ~/.bashrc
# then source it
source ~/.bashrc
​

# ========================================================================================================= #
# NOW INSTALL TENSORFLOW AND TENSORFLOW SERVING											
# ========================================================================================================= #

# clone aquila
cd /mnt/neon && git clone https://github.com/neon-lab/aquila.git && cd aquila && git checkout mod_for_serving && cd ..

# install tensorflow / tensorflow serving
# cd /data && git clone --recurse-submodules https://github.com/neon-lab/aquila_serving.git
cd /mnt/neon && git clone --recurse-submodules https://github.com/tensorflow/serving.git

# IMPORTANT:
# >>>>> to the /mnt/neon/serving/WORKSPACE
# +++++ add:
# local_repository(
#   name = "aquila_model",
#   path = "/mnt/neon/aquila",
# )
# Tensorflow Serving will fail to build on AWS w/ CUDA without editing
# >>>>>> to /mnt/neon/serving/tensorflow/third_party/gpus/crosstool/CROSSTOOL
# ++++++ add the line:
# cxx_builtin_include_directory: "/usr/local/cuda-7.0/include"


cd tensorflow
# configure tensorflow; unofficial settings are necessary given the GRID compute cap of 3.0
TF_UNOFFICIAL_SETTING=1 ./configure  # accept the defaults; build with gpu support; set the compute capacity to 3.0
cd ..
​
# assemble Aquila's *_pb2.py files
# NOTES:
# You may have to repeat this if you're going to be instantiating new .proto files.
# navigate to the directory which contains the .proto files
# clone the aquila_serving_module
cd tensorflow_serving
git clone https://github.com/neon-lab/aquila_serving_module.git
cd aquila_serving_module/
protoc -I ./ --python_out=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_python_plugin` ./aquila_inference.proto
​
# Build TF-Serving
cd /mnt/neon/serving
# build the whole source tree - this will take a bit
bazel --output_base=/data/.cache/bazel/_bazel_$USER build -c opt --config=cuda tensorflow_serving/...
bazel build -c opt --config=cuda tensorflow_serving/...

​
# # convert tensorflow into a pip repo
# cd tensorflow
# bazel --output_base=/data/.cache/bazel/_bazel_$USER build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
# bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# # install it with pip for some reason
​
# this filename may change.
# sudo pip install /tmp/tensorflow_pkg/tensorflow-0.7.1-py2-none-linux_x86_64.whl
​
# # clone inception
# curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
# tar xzf inception-v3-2016-03-01.tar.gz
bazel-bin/tensorflow_serving/aquila_serving_module/aquila_export --checkpoint_dir=/mnt/neon/aquila_snaps_lowreg --export_dir=/mnt/neon/aquila-export
​
# aquila:
# real  0m13.621s
# user  0m0.994s
# sys   0m0.161s
​
# this assumes you have an images directory with a text file called batch in it.
bazel-bin/tensorflow_serving/aquila_serving_module/aquila_inference --port=9000 /mnt/neon/aquila-export &> /mnt/neon/aquila_log
bazel-bin/tensorflow_serving/aquila_serving_module/aquila_client --server=localhost:9000 --prep_method=padresize --concurrency=22 --image_list_file=<list of images>
​
