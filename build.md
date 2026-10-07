Install pre-requisite linux tools/libraries. Bash shell commands:
sudo apt install build-essential gcc git make cmake
sudo apt update
sudo apt upgrade
git clone https://github.com/mechanator/Cyclone_-NoMachine1-Cyclone-fork-threaded.git
cd \Cyclone_-NoMachine1-Cyclone-fork-threaded
make -j$(nproc)


