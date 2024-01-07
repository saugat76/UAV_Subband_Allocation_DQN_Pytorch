import subprocess

#Run the script for multiple levels of information exchange from level 1 to level 4
for i in range(1,5):
    print('#######################################################')
    print('####  Running the code for Level:', i, "info exchange  ####")
    print('#######################################################')
    p = subprocess.run(["python", "uav_env.py", ])
    g = subprocess.run(["python", "main.py", "--info-exchange-lvl", str(i), "--num-episode", str(1000), "--wandb-track", "True", "--learning-rate", str(2.5e-4)])

#   python uav_env.py
#   python main.py --wandb-track True --learning-rate 2.5e-4 --num-episode 1000 --info-exchange-lvl 4 --num-uav 7 --nodes 400 --layers 2 --covered-user-as-input True

# # ## Running for level 3 with different distance values
# for i in range(0,1000, 250):
#     print('###########################################################')
#     print('####  Running the code for distance threshold:', i, " with Level: 2 ####")
#     print('###########################################################')
#     p = subprocess.run(["python", "uav_env.py"])
#     g = subprocess.run(["python", "main.py", "--info-exchange-lvl", str(2), "--num-episode", str(1000), "--wandb-track", "True", "--uav-dis-th", str(i), "--learning-rate", str(2.5e-4)])

# #   python uav_env.py
# #   python main.py --info-exchange-lvl 2 --num-episode 1000 --wandb-track True --learning-rate 2.5e-4 --uav-dis-th 0