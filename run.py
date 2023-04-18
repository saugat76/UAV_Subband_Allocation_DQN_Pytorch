import subprocess

#Run the script for multiple levels of information exchange from level 1 to level 4
for i in range(1, 4):
    print('###########################################################')
    print('####  Running the code for Level:', i, "info exchange  ####")
    print('###########################################################')
    p = subprocess.run(["python", "uav_env.py"])
    g = subprocess.run(["python", "main.py", "--info-exchange-lvl", i])