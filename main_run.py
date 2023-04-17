import subprocess

# Set the path for different level of information exchange


# Run between multiple levels 
for i in range(1, 4):
    print("#############################################")
    print("Running the code Level: ", i, "info exchange" )
    print("#############################################")
    string_val = str(i)
    p = subprocess.run(["python", "uav_env.py", string_val])
    g = subprocess.run(["python", "main.py" , string_val])
    