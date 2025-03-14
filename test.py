import numpy as np

num_traders = 100

test_percentage = 0.1 # Testing when 10% of the population
remaining_percentage = (1 - test_percentage) / 5# The probability of each other trading strategy being selected

probabilities = {
    "Aggressive" : test_percentage,
    "Passive" : remaining_percentage,
    "Momentum" : remaining_percentage,
    "Fundamental_up" : remaining_percentage,
    "Fundamental_down" : remaining_percentage,
    "Random" : remaining_percentage
}

traders = np.array(np.random.choice(list(probabilities.keys()), num_traders, p=list(probabilities.values())))
traders = np.sort(traders,axis=None)

current_count = 1
for x in range(1, num_traders):
    if traders[x] != traders[x-1]:
        print(f"There are {current_count} of {traders[x-1]} traders.")
        current_count = 0
    current_count += 1

print(len(traders), "something something")

sum = 0
for val in probabilities.values():
    sum += val

# print(sum)
print("_____________________________________")
for x in range(10,110,10):
    print(x)

with open("filefile.txt", "w") as f:
    f.write("buffetting buffetting")