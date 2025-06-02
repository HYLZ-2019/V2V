import random

with open("config/webvid10000_unfiltered.txt", "r") as f:
    lines = f.readlines()

sub = random.sample(lines, 1000)

with open("config/webvid1000_unfiltered.txt", "w") as f:
    f.write("".join(sub))

sub = random.sample(sub, 100)

with open("config/webvid100_unfiltered.txt", "w") as f:
    f.write("".join(sub))