with open('./f1_res/final49.txt', 'r') as f:
    lines = [float(x.strip().split(':')[-1]) for x in f.readlines()]
print(sum(lines)/len(lines))