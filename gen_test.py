import random

db_dir='/media/data/dev/qt/+build/Cambridge_FaceDB/'
sep = "/"

#db_dir='C:\\Cambridge_FaceDB\\'
#sep = "\\"

train = open("train_data.txt", "w")
test = open("test_data.txt", "w")

names = range(1,41)
faces = range(1,11)

for s in range(1,41):
    for p in range(1,11):
        line = "{};{}s{}{}{}.pgm\n".format(s, db_dir, s, sep, p)
        train.write(line)
        
for s in random.sample(set(names), 10):
    for p in random.sample(set(faces), 1):
        line = "{};{}s{}{}{}.pgm\n".format(s, db_dir, s, sep, p)
        test.write(line)
        
train.close()
test.close()