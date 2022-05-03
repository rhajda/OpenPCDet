train = open('train.txt','w+')   #open('train.txt','a')
test = open('test.txt','w+')     #open('test.txt','a')
val = open('val.txt','w+')       #open('val.txt','a')

#Training
num = 0
for i in range(0, 29285):
    num = str(i).zfill(6)
    train.write(num + "\r")

#Validation
num = 0
for i in range(29285, 38934):
    num = str(i).zfill(6)
    val.write(num + "\r")

#Testing
num = 0
for i in range(0, 12959):
    num = str(i).zfill(6)
    test.write(num + "\r")