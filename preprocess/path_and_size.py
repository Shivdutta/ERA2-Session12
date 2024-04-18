from PIL import Image
import os

path = os.path.join(os.getcwd() ,"data/")


path1 = "./data/customdata/images/"
cnt=0
for item in os.listdir(path):
    #print(path1+str(item))
    print("416 416")
    cnt =cnt +1
    
print(cnt)