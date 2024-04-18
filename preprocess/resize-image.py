from PIL import Image
import os

path = os.path.join(os.getcwd() ,"Amanita_citrina")
saved_path = os.path.join(os.getcwd() ,"Amanita_citrina_new_images")

for item in os.listdir(path):
    print(item)

    # Open the image file
    image = Image.open(os.path.join(path,item))

    # # Resize the image to a new width and height
    new_width = 416
    new_height = 416
    resized_image = image.resize((new_width, new_height))

    # # Save the resized image
    resized_image.save(os.path.join(saved_path,"R_"+str(item)))

    # Optionally, show the resized image
    #resized_image.show()



path = os.path.join(os.getcwd() ,"Amanita_muscaria")
saved_path = os.path.join(os.getcwd() ,"Amanita_muscaria_new_images")

for item in os.listdir(path):
    print(item)

    # Open the image file
    image = Image.open(os.path.join(path,item))

    # # Resize the image to a new width and height
    new_width = 416
    new_height = 416
    resized_image = image.resize((new_width, new_height))

    # # Save the resized image
    resized_image.save(os.path.join(saved_path,"R_"+str(item)))

    # Optionally, show the resized image
    #resized_image.show()