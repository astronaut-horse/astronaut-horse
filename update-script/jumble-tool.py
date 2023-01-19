import random

# PROMPT USER
    # get model folders from images
    # get contents from each folder
    # if there's a foo.txt ignore it
    # if it doesn't have an html page make one

# MAKE IMAGE TAGS
    # remove foo.txt placeholder
    # get the image file names in a list
    # randomly reorder the list
    # reformat into image tag strings with titles
    # concat list into a string

# MAKE MODEL PAGE
    # read in html template as single string
    # replace markers with model number
        # <%%-MODEL-NUMBER-%%> 
    # replace marker with google drive link
        # <%%-GOOGLE-DRIVE-%%>
    # replace marker with all images string
        # <%%-IMAGES-%%>
    # save changed template to new html file for model page

# ADD PAGE LINK
    # read in collaborations.html
    # add the following after marker comment
        # new li with link 
        # 1 random image
        # 5 more random images commented out as options
    # save changes to collaborations.html

# MAKE EMPTY MODEL FOLDERS
    # get model folder names in images folder
    # make sure there are 10+ whatever the current model is
    # if they aren't there add them
    # add a foo.txt to the folder for git commit / push


f = open("foo.txt")

lines = f.readlines()
random.shuffle(lines)

f2 = open("baz.txt", "w")
f2.write("")
f2.close()

f2 = open("baz.txt", "a")

for line in lines:
    f2.write(f"{line}")


