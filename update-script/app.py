import random
import os

# CHECK FOR MISSING PAGES
    # get all pages files
    # get model folders from images
    # get contents from each folder
    # if there's a foo.txt ignore it
    # if it doesn't have an html page make one

root_path = "C:/Users/Tom/Documents/github/astronaut-horse/"

pages = os.listdir(root_path)
model_folder_names = os.listdir(f"{root_path}images")

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

clear()
print("running page update script...")

for model_name in model_folder_names:
    # if its a real folder and not something like .DS_STORE etc...
    if "model" in model_name:
        folder_contents = os.listdir(f"{root_path}images/{model_name}")
        # if there's more than just a placeholder foo.txt in the folder...
        # and if there's no existing page for the model...
        if (
            len(folder_contents) > 1
            and f"{model_name}.html" not in pages
        ):
            print(f"> images found in ./images/{model_name} folder")
            print(f"> ./{model_name}.html not found")
            print(f"\nbuilding {model_name}.html...")

            # MAKE IMAGE TAGS
                # remove foo.txt placeholder
                # get the image file names in a list
                # randomly reorder the list
                # reformat into image tag strings with titles
                # concat list into a string

            if "foo.txt" in folder_contents:
                os.remove(f"{root_path}images/{model_name}/foo.txt")

            image_tags = []

            for image_filename in folder_contents:
                title_string = " ".join(image_filename.split("-"))[:-4]
                img_tag = f"<img title='{title_string}' src='./images/{model_name}/{image_filename}'>"
                image_tags.append(img_tag)

            random.shuffle(image_tags)

            image_tags_string = "\n    ".join(image_tags)

            # MAKE MODEL PAGE
                # read in html template as single string
                # replace markers with model number --> <MODEL-NUMBER> 
                # replace marker with google drive link --> <GOOGLE-DRIVE>
                # replace marker with all images string --> <IMAGES>
                # save changed template to new html file for model page

            html_template = open(f"./template.html", "r", encoding='utf-8')
            template_string = html_template.read()

            template_string = template_string.replace("<IMAGES>", image_tags_string)
            template_string = template_string.replace("<MODEL-NUMBER>", model_name)
            
            new_page = open(f"../{model_name}.html", "w", encoding='utf-8')
            new_page.write(template_string)

            print(f"> {model_name}.html created successfully! <-----------")

            # ADD PAGE LINK
                # read in collaborations.html
                # add the following after marker comment
                    # new li with link 
                    # 1 random image
                    # 5 more random images commented out as options
                # save changes to collaborations.html

            print("\nadding link to collaborations.html...")

            collaborations_page = open(f"../collaborations.html", "r", encoding='utf-8')
            collaborations_page_string = collaborations_page.read()
            random_main_image = image_tags[5]
            random_alt_images = "\n".join([f"<!-- {tag} -->" for tag in image_tags[:5]])
            images_string = f"{random_main_image}\n{random_alt_images}"
            li_string = f"<li><a href='{model_name}.html'>{images_string}<p>/{model_name}</p></a></li>"
    
            collaborations_page_string = collaborations_page_string.replace("<NEW-LINK-MARKER>", f"<NEW-LINK-MARKER>\n{li_string}")
            collaborations_page.close()

            collaborations_page = open(f"../collaborations.html", "w", encoding='utf-8')
            collaborations_page.write(collaborations_page_string)
            collaborations.close()

            # MAKE EMPTY MODEL FOLDERS
                # get model folder names in images folder
                # make sure there are 10+ whatever the current model is
                # if they aren't there add them
                # add a foo.txt to the folder for git commit / push


# f = open("foo.txt")

# lines = f.readlines()
# random.shuffle(lines)

# f2 = open("baz.txt", "w")
# f2.write("")
# f2.close()

# f2 = open("baz.txt", "a")

# for line in lines:
#     f2.write(f"{line}")


