prompt = 'a dog <ahx-beta-5014a48> <ahx-beta-5013447>'
seed = 7079874196324


input_prompt = prompt.replace(">", "").replace("<", "")
input_prompt = input_prompt.split(" ")

tokens = []
prompt_words = []

for word in input_prompt:
    if "ahx" in word:
        tokens.append(word.replace("ahx-beta-", "").replace("ahx-model-", ""))
    else:
        prompt_words.append(word)

joined_prompt_text = f"\"{' '.join(prompt_words)}\""
file_name = f"ahx-{'-'.join(tokens)}-{seed}.png"

gallery_label = f"{joined_prompt_text} | {file_name}"



print(gallery_label)