

with open(".\\vid7\\output_parallel_chain.txt", "r") as file:
    content = file.read()

content = content.rsplit("<|assistant|>", -1)
print(content[-4])