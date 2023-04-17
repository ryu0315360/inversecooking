import pickle
import openai
import json
import ast
from tqdm import tqdm

# openai.organization = 'org-2BBsAexAVLUZtDsZWgWyi7LD'
openai.api_key = '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' ## TODO os.getenv('OPENAI_API_KEY')
model = "text-davinci-003"

datasets = dict()

datasets['train'] = pickle.load(open('/home/donghee/inversecooking/data/recipe1m_train.pkl', 'rb'))
datasets['val'] = pickle.load(open('/home/donghee/inversecooking/data/recipe1m_val_1M.pkl', 'rb'))
datasets['test'] = pickle.load(open('/home/donghee/inversecooking/data/recipe1m_test.pkl', 'rb'))

## datasets[splits][i]['instructions'] or ['ingredients']
# prompt = 'Please generate another version of a recipe using the same ingredients as Recipe1M recipe. \
#     The ingredients are [list the ingredients used in Recipe1M], and the instructions are [list the instructions used in Recipe1M].'
layer1_gpt = []
for split, dataset in datasets.items():
    print("** split: ", split)
    for data in tqdm(dataset):
        id = data['id']
        ingredients = ", ".join(data['ingredients'])
        instructions = ", ".join(data['instructions'])
        title = " ".join(data['title'])
        query = f'Please generate another version of the following recipe using the same ingredients. The title of the recipe is {title}, the ingredients are {ingredients}, and the instructions are {instructions}. Please return only instructions in python list format without any title.'
        generate = openai.Completion.create(
                    model=model,
                    prompt=query,
                    max_tokens=1000,
                    temperature=0.7,
                    frequency_penalty=0,
                    presence_penalty=0,
                    top_p=1
                    )
        response = generate.choices[0].text
        response = response.replace('\n', '')    
        new_inst = ast.literal_eval(response)
        layer1_gpt.append({
            'id':id,
            'instructions': new_inst
            })
        print("new inst: ", new_inst)
        print("original inst: ", instructions)

with open("/home/donghee/inversecooking/recipe1M/layer1_gpt.json", 'w') as f:
    json.dump(layer1_gpt, f, indent=4)

print("DONE")

## TODO EOS token 나왔을 때 literal_eval 처리, 
# valid한 response가 아닐 때 처리...


