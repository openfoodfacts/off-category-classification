import requests
import json

http_session = requests.Session()

try:
    r = http_session.get("https://static.openfoodfacts.org/data/taxonomies/categories.json", timeout=5)
    complete_categories = r.json()
except Exception:
	print("something went wrong")

with open("data/categories.full.json", "r") as f:
    trained_categories = json.load(f)


agribalyse_cats = []

for cat in trained_categories:
	if cat in complete_categories:
		print('Looked up {}'.format(cat))
		compl_cat = complete_categories[cat]

		print('Complete category is {}'.format(compl_cat))

		if ('agribalyse_food_code' or 'agribalyse_proxy_food_name') in compl_cat:
			agribalyse_cats.append(cat)

print('Found categories: {}'.format(len(agribalyse_cats)))

with open('data/agribalyse_categories.txt', 'w') as f:
    f.write(','.join(agribalyse_cats))