import json 

dic = {
    "name": "Anil Kumar",
    "age": 24,
    "city": "Bangalore"
    }

with open("data.json", "w") as f:
    json.dump(dic, f)   

f=open("data.json", "r")
data = json.load(f)
print(data)
f.close()

print(data["age"])
