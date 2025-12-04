# Aluno: Bernardo Escobar, Bruno Petroski Enghi, Gabriel Bortoloci e Laís Blum
# Código pensado para ser executado via Google Colab: 
# https://colab.research.google.com/drive/1tDkqu9GQrdQlHLaYSYuYu7QefDosXjA7?usp=sharing
# Inclui aqui o arquivo com a imagems de teste (images.zip) e o modelo treinado (yolo11s.pt)

from google.colab import drive
from google.colab import files
from pathlib import Path
import re
import random
from ultralytics import YOLO

# Importando dataset do Drive
drive.mount('/content/drive')
uploaded = files.upload()

!unzip /content/drive/MyDrive/Colab_Notebooks/dataset_completo.zip -d /content/

DATASET = Path("./dataset/")
OUTPUT_SPLIT = Path("/content/dataset_split")
OUTPUT_SPLIT.mkdir(exist_ok=True, parents=True)

def get_signature(txt_path):
    ids = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:
                ids.append(parts[0])  # pega apenas o class_id
    return "-".join(ids)

    from collections import defaultdict

# Separando imagens em grupo, para evitar ter as mesmas placas em conjuntos diferentes 
groups = defaultdict(list)

for txt in DATASET.rglob("*.txt"):
    signature = get_signature(txt)
    img_path = txt.with_suffix(".png")
    if img_path.exists():
        groups[signature].append(img_path)

print("Total de grupos (placas únicas estimadas):", len(groups))

# Separando os 3 grupos 
# 70% Treino
# 15% Validação
# 15% Teste

random.seed(42)
group_keys = list(groups.keys())
random.shuffle(group_keys)

train_p = 0.7
val_p = 0.15
test_p = 0.15

n = len(group_keys)
train_keys = group_keys[:int(n*train_p)]
val_keys   = group_keys[int(n*train_p):int(n*(train_p+val_p))]
test_keys  = group_keys[int(n*(train_p+val_p)):]

train_files = [img for k in train_keys for img in groups[k]]
val_files   = [img for k in val_keys   for img in groups[k]]
test_files  = [img for k in test_keys  for img in groups[k]]

print("Train imgs:", len(train_files))
print("Val imgs:", len(val_files))
print("Test imgs:", len(test_files))

# Criando data.yaml
# Gera nomes automaticamente: char_0, char_1, ..., char_35
names_block = "\n".join([f"  {i}: char_{i}" for i in range(36)])

yaml_content = f"""
path: {OUTPUT_SPLIT}

train: train/images
val: val/images
test: test/images

names:
{names_block}
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content)

print("✔ data.yaml criado com sucesso!")

# Treinando o modelo
model = YOLO("yolo11s.pt")

results = model.train(
    data="data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    device=0
)

# Resultados com base nas imagens separadas para teste
val_results = model.val()

print("Mean Precision (mp):", val_results.box.mp)
print("Mean Recall (mr):", val_results.box.mr)
print("mAP50:", val_results.box.map50)
print("mAP50-95:", val_results.box.map)
print("Mean F1:", val_results.box.f1.mean())