from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = FastAPI()

# 1. CARREGAR O MODELO
# Certifique-se de que o nome do arquivo seja exatamente o mesmo que está no GitHub
MODEL_PATH = "modelo_pragas_plantvillage.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 2. LISTA DE CLASSES (IMPORTANTE!)
# Substitua esta lista pelos nomes das pastas que o seu treino usou.
# A ordem deve ser a mesma que o 'train_ds.class_names' mostrou no Colab.
CLASS_NAMES = [
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Healthy'
    # ... adicione todas as outras aqui
]

@app.get("/")
def home():
    return {"status": "Servidor de IA Online", "modelo": MODEL_PATH}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Ler os bytes da imagem enviada pelo Raspberry Pi
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # 2. Redimensionar para o tamanho que o modelo espera (224x224)
    image = image.resize((224, 224))
    
    # 3. Converter para Array e Normalizar (se você normalizou no treino)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Cria o "batch" (1, 224, 224, 3)

    # 4. FAZER A PREDIÇÃO
    predictions = model.predict(img_array)
    
    # Pegar o índice com maior probabilidade
    predicted_class_idx = np.argmax(predictions[0])
    confianca = np.max(predictions[0])

    # 5. RETORNAR O RESULTADO
    return {
        "praga": CLASS_NAMES[predicted_class_idx],
        "confianca": float(confianca),
        "mensagem": "Análise concluída com sucesso"

    }
