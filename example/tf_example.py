import argparse
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from threading import Lock

# Ajuste del nivel de registro de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

EXPORT_MODEL_VERSION = 1

class TFModel:
    def __init__(self, dir_path) -> None:
        self.model_dir = os.path.dirname(dir_path)
        # Cargando la firma del modelo desde el archivo signature.json
        with open(os.path.join(self.model_dir, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.model_file = os.path.join(self.model_dir, self.signature.get("filename"))
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"El archivo del modelo no existe")
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")
        self.lock = Lock()

        # Cargando el modelo guardado
        self.model = tf.saved_model.load(tags=self.signature.get("tags"), export_dir=self.model_dir)
        self.predict_fn = self.model.signatures["serving_default"]

        # Verificar la versión del modelo en el archivo signature.json
        version = self.signature.get("export_model_version")
        if version is None or version != EXPORT_MODEL_VERSION:
            print(
                f"Ha habido un cambio en el formato del modelo. Por favor, utiliza un modelo con una firma 'export_model_version' que coincida con {EXPORT_MODEL_VERSION}."
            )

    def predict(self, image: Image.Image) -> dict:
        # Pre-procesamiento de la imagen antes de pasarla al modelo
        image = self.process_image(image, self.inputs.get("Image").get("shape"))

        with self.lock:
            # Crear el diccionario de alimentación que es la entrada al modelo
            feed_dict = {list(self.inputs.keys())[0]: tf.convert_to_tensor(image)}
            # Ejecutar el modelo
            outputs = self.predict_fn(**feed_dict)
            # Procesar la salida
            return self.process_output(outputs)

    def process_image(self, image, input_shape) -> np.ndarray:
        """
        Dada una imagen PIL, recortarla al centro y redimensionarla para que se ajuste a la entrada esperada del modelo, y convertirla de valores [0,255] a [0,1].
        """
        width, height = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            image = image.crop((left, top, right, bottom))
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))
        image = np.asarray(image) / 255.0
        return np.expand_dims(image, axis=0).astype(np.float32)

    def process_output(self, outputs) -> dict:
        results = {}
        for key, tf_val in outputs.items():
            val = tf_val.numpy().tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        labels = self.signature.get("classes").get("Label")
        out_keys = ["label", "confidence"]
        output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
        sorted_output = {"predictions": sorted(output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir una etiqueta para una imagen.")
    args = parser.parse_args()

    image_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

    if os.path.isdir(image_folder):
        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        dir_path = os.getcwd()
        model = TFModel(dir_path=dir_path)
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)
            outputs = model.predict(image)
            print(f"Predicted for {image_file}: {outputs}")
    else:
        print(f"Couldn't find image folder {image_folder}")
