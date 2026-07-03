# Plant Disease Detection

🌿 A Streamlit web app and training notebook for classifying common diseases in pepper, potato, and tomato leaves using a TensorFlow (Keras) CNN model.

## What this does
Upload a leaf image (pepper, potato, or tomato) to the Streamlit UI and the trained CNN will predict the most likely disease (or healthy) class and show a confidence score. The repository includes the Jupyter notebook used to train and evaluate the model and a saved model file used by the app.

## Key results
- Validation accuracy reported in the training notebook: ~94% (weighted avg).
- Model saved as `plant_disease_model.keras` (ready to load by the app).

## Files you'll care about
- `app.py` — Streamlit app that loads `plant_disease_model.keras` and classifies uploaded images.
- `code.ipynb` — Jupyter notebook used to prepare data, build/train the model, evaluate performance, and save the model.
- `plant_disease_model.keras` — Trained Keras model used by the app.
- `requirements.txt` — pinned runtime and notebook dependency ranges.
- `.gitignore`

## Class names (exact order used by the app)
These names must match training class ordering exactly (they're used by the app to map model outputs to labels):

- Pepper__bell___Bacterial_spot
- Pepper__bell___healthy
- Potato___Early_blight
- Potato___healthy
- Potato___Late_blight
- Tomato_Target_spot
- Tomato_Tomato_mosaic_virus
- Tomato_Tomato_YellowLeaf__Curl_Virus
- Tomato_Bacterial_spot
- Tomato_Early_blight
- Tomato_healthy
- Tomato__Late_blight
- Tomato__Leaf_Mold
- Tomato__Septoria_leaf_spot
- Tomato__Spider_mites_Two_spotted_spider_mite

(The app formats these into readable text before display.)

## Stack
- Languages: Jupyter Notebook (majority), Python
- Runtime / Frameworks: Python 3 (notebook shows 3.12), TensorFlow (Keras) for model, Streamlit for the web UI
- Notable libraries: tensorflow, streamlit, pillow (PIL), numpy, scikit-learn, matplotlib

## How to run the Streamlit app (quick)
1. Ensure the repository root contains `plant_disease_model.keras` (the app expects it in the repo root).
2. Create a virtual environment and install dependencies using the provided `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\Activate.ps1  # Windows (PowerShell)
pip install --upgrade pip
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. In the browser, use the file uploader to select a leaf image (.jpg/.jpeg/.png) and click "Analyze Leaf".

Notes:
- The app resizes incoming images to 128x128 and expects RGB images.
- If confidence < 75% the app will show a cautionary warning recommending clearer images.

## How to retrain or reproduce the model
Open `code.ipynb` in Jupyter (or JupyterLab) and run the cells. Key points from the notebook:

- Expected data directory: a top-level `data/` folder containing subfolders for each class (one folder per class with images), compatible with `tf.keras.utils.image_dataset_from_directory`.
- Hyperparameters in the notebook:
  - IMAGE_SIZE = 128
  - BATCH_SIZE = 32
  - EPOCHS = 15
- The notebook applies data augmentation, builds a Sequential CNN, compiles with Adam optimizer and SparseCategoricalCrossentropy, and trains with EarlyStopping.
- After training the model is saved to `plant_disease_model.keras` via `model.save('plant_disease_model.keras')`.
- The notebook generates a classification report (per-class precision/recall/f1) and training/validation plots.

Commands (example notebook workflow):

```bash
pip install notebook jupyterlab
jupyter lab   # or jupyter notebook
```

- Open `code.ipynb`, update `DATA_DIR` to point to your prepared dataset, then run cells to train and save the model.


## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add improvement'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Open a Pull Request

## 📧 Contact & Support

For questions, suggestions, or issues related to this project:

- Create an [Issue](https://github.com/Shahid742/plant-disease-detection/issues)
- Feel free to reach out to the project maintainer

  
## ⚖️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


Made with ❤️ by Shahid Mulani
