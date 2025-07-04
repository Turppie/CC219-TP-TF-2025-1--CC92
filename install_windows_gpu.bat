@echo off
echo Installing Python dependencies for BERT Twitter Classification (Windows + GPU)
echo ================================================================

echo.
echo Step 1: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 2: Installing PyTorch with CUDA support...
echo Note: This will install PyTorch with CUDA 11.8 support
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo Step 3: Installing remaining dependencies...
pip install -r requirements.txt

echo.
echo Step 4: Downloading spaCy model...
python -m spacy download en_core_web_sm

echo.
echo Step 5: Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"

echo.
echo Step 6: Verifying GPU installation...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}');"

echo.
echo Installation complete!
echo You can now run the Jupyter notebook to train your BERT model.
pause
