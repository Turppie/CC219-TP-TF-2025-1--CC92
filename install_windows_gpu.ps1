# PowerShell script for installing BERT dependencies on Windows with GPU
Write-Host "Installing Python dependencies for BERT Twitter Classification (Windows + GPU)" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green

Write-Host "`nStep 1: Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "`nStep 2: Installing PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "Note: This will install PyTorch with CUDA 11.8 support" -ForegroundColor Cyan
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

Write-Host "`nStep 3: Installing remaining dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nStep 4: Downloading spaCy model..." -ForegroundColor Yellow
python -m spacy download en_core_web_sm

Write-Host "`nStep 5: Downloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"

Write-Host "`nStep 6: Verifying GPU installation..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}');"

Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "You can now run the Jupyter notebook to train your BERT model." -ForegroundColor Green
Read-Host "Press Enter to continue"
