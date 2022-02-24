# Install tools
pip install kaggle
pip install unzip

# Download Dataset
kaggle datasets download -d kylewang1999/hate-speech-datasets
unzip hate-speech-datasets.zip -d datasets
rm *.zip