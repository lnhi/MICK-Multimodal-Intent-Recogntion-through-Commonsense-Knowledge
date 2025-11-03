# MICK – Multimodal Intent Recognition through Commonsense Knowledge Extraction

## 1. Generate Commonsense Knowledge using COMET & VisualCOMET

### 1.1 Prepare COMET (Textual Commonsense)

```bash
cd comet-commonsense
bash scripts/setup/get_atomic_data.sh
bash scripts/setup/get_conceptnet_data.sh
bash scripts/setup/get_model_files.sh
python scripts/data/make_atomic_data_loader.py
python scripts/data/make_conceptnet_data_loader.py
```

Download pre-trained COMET model:  
[https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB](https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB)

### 1.2 Prepare VisualCOMET (Visual Commonsense Inference)

```bash
cd ../visual-comet
git clone --recursive https://github.com/jamespark3922/visual-comet.git .
pip install -r requirements.txt
```

Download annotations and visual features:

```bash
mkdir -p data/visualcomet && cd data/visualcomet
wget https://storage.googleapis.com/ai2-mosaic/public/visualcomet/visualcomet.zip
unzip visualcomet.zip

wget https://storage.googleapis.com/ai2-mosaic/public/visualcomet/features.zip
unzip features.zip
```

Update paths in `config.py`:

```python
VCR_IMAGES_DIR = "/path/to/vcr1images"
VCR_FEATURES_DIR = "/path/to/features"
```

### 1.3 Generate Commonsense Inferences

#### Textual Commonsense (COMET)

```bash
cd ../../comet-commonsense
mkdir -p input output
# Put your text utterances (one per line) in input/text.txt
python generate_relations.py --input_file input/text.txt --output_dir output/
```

#### Visual Commonsense (VisualCOMET)

```bash
cd ../visual-comet
python scripts/run_generation.py \
  --data_dir data/visualcomet/ \
  --model_name_or_path experiments/image_inference/ \
  --split val \
  --num_beams 5 \
  --output_file ../output/visualcomet_inferences.json
```
---

## 2. Running MICK (shark implementation)

```bash
cd ../mick
pip install -r requirements.txt
```

### Download MIntRec Dataset

```bash
# MIntRec dataset and features
https://drive.google.com/drive/folders/18iLqmUYDDOwIiiRbgwLpzw76BD62PK0p
```

Extract to: `datasets/MIntRec/`

### Run Training & Evaluation

```bash
sh scripts/run_mick.sh
```


## 3. Acknowledgments

This work builds upon:
- [MIntRec](https://github.com/thuiar/MIntRec)
- [VisualCOMET](https://github.com/jamespark3922/visual-comet)
- [COMET](https://github.com/allenai/comet-commonsense)

We sincerely thank the authors for their open-source contributions.


