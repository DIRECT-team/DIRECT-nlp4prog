# Code for the paper "DIRECT : A Transformer-based Model for Decompiled Variable Name Recovery"

<p align="center"> <img src="https://github.com/DIRECT-team/DIRECT-nlp4prog/blob/main/DIRECT_overview_3.png" width="80%"> </p>

This code is adapted from [DIRE](https://github.com/pcyin/dire).

### Setting up the data and packages

1. Run `pip install -r requirements.txt` to install the required packages.

1. Download the preprocessed data along with training-test splits from [this link](https://drive.google.com/drive/folders/19Rf7NtW56r6fz-ycldZq9hjxNr5osAJW?usp=sharing), and put them in `data/preprocessed_data`.

1. Create a symbolic link in the `src` folder by running `ln -s data ./src/data`.

### Pretraining

To pretrain the BERT encoder and decoder from scratch, run

```
python bert_pretrain.py [-decoder]
```

### Training

To train the DIRECT model from scratch, first pretrain the BERT encoder and decoder. Then run

```
python main.py -train
```

### Prediction

To evaluate a trained DIRECT model, assuming it is saved at `src/saved_checkpoints/direct.pth`, run

```
python main.py -restore -name direct [-val] [-top_k 1] [-approx] [-conf_piece] [-short_only]
```

Running the above evaluation dumps the predictions to `src/predictions/<fname>.pkl`. To evaluate these predictions with other metrics like Top-5 accuracy, Jaccard distance and Character Error Rate, run

```
python top5_analysis.py -fname <fname>.pkl
```


### Results

| Model       | Accuracy(%) | Top-5 Accuracy (%) | CER  | Jaccard Dist |
|-------------|-------------|--------------------|------|--------------|
| DIRE        | 35.8        | 41.5               | .664 | .537         |
| DIRECT      | 42.8        | 49.3               | .663 | .501         |
| Improvement | 20%         | 19%                | .2%  | 6.5%         |

