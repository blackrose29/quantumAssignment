# Dataset: UCI Banknote Authentication

| Item | Value |
|---|---|
| Name | Banknote Authentication |
| Repository | UCI Machine Learning Repository, dataset ID 267 |
| Page | https://archive.ics.uci.edu/dataset/267/banknote+authentication |
| DOI | 10.24432/C55P57 |
| Creator | Volker Lohweg (dataset donated 2012) |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Size | 1,372 rows, 4 numeric features + 1 integer label, no missing values |
| Classes | 762 rows with label 0, 610 rows with label 1 |
| Related paper | V. Lohweg et al., "Banknote authentication with mobile devices," *Proc. SPIE* 8665, 866507, 2013, doi:10.1117/12.2001444 |

## How the data were produced

According to the UCI description, images were taken of genuine and forged banknote-like specimens with an industrial camera normally used for print inspection. The images are 400 × 400 pixel grey-scale pictures at about 660 dpi. A wavelet transform was applied to each image, and four statistics were extracted:

| Column | Meaning |
|---|---|
| `variance` | variance of the wavelet-transformed image |
| `skewness` | skewness of the wavelet-transformed image |
| `curtosis` | kurtosis of the wavelet-transformed image (spelled `curtosis` in the dataset) |
| `entropy` | entropy of the image |
| `class` | label, 0 or 1 |

**Label meaning.** The UCI page does not name the classes. This project follows earlier quantum-ML studies on this dataset (Kariya and Behera, arXiv:2112.06912; Kölle et al., arXiv:2311.05559): **0 = genuine, 1 = forged**. This reading matches the reported counts of 762 genuine and 610 forged notes.

**Duplicates.** The file contains 24 exactly duplicated rows. They are kept, as in other published work, and this is listed as a limitation in the report.

## Download

The code downloads the dataset automatically (`src/data_preprocessing.py::download_dataset`) if `data/data_banknote_authentication.txt` is missing. To download it manually:

```bash
curl -L -o banknote.zip "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"
```

```bash
unzip banknote.zip -d data/
```

The file has no header row. Each line is `variance,skewness,curtosis,entropy,class`, for example `3.6216,8.6661,-2.8073,-0.44699,0`.

SHA-256 of the copy used for all reported results:
`d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9`
