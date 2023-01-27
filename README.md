# Can Cross-domain Term Extraction Benefit from Cross-lingual Transfer and Nested Term Labeling?

## 1. Description

In this repo, we further extend our work in [Can Cross-domain Term Extraction Benefit from Cross-lingual Transfer?](https://github.com/honghanhh/ate-2022) by introducing a novel nested term labeling mechanism and evaluating the performance of the model in the cross-lingual and multi-lingual settings in comparison with the traditional BIO annotation regime.

---

## 2. Requirements

Please install all the necessary libraries noted in [requirements.txt](./requirements.txt) using this command:

```
pip install -r requirements.txt
```

## 3. Data

The experiments were conducted on 2 datasets:

||ACTER dataset| RSDO5 dataset|
|:-:|:-:|:-:|
|Languages|English, French, and Dutch|Slovenian|
|Domains|Corruption,  Wind energy, Equitation, Heart failure|Biomechanics, Chemistry, Veterinary, Linguistics |
|Original version|  [AylaRT/ACTER](https://github.com/AylaRT/ACTER.git) | [Corpus of term-annotated texts RSDO5 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1400) |

## 4. Implementation

### 4.1. Preprocessing

The newly nested term labeling mechanism (NOBI) and the labeled data can be accessed at [@honghanhh/nobi_annotation_regime](https://github.com/honghanhh/nobi_annotation_regime).

### 4.2. Workflow

The workflow of the model is described in our coming paper in 2023.
To reproduce the results, please run the following command:

```python
chmod +x run.sh
./run.sh
```

which will run the model that covers all the following scenarios:

- ACTER dataset with XLM-RoBERTa in mono-lingual, cros-lingual, and multi-lingual settings with both ANN and NES version with multi-lingual settings covering only three languages from ACTER and additional Slovenian add-ons (10 scenarios).

- RSDO5 dataset with XLM-RoBERTa in mono-lingual, cros-lingual, and multi-lingual settings with cross-lingual and multi-lingual taking into account the ANN and NES version (48 scenarios).

Note that the model produces the results for NOBI annotated set. To reproduce the results for BIO annotated set, please refers to [@honghanhh/ate-2022](https://github.com/honghanhh/ate-2022).

### 4.3. Model configuration

Feel free to hyper-parameter tune the model. The current settings are:
```python
num_train_epochs=20,             # total # of training epochs
per_device_train_batch_size=32,  # batch size per device during training
per_device_eval_batch_size=32,   # batch size for evaluation 
learning_rate=2e-5,              # learning rate
eval_steps = 500,
load_best_model_at_end=True,     # load the best model at the end of training
metric_for_best_model="f1",
greater_is_better=True
```

## 5. Results

Plesae refer the results and error analysis to our coming paper in 2023.

## References

Tran, Hanh Thi Hong, et al. "[Can Cross-Domain Term Extraction Benefit from Cross-lingual Transfer?](https://link.springer.com/chapter/10.1007/978-3-031-18840-4_26)." Discovery Science: 25th International Conference, DS 2022, Montpellier, France, October 10–12, 2022, Proceedings. Cham: Springer Nature Switzerland, 2022.

## Contributors:

- 🐮 [TRAN Thi Hong Hanh](https://github.com/honghanhh) 🐮
- Prof. [Senja POLLAK](https://github.com/senjapollak)
- Prof. [Antoine DOUCET](https://github.com/antoinedoucet)
- Prof. [Matej MARTINC](https://github.com/matejMartinc)
