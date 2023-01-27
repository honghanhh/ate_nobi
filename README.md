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

- The ACTER dataset can be downloaded at [AylaRT/ACTER](https://github.com/AylaRT/ACTER.git).
- The RSDO dataset can be downloaded at [Corpus of term-annotated texts RSDO5 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1400).

## 4. Implementation

### 4.1. Preprocessing

(Update in details after the paper is accepted)
The newly nested term labeling mechanism (NOBI) and the labeled data can be accessed at [honghanhh/NOBI](https://github.com/honghanhh/nobi).

## References

Tran, Hanh Thi Hong, et al. "[Can Cross-Domain Term Extraction Benefit from Cross-lingual Transfer?](https://link.springer.com/chapter/10.1007/978-3-031-18840-4_26)." Discovery Science: 25th International Conference, DS 2022, Montpellier, France, October 10–12, 2022, Proceedings. Cham: Springer Nature Switzerland, 2022.

## Contributors:

- 🐮 [TRAN Thi Hong Hanh](https://github.com/honghanhh) 🐮
- Prof. [Senja POLLAK](https://github.com/senjapollak)
- Prof. [Antoine DOUCET](https://github.com/antoinedoucet)
- Prof. [Matej MARTINC](https://github.com/matejMartinc)
