# api-doc-kn-identification

This project provides NLP tools based on LSTM for Multi-Lable Calssification of Documents.

To run the lstm classifier:

 python scripts/multi_label_classifier_lstm.py <glove_path> <embedding_dim> <train_data_path> <test_data_path> <result_path> <train_embeddings>

 *train_embeddings: False (for CC), True (for CCOTF)

dependencies:
 + anaconda
 + keras
 + sklearn
 + csv

path to the embeddings of each model:

 CC & CCOTF:
  http://nlp.stanford.edu/data/glove.840B.300d.zip

 SO:
	Can be requested form the authors.

 SOAPI:
	Can be requested form the authors.

The complete dataset can be requested from the authors or should be scraped from the Internet.

When (re-)using (part) of this work, please cite the following publication:

```
@inproceedings{FMM19,
  title={On Using Machine Learning to Identify Knowledge in API Reference Documentation},
  author={Fucci, Davide and M. Alizadeh B., Alireza and Maalej, Walid},
  booktitle={27th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  pages={103--112},
  year={2019},
  doi={10.1145/3338906.3338943}
  organization={IEEE}
}
```
