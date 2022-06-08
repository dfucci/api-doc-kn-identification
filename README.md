# api-doc-kn-identification

## Terms of use
Please, see our [license](LICENSE.md).  
When (re-)using (part) of this work, you must cite the following publication:

```
@inproceedings{FMM19,
  title={On Using Machine Learning to Identify Knowledge in API Reference Documentation},
  author={Fucci, Davide and Mollaalizadehbahnemiri Alireza and Maalej, Walid},
  booktitle={27th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  pages={103--112},
  year={2019},
  doi={10.1145/3338906.3338943}
  organization={IEEE}
}
```

## How to
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

 StackOverflow (SO):
	Can be requested from the authors.

 StackOverflow API (SOAPI):
	Can be requested from the authors.



