Download and extract these files in the `data` directory.

The following file can be downloaded from [the link](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/data_train_val.zip)
- data_img.h5
- data_prepro.h5
- data_prepro.json

- [Glove vectors](http://nlp.stanford.edu/data/glove.6B.zip): We use 300 embedding dimension, so we only require `glove.6B.300d.txt`
- [Validation COCO Annotations](http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip): For validation we need `mscoco_val2014_annotations.json`

When you run the `mainfile`, the model makes the embeddings file. Alternatively, you could download the pre-trained embeddings file
- [embeddings_300.h5](https://drive.google.com/open?id=1o9nLSB9zwmWlbtX2B1ahaDPuYTGmfk2a)
