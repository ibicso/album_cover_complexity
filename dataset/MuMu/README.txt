MuMu is a Multimodal Music dataset with multi-label genre annotations that combines information from the Amazon Reviews dataset and the Million Song Dataset (MSD). The former contains millions of album customer reviews and album metadata gathered from Amazon.com. The latter is a collection of metadata and precomputed audio features for a million songs. 

To map the information from both datasets we use MusicBrainz. This process yields the final set of 147,295 songs, which belong to 31,471 albums. For the mapped set of albums, there are 447,583 customer reviews from the Amazon Dataset. The dataset have been used for multi-label music genre classification experiments in the related publication. In addition to genre annotations, this dataset provides further information about each album, such as genre annotations, average rating, selling rank, similar products, and cover image url. For every text review it also provides helpfulness score of the reviews, average rating, and summary of the review. 

The mapping between the three datasets (Amazon, MusicBrainz and MSD), genre annotations, metadata, data splits, text reviews and links to images are available here. Images and audio files can not be released due to copyright issues.

MuMu dataset (mapping, metadata, annotations and text reviews)
Data splits and multimodal embeddings for ISMIR and TISMIR multi-label classification experiments 

These data can be used together with the Tartarus deep learning library https://github.com/sergiooramas/tartarus.

Dataset compiled and authored by Sergio Oramas. Copyright ©2017 Music Technology Group, Universitat Pompeu Fabra. All Rights Reserved.

Scientific References

Please cite the following papers if using MuMu dataset or Tartarus library.

Oramas, S., Barbieri, F., Nieto, O., and Serra, X (2018). Multimodal Deep Learning for Music Genre Classification, Transactions of the International Society for Music Information Retrieval, V(1).

Oramas S., Nieto O., Barbieri F., & Serra X. (2017). Multi-label Music Genre Classification from audio, text and images using Deep Features. In Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR 2017).
