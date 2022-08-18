# Word Embeddings of Material Science Literature

## Set up 
1. Make sure you have at least Python 3.8 installed. We recommend using [conda](https://conda.io). 
2. Run `pip install .` from within our directory to install all requirements. 
3. Place a Gensim model called `pretrained_embeddings` in `data/material2vec`
4. Run the evaluation, and show and save plots by calling `python `

We can recommend to use the trained mat2vec embeddings form Tshitoyan et al., provided in 
their [GitHup Repository](https://github.com/materialsintelligence/mat2vec) [1].

Our focus is on heusler compounds, therefore, we provide some visualizations to analyze the embeddings of those 
materials in space. 

## Related Work
* [1] Tshitoyan, V., Dagdelen, J., Weston, L. et al. Unsupervised word embeddings capture latent knowledge from materials science literature. Nature 571, 95–98 (2019). https://doi.org/10.1038/s41586-019-1335-8