# Classification of pairs Sequence feature/Categorical feature 
This small repository enables running training of a few simple models (Random Forest, Fully Connected, Bert + Fully Connected).
The data consisting of 2 features -a sequence and a category- is private and not made available here.

## Description
I developed this code to try very simple solutions on medical data classification. It consists mainly of a few notebooks to run the different experiments. \
The notebooks call modules that I developed: typically 2 simple pytorch datasets to load the features with a an oversampler (for unbalanced datasets), as well as pytorch models and preprocessing functions (one hot encoder etc). \
Feel free to use it on data that has similar features, as only the preprocessing would change. 

## Getting Started
### Dependencies
cuda-libraries=11.8.0
imbalanced-learn=0.11.0 \
jupyter_core=5.5.0 \
matplotlib=3.8.0 \
numpy=1.26.2 \
pandas=2.1.1 \
python=3.11.6 \
pytorch=2.1.1 \
pytorch-cuda=11.8 \
scikit-learn=1.3.2 \
transformers=4.32.1
### Installing
cd SimpleSequenceClassif \
conda create --name {env} --file requirements.txt
### Executing program
Simply run notebooks cells.
## Help
Please reach out.
## Authors
Mathieu Charbonnel
## Version History

* December 2023
    * Initial Release

## License
This project is not licensed.

## Acknowledgments
This work utilizes the BERT model, which was introduced by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova in the paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," published in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.

Reference:
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019).

We also express our gratitude to the contributors of the Hugging Face Transformers library for providing a user-friendly and efficient implementation of the BERT model.

Reference:
Hugging Face Transformers Library. Available at: https://github.com/huggingface/transformers

