# HERDPhobia: A dataset for Hate speech detection against Fulani Herdsmen in Nigeria

## Description

This project creates a model for the detection of hate speech against the Fulani herdsmen in Nigeria. Based on the original HERDPhobia dataset (Aliyu et al., 2022), the collected tweets were annotated into 2 categories: hate (HT) and non-hate (NHT). The original dataset also included an 'indeterminate' category that was excluded in the training process. This dataset addresses the growing concern of online hate speech targeting specific ethnic groups in Nigeria, particularly the Fulani herdsmen, who have been subject to stereotyping and hate speech in online discourse.

The original dataset contains tweets in three languages: English (97.2%), Hausa (1.8%), and Nigerian-Pidgin (1%), reflecting the multilingual nature of Nigerian social media discourse.

## Link to paper
https://arxiv.org/abs/2211.15262

## DOI
https://doi.org/10.48550/arXiv.2211.15262

## Project Structure

This repository contains the implementation of various models for hate speech detection:

- `Grp_Ass_Base_model.ipynb`: Baseline model implementation
- `Grp_Ass_Augm_Data.ipynb`: Improved model with data augmentation techniques

## Dataset

The dataset consists of tweets collected and annotated for hate speech detection:
- Training set: 3,090 tweets
- Validation set: 441 tweets  
- Test set: 884 tweets

The data shows imbalance with approximately 80% non-hate and 20% hate speech examples.

## Models

### Baseline Model

We implemented a baseline model using the pretrained `masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0` transformer model from Hugging Face. This model was fine-tuned on our dataset with the following hyperparameters:

- Learning rate: 2e-5
- Batch size: 8
- Training epochs: 5
- Weight decay: 0.01

### Improved Model

Our improved model incorporates data augmentation to address class imbalance. Key improvements include:

- Balanced dataset through augmentation of the minority class
- Increased learning rate to 5e-5
- Mixed precision training (FP16)
- Warmup ratio of 0.1
- Best model selection based on F1 score

## Results

### Baseline Model Performance

- Accuracy: 84%
- F1 Score: 84%
- Precision: 84%
- Recall: 84%

Class-specific results:
- Class 0 (Non-hate): Precision 90%, Recall 90%, F1 90%
- Class 1 (Hate): Precision 60%, Recall 61%, F1 60%

### Improved Model Performance

- Accuracy: 84%
- F1 Score: 82%
- Precision: 82%
- Recall: 83%

Class-specific results:
- Class 0 (Non-hate): Precision 88%, Recall 91%, F1 89% 
- Class 1 (Hate): Precision 58%, Recall 51%, F1 54%

### Comparison with Original Paper

The original HERDPhobia paper (Aliyu et al., 2022) reported significantly higher performance metrics:

| Model | Weighted F1-score |
|-------|------------------|
| XLM-T (Barbieri et al., 2022) | 99.83% |
| mBERT (Devlin et al., 2019) | 80.96% |
| AfriBERTa (Ogueji et al., 2021) | 78.07% |

Our implementation's performance differs substantially from the original paper, particularly compared to the XLM-T model which reported near-perfect performance. This disparity may be due to differences in model selection, hyperparameters, preprocessing approaches, and evaluation methodologies.

## Usage

### Requirements

```
transformers
datasets
torch
pandas
numpy
scikit-learn
```

### Training

To train the baseline model:

```python
# Load data
train_df = pd.read_csv("HERDPhobia/train.tsv", sep="\t")
dev_df = pd.read_csv("HERDPhobia/dev.tsv", sep="\t")
test_df = pd.read_csv("HERDPhobia/test.tsv", sep="\t")

# Convert to Hugging Face datasets
# See notebooks for complete implementation details
```

### Inference

```python
# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")

# Make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions.argmax().item()
```

## Limitations

Our implementation has several limitations that should be considered:

1. **Performance Gap**: Significant difference between our results (84% accuracy) and the original paper's results (up to 99.83% weighted F1).

2. **Class Imbalance**: Both our models struggle with the minority class (hate speech), achieving much lower F1 scores for hate detection than for non-hate detection.

3. **Data Augmentation Challenges**: Simple oversampling of the minority class did not substantially improve performance.

4. **Cultural and Linguistic Limitations**: Limited exploration of the multilingual aspect (English, Hausa, and Nigerian-Pidgin) and culturally specific expressions of hate.

5. **Methodological Differences**: Different hyperparameters than the original paper (5 epochs vs. 20, batch size 8 vs. 32), making direct comparisons difficult.

6. **Model Selection**: We used masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0 rather than the XLM-T, mBERT, and AfriBERTa models from the original paper.

## Conclusion

Our experiments show that while data augmentation helped balance the class distribution, the baseline model performed better on hate speech detection. This suggests that simply balancing the dataset is not enough, and more sophisticated techniques may be needed to improve the detection of minority class examples.

Both models show significantly higher performance on non-hate speech examples, indicating the challenge of detecting hate speech, which often uses specific linguistic patterns and contextual clues.

The substantial gap between our implementation's performance and the results reported in the original paper warrants further investigation and highlights the challenges of reproducing NLP research in specialized domains.

## Future Work

Based on our findings and the suggestions in the original paper, future work could include:

1. Replicating the original methodology with XLM-T to verify the reported 99.83% weighted F1 score
2. Developing more sophisticated data augmentation techniques
3. Expanding to include other ethnic groups in Nigeria
4. Adding hate speech types and intensity labeling
5. Exploring explainable AI approaches to increase model transparency

## Citation

If you use this dataset or the models in your research, please cite:

```
@article{HERDPhobia2022,
  title={HERDPhobia: A dataset for Hate speech detection against Fulani Herdsmen in Nigeria},
  author={Aliyu, Saminu Mohammad and Wajiga, Gregory Maksha and Murtala, Muhammad and Muhammad, Shamsuddeen Hassan and Abdulmumin, Idris and Ahmad, Ibrahim Said},
  journal={arXiv preprint arXiv:2211.15262},
  year={2022}
}
```

## License

[Specify license information here]

## Contributors

- Mouliom Pagna Mohamed Abdounour
- Dze-kum Shalom Chow
- Ombok Patrick
- Rachel Mukwania
- Tigist Fantahun
- HuggenFace Community 