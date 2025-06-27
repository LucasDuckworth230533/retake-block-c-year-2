# Model Card for BERTje Dutch Emotion Classifier

## Model Details

**Model Name:** BERTje Dutch Emotion Classifier  
**Model Type:** Transformer-based model (BERT)  
**Base Model:** `wietsedv/bert-base-dutch-cased` (BERTje)  
**Tokenizer:** `wietsedv/bert-base-dutch-cased` tokenizer  
**Model Architecture:** BertForSequenceClassification with 7 output classes  
**Parameters:** ~110M parameters (BERT-base architecture)

## Intended Use

This model is intended for classifying Dutch sentences into one of 7 emotion categories:

- **Happiness** (9,030 samples)
- **Sadness** (6,080 samples) 
- **Anger** (2,950 samples)
- **Fear** (2,438 samples)
- **Surprise** (779 samples)
- **Disgust** (15 samples)
- **Neutral** (202 samples)

### Use Cases
- Detecting emotional tone in Dutch written text
- Sentiment analysis in Dutch customer reviews, social media posts, or other textual data
- Creating subtitles with tone tags for the auditory impaired in Dutch content
- Emotion-aware chatbots and virtual assistants for Dutch speakers

### Limitations
- The model is specifically trained for Dutch text and will not perform well on other languages
- Significant class imbalance exists, with very few examples of disgust (15) and neutral (202) emotions
- The model might not generalize well to domains that differ significantly from the training data (e.g., highly specialized texts, code-switching, etc.)
- It may struggle with context-dependent emotions, where the emotional tone depends on surrounding sentences or external context
- Performance is particularly poor on underrepresented classes (F1-score of 0.00 for fear, disgust, surprise, and neutral)

## Data Preprocessing

**Source Datasets:**
- Original datasets: "FRIENDS", "SMILE", "CARER" provided by BUaS in 2024
- Combined dataset size: 400,000+ datapoints
- Sampled dataset: 5% of original data (21,494 datapoints for training)
- Test dataset: Separate dataset with emotion mapping and deduplication

**Preprocessing Steps:**
1. **Data Combination:** Multiple emotion datasets were merged and unified
2. **Emotion Mapping:** Complex emotions were mapped to 7 core categories using predefined mappings:
   - Anger: ['disapproval', 'annoyance']
   - Fear: ['fear', 'nervousness'] 
   - Happiness: ['admiration', 'excitement', 'relief', 'amusement', 'optimism', 'approval', 'gratitude', 'caring', 'joy', 'pride', 'desire', 'love']
   - Sadness: ['sadness', 'embarrassment', 'disappointment', 'remorse']
   - Surprise: ['curiosity', 'realization', 'confusion', 'surprise']
   - Neutral: ['nan', 'neutral']
   - Disgust: ['disgust']

3. **Data Augmentation:** Applied synonym replacement using Dutch WordNet to address class imbalance:
   - Surprise: 3x augmentation
   - Anger: 15x augmentation  
   - Fear: 6x augmentation
   - Sadness: 3x augmentation
   - Disgust: 8x augmentation

4. **Data Cleaning:**
   - Removed duplicate sentences within training data
   - Removed sentences that appeared in both training and test sets
   - Handled missing values and NaN emotions
   - Text converted to string format for consistent processing

5. **Tokenization:** 
   - Used BERTje tokenizer with max_length=128
   - Applied padding and truncation
   - Generated attention masks for proper sequence handling

## Training Details

**Training Configuration:**
- **Model:** BERTje (`wietsedv/bert-base-dutch-cased`)
- **Training Data Size:** 21,494 datapoints (after augmentation: ~25,000+ datapoints)
- **Validation Split:** 20% of training data
- **Test Data Size:** 750 datapoints (after deduplication)
- **Max Sequence Length:** 128 tokens
- **Batch Size:** 16
- **Learning Rate:** 2e-5 (AdamW optimizer)
- **Epochs:** 3
- **Loss Function:** Cross-Entropy Loss
- **Hardware:** Local GPU (CUDA enabled)
- **Evaluation Metrics:** Accuracy, F1-score (weighted), Classification Report

**Training Process:**
- 80% training, 20% validation split with stratified sampling
- Model trained for 3 epochs with early stopping potential
- Evaluation performed on validation set after each epoch
- Final evaluation on held-out test set

## Performance

**Final Test Results:**
- **Accuracy:** 0.643 (64.3%)
- **Precision:** 0.621 (macro average)
- **Recall:** 0.643 (macro average)
- **F1-Score by Class:**
  - Happiness: 0.27
  - Anger: 0.20
  - Sadness: 0.07
  - Fear: 0.00
  - Disgust: 0.00
  - Surprise: 0.00
  - Neutral: 0.00

**Model Evaluation:**
The model demonstrates acceptable overall accuracy but shows significant challenges with class imbalance. Performance is heavily skewed toward the happiness class, while severely underrepresented classes (disgust, neutral, fear, surprise) show zero F1-scores, indicating the model fails to correctly classify these emotions.

## Error Analysis & Explainability

**Error Analysis:** Comprehensive error analysis has been performed and documented at:
https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group2-dutch/blob/main/evidence_map/task_8_evidence/Task_8_error_analysis.pdf

**Explainable AI (XAI):** Model interpretability analysis has been conducted and documented at:
https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group2-dutch/blob/main/evidence_map/task_9_evidence/XAI_report.md

## Usage

To use this model, load it using the Hugging Face `transformers` library:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer
model_name = "wietsedv/bert-base-dutch-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained('[path_to_your_fine_tuned_model]')

# Prepare input text (Dutch)
text = "Ik ben heel blij met dit resultaat!"
inputs = tokenizer(text, return_tensors="pt", max_length=128, 
                   padding="max_length", truncation=True)

# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)

# Class mapping
class_names = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
predicted_emotion = class_names[predicted_class.item()]
```

## Sustainability & Energy Usage

### Training Energy Consumption

**Training Configuration:**
- **Hardware:** Local GPU (estimated ~200W power draw)
- **Training Duration:** ~3 hours for 3 epochs
- **Energy Calculation:** 
  - Energy (kWh) = Runtime (hours) × Power Draw (Watts) ÷ 1000
  - Training Energy = 3h × 200W ÷ 1000 = **0.6 kWh**

**Environmental Impact Context:**
- Equivalent to charging a smartphone ~50 times
- Same energy as running a microwave for 36 minutes
- Significantly lower than cloud-based training due to local GPU usage

### Inference Energy Estimation

**Per Inference Calculation:**
- **GPU Power:** ~200W (estimated local GPU)
- **Inference Time:** ~0.1 seconds per sentence (estimated)
- **Energy per Inference:** (200W × 0.1s) ÷ 3600 = **0.0056 Wh**

**Scaling Estimates:**

| Usage Scale | Energy Consumption | Real-World Equivalent |
|-------------|-------------------|----------------------|
| 1,000 inferences | 5.6 Wh | Charging smartphone 1/2 full |
| 10,000 inferences | 56 Wh | Running laptop for 1 hour |
| 100,000 inferences | 560 Wh (0.56 kWh) | Running dishwasher once |
| 1,000,000 inferences | 5.6 kWh | Average household electricity for 5 hours |

**Monthly Usage Example:**
- 100 daily users × 10 inferences each × 20 days = 20,000 monthly inferences
- Monthly energy: 20,000 × 0.0056 Wh = **112 Wh (0.112 kWh)**
- Cost (at €0.25/kWh): **€0.028/month**

### Sustainability Recommendations

1. **Model Optimization:** Consider model distillation or quantization to reduce inference energy
2. **Batch Processing:** Process multiple texts simultaneously to reduce per-inference overhead
3. **Local Deployment:** Continue using local hardware to avoid cloud GPU energy overhead
4. **Class Balancing:** Improve training data balance to potentially reduce model size while maintaining performance

## System Requirements

**Training Requirements:**
- **Hardware:** GPU with minimum 8GB VRAM (local training performed)
- **Software:** Python 3.7+, PyTorch 1.6+, Transformers library
- **Memory:** 16GB+ RAM recommended
- **Storage:** 2GB for model and data

**Inference Requirements:**
- **Minimum:** CPU-based inference possible but slower
- **Recommended:** GPU with 4GB+ VRAM for efficient inference
- **Dependencies:** transformers, torch, numpy

## Limitations & Future Work

**Current Limitations:**
1. **Class Imbalance:** Severe imbalance leads to poor performance on minority classes
2. **Language Limitation:** Dutch-only, no multilingual capability
3. **Context Limitation:** Single sentence classification without conversational context
4. **Domain Specificity:** May not generalize to formal or technical Dutch texts

**Recommendations for Improvement:**
1. **Data Collection:** Gather more balanced training data, especially for underrepresented emotions
2. **Advanced Techniques:** Implement focal loss, class weighting, or oversampling strategies
3. **Model Architecture:** Experiment with newer architectures or ensemble methods
4. **Evaluation:** Implement more robust evaluation metrics for imbalanced datasets (e.g., macro F1, balanced accuracy)

## Citation & Acknowledgments

**Dataset:** Provided by Breda University of Applied Sciences (BUaS), 2024
**Base Model:** BERTje by Wietse de Vries et al.
**Development:** Created as part of AI/ML coursework focusing on sustainable NLP practices