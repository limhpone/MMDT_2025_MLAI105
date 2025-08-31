# Fuzzy Evaluation Report

**Dataset:** BLURRY_NOISY  
**Evaluation Method:** Multi-Fuzzy Similarity Analysis  

---

## Evaluation Methodology

- **Exact Match:** Identical predictions (100% score)
- **High Similarity:** 60%+ fuzzy match (e.g., 'snow_leopard' vs 'leopard')
- **Medium Similarity:** 30-60% fuzzy match (e.g., 'ice_bear' vs 'polar_bear')
- **Low Similarity:** <30% fuzzy match (unrelated terms)

## Fuzzy Accuracy Results

| Model | Exact Top-1 | Fuzzy Top-1 | Fuzzy Top-3 | Weighted Score |
|-------|-------------|-------------|-------------|----------------|
| ResNet50 | 9.1% | 24.2% | 33.3% | 15.6% |
| VGGNet16 | 9.1% | 21.2% | 36.4% | 14.5% |
| InceptionV3 | 12.1% | 30.3% | 36.4% | 20.1% |
| ConvNeXt | 12.1% | 24.2% | 39.4% | 17.9% |
| EfficientNet | 12.1% | 27.3% | 39.4% | 18.5% |

---

## Detailed Model Analysis

### RESNET50

**Exact Accuracy:**
- Top-1: 9.1%
- Top-2: 12.1%
- Top-3: 12.1%

**Fuzzy Accuracy:**
- Top-1: 24.2%
- Top-2: 30.3%
- Top-3: 33.3%

**Similarity Breakdown:**
- Exact: 4
- High: 0
- Medium: 7
- Low: 22

### VGGNET16

**Exact Accuracy:**
- Top-1: 9.1%
- Top-2: 9.1%
- Top-3: 12.1%

**Fuzzy Accuracy:**
- Top-1: 21.2%
- Top-2: 27.3%
- Top-3: 36.4%

**Similarity Breakdown:**
- Exact: 4
- High: 0
- Medium: 8
- Low: 21

### INCEPTIONV3

**Exact Accuracy:**
- Top-1: 12.1%
- Top-2: 15.2%
- Top-3: 15.2%

**Fuzzy Accuracy:**
- Top-1: 30.3%
- Top-2: 36.4%
- Top-3: 36.4%

**Similarity Breakdown:**
- Exact: 5
- High: 0
- Medium: 7
- Low: 21

### CONVNEXT

**Exact Accuracy:**
- Top-1: 12.1%
- Top-2: 12.1%
- Top-3: 15.2%

**Fuzzy Accuracy:**
- Top-1: 24.2%
- Top-2: 36.4%
- Top-3: 39.4%

**Similarity Breakdown:**
- Exact: 5
- High: 0
- Medium: 8
- Low: 20

### EFFICIENTNET

**Exact Accuracy:**
- Top-1: 12.1%
- Top-2: 12.1%
- Top-3: 12.1%

**Fuzzy Accuracy:**
- Top-1: 27.3%
- Top-2: 39.4%
- Top-3: 39.4%

**Similarity Breakdown:**
- Exact: 4
- High: 0
- Medium: 9
- Low: 20

## Recommendations

- **Best Exact Accuracy:** InceptionV3 (12.1%)
- **Best Fuzzy Accuracy:** InceptionV3 (30.3%)
- **Best Weighted Score:** InceptionV3 (20.1%)
