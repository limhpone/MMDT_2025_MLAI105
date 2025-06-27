# Fuzzy Evaluation Report

**Dataset:** MAMMALS  
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
| ResNet50 | 34.0% | 55.3% | 57.4% | 46.0% |
| VGGNet16 | 34.0% | 55.3% | 55.3% | 46.0% |
| InceptionV3 | 34.0% | 57.4% | 57.4% | 46.9% |
| ConvNeXt | 34.0% | 55.3% | 57.4% | 46.0% |
| EfficientNet | 34.0% | 57.4% | 59.6% | 46.9% |

---

## Detailed Model Analysis

### RESNET50

**Exact Accuracy:**
- Top-1: 34.0%
- Top-2: 34.0%
- Top-3: 34.0%

**Fuzzy Accuracy:**
- Top-1: 55.3%
- Top-2: 57.4%
- Top-3: 57.4%

**Similarity Breakdown:**
- Exact: 16
- High: 3
- Medium: 8
- Low: 20

### VGGNET16

**Exact Accuracy:**
- Top-1: 34.0%
- Top-2: 34.0%
- Top-3: 34.0%

**Fuzzy Accuracy:**
- Top-1: 55.3%
- Top-2: 55.3%
- Top-3: 55.3%

**Similarity Breakdown:**
- Exact: 16
- High: 3
- Medium: 7
- Low: 21

### INCEPTIONV3

**Exact Accuracy:**
- Top-1: 34.0%
- Top-2: 34.0%
- Top-3: 34.0%

**Fuzzy Accuracy:**
- Top-1: 57.4%
- Top-2: 57.4%
- Top-3: 57.4%

**Similarity Breakdown:**
- Exact: 16
- High: 3
- Medium: 8
- Low: 20

### CONVNEXT

**Exact Accuracy:**
- Top-1: 34.0%
- Top-2: 34.0%
- Top-3: 34.0%

**Fuzzy Accuracy:**
- Top-1: 55.3%
- Top-2: 57.4%
- Top-3: 57.4%

**Similarity Breakdown:**
- Exact: 16
- High: 3
- Medium: 8
- Low: 20

### EFFICIENTNET

**Exact Accuracy:**
- Top-1: 34.0%
- Top-2: 34.0%
- Top-3: 34.0%

**Fuzzy Accuracy:**
- Top-1: 57.4%
- Top-2: 57.4%
- Top-3: 59.6%

**Similarity Breakdown:**
- Exact: 16
- High: 3
- Medium: 9
- Low: 19

## Recommendations

- **Best Exact Accuracy:** ResNet50 (34.0%)
- **Best Fuzzy Accuracy:** InceptionV3 (57.4%)
- **Best Weighted Score:** InceptionV3 (46.9%)
