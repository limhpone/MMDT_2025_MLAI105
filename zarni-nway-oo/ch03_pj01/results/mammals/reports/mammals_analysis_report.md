# CNN Model Analysis Report

**Dataset:** MAMMALS  
**Top-K Value:** 3  
**Generated:** 2025-06-27 07:08:38  

---

## Performance Summary

| Model | Rank | Avg Confidence | Avg Time (s) | High Confidence % | Overall Score |
|-------|------|----------------|--------------|-------------------|---------------|
| ResNet50 | #2 | 0.738 | 0.056 | 55.3% | 75.9 |
| VGGNet16 | #1 | 0.768 | 0.075 | 61.7% | 80.1 |
| InceptionV3 | #3 | 0.702 | 0.050 | 59.6% | 60.4 |
| ConvNeXt | #4 | 0.716 | 0.278 | 55.3% | 20.9 |
| EfficientNet | #5 | 0.685 | 0.341 | 40.4% | 0.0 |

## InceptionV3 Analysis

- **Rank:** #3 out of 5 models
- **Confidence Score:** 0.702
- **Speed Performance:** 0.050s average
- **Reliability:** 59.6% high confidence predictions

## Recommendations

- **Best Overall:** VGGNet16
- **Fastest:** InceptionV3
- **Most Confident:** VGGNet16
