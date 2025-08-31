# CNN Model Analysis Report

**Dataset:** BLURRY_NOISY  
**Top-K Value:** 3  
**Generated:** 2025-06-27 21:32:08  

---

## Performance Summary

| Model | Rank | Avg Confidence | Avg Time (s) | High Confidence % | Overall Score |
|-------|------|----------------|--------------|-------------------|---------------|
| ResNet50 | #1 | 0.569 | 0.065 | 30.3% | 93.9 |
| VGGNet16 | #3 | 0.447 | 0.080 | 15.2% | 33.8 |
| InceptionV3 | #2 | 0.537 | 0.058 | 24.2% | 87.1 |
| ConvNeXt | #4 | 0.517 | 0.292 | 18.2% | 31.2 |
| EfficientNet | #5 | 0.517 | 0.379 | 15.2% | 28.8 |

## InceptionV3 Analysis

- **Rank:** #2 out of 5 models
- **Confidence Score:** 0.537
- **Speed Performance:** 0.058s average
- **Reliability:** 24.2% high confidence predictions

## Recommendations

- **Best Overall:** ResNet50
- **Fastest:** InceptionV3
- **Most Confident:** ResNet50
