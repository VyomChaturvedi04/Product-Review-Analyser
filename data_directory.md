# Data Directory

## Real dataset (recommended for training)

Download the **Amazon Fine Food Reviews** dataset from Kaggle:

```
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
```

Place `Reviews.csv` in this directory. The file has ~568,000 rows with columns:
`Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text`

The training script uses the `Text` and `Score` columns.

## Quick-start sample (for testing)

Generate a 500-review synthetic sample without Kaggle:

```bash
python data/generate_sample.py
```

This creates `reviews_sample.csv` in this directory.
Then train with:

```bash
python model/train.py --data data/reviews_sample.csv
```
