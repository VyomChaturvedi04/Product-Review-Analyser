"""
data/generate_sample.py
Generates a synthetic sample CSV (500 reviews) for quick testing
without needing to download the full Kaggle dataset.

For the real dataset, download from:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
and place 'Reviews.csv' in this directory.

Usage:
    python data/generate_sample.py
"""

import csv
import random
import os

random.seed(42)

POSITIVE_SENTENCES = [
    "This product is absolutely amazing and exceeded all my expectations.",
    "Great quality for the price. Would definitely recommend to anyone.",
    "Works perfectly. Very happy with this purchase!",
    "Fast shipping and the product is exactly as described.",
    "Best purchase I have made in a long time. Love it!",
    "Excellent build quality and very durable. Highly recommend.",
    "My whole family loves it. Will be buying again.",
    "Perfect product. Does exactly what it says it will do.",
    "Outstanding customer service and a fantastic product.",
    "Very impressed with the quality. Five stars!",
]

NEGATIVE_SENTENCES = [
    "Complete waste of money. Broke after one day of use.",
    "Terrible quality. Nothing like the pictures shown.",
    "Do not buy this product. It stopped working within a week.",
    "Extremely disappointed. The product is cheaply made.",
    "Worst purchase I have ever made. Total garbage.",
    "Returned immediately. The item was damaged on arrival.",
    "Poor quality control. My product had several defects.",
    "False advertising. This product does not work as described.",
    "Zero stars if I could. Horrible experience overall.",
    "Save your money and buy something else.",
]

NEUTRAL_SENTENCES = [
    "It is okay for the price. Nothing special but gets the job done.",
    "Average product. Some good points and some bad points.",
    "Works as expected. Not great, not terrible.",
    "Decent quality but there is definitely room for improvement.",
    "Somewhat useful but not as good as I had hoped.",
    "Mixed feelings. Some features are great, others not so much.",
    "It does what it claims but the build quality could be better.",
    "Reasonable value for money but I expected a bit more.",
    "Mediocre performance overall. Would consider alternatives.",
    "It is fine. Does the job but nothing to write home about.",
]


def generate_review(rating: int) -> str:
    if rating >= 4:
        pool = POSITIVE_SENTENCES
    elif rating == 3:
        pool = NEUTRAL_SENTENCES
    else:
        pool = NEGATIVE_SENTENCES

    num_sentences = random.randint(2, 5)
    sentences = random.choices(pool, k=num_sentences)
    return " ".join(sentences)


def main():
    out_path = os.path.join(os.path.dirname(__file__), "reviews_sample.csv")
    n = 500

    ratings_dist = {5: 200, 4: 100, 3: 80, 2: 70, 1: 50}
    rows = []
    idx = 1
    for rating, count in ratings_dist.items():
        for _ in range(count):
            rows.append({"Id": idx, "Score": rating, "Text": generate_review(rating)})
            idx += 1

    random.shuffle(rows)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Id", "Score", "Text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {n} sample reviews → {out_path}")
    print("Rating distribution:")
    for r, c in ratings_dist.items():
        print(f"  {r} stars: {c} reviews")


if __name__ == "__main__":
    main()
