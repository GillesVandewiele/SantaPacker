import numpy as np


def sample_weight(_type):
    if _type == "horse":
        return max(0, np.random.normal(5, 2, 1)[0])
    if _type == "ball":
        return max(0, 1 + np.random.normal(1, 0.3, 1)[0])
    if _type == "bike":
        return max(0, np.random.normal(20, 10, 1)[0])
    if _type == "train":
        return max(0, np.random.normal(10, 5, 1)[0])
    if _type == "coal":
        return 47 * np.random.beta(0.5, 0.5, 1)[0]
    if _type == "book":
        return np.random.chisquare(2, 1)[0]
    if _type == "doll":
        return np.random.gamma(5, 1, 1)[0]
    if _type == "blocks":
        return np.random.triangular(5, 10, 20, 1)[0]
    if _type == "gloves":
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]

GIFT_CLASSES = ['horse', 'ball', 'bike', 'train', 'coal', 'book', 'doll', 'blocks', 'gloves']

NR_SAMPLES = int(1e6)

samples = {}
for GIFT_CLASS in GIFT_CLASSES:
    samples[GIFT_CLASS] = []


for GIFT_CLASS in GIFT_CLASSES:
    for SAMPLE in range(NR_SAMPLES):
        samples[GIFT_CLASS].append(sample_weight(GIFT_CLASS))
    print(GIFT_CLASS, np.percentile(samples[GIFT_CLASS], 55))