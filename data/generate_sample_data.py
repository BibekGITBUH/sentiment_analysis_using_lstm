"""
Generates a synthetic but linguistically varied labeled sentiment dataset
so the pipeline can run fully offline. Replace with a real dataset (IMDB,
Sentiment140, Amazon Reviews, etc.) by producing a CSV with the same
`text,label` schema.
"""
import csv
import random
import os

random.seed(42)

POSITIVE_TEMPLATES = [
    "I absolutely loved {subject}, it exceeded all my expectations.",
    "{subject} was fantastic, I would highly recommend it to anyone.",
    "What a wonderful experience with {subject}, truly outstanding quality.",
    "This {subject} is amazing, best purchase I've made in a while.",
    "I'm so happy with {subject}, everything worked perfectly.",
    "{subject} exceeded my expectations, the quality is superb.",
    "Great {subject}! Fast, reliable, and exactly what I needed.",
    "I can't stop recommending {subject} to my friends, it's brilliant.",
    "The {subject} was delightful from start to finish, five stars.",
    "Absolutely fantastic {subject}, will definitely buy again.",
    "This {subject} made my day, I'm thrilled with the results.",
    "Excellent {subject}, the staff and service were both wonderful.",
    "{subject} is a masterpiece, beautifully crafted and thoughtful.",
    "I had an amazing time with {subject}, highly satisfying overall.",
    "Top notch {subject}, worth every penny and more.",
]

NEGATIVE_TEMPLATES = [
    "I hated {subject}, it was a complete waste of money.",
    "{subject} was terrible, nothing worked as advertised.",
    "What a disappointing experience with {subject}, very frustrating.",
    "This {subject} is awful, I regret buying it immediately.",
    "I'm so upset with {subject}, everything went wrong.",
    "{subject} fell short of my expectations, the quality is poor.",
    "Terrible {subject}! Slow, unreliable, and not what I needed.",
    "I would never recommend {subject} to anyone, it's horrible.",
    "The {subject} was miserable from start to finish, one star.",
    "Absolutely awful {subject}, will never buy again.",
    "This {subject} ruined my day, I'm furious about the results.",
    "Poor {subject}, the staff and service were both unpleasant.",
    "{subject} is a disaster, badly made and careless.",
    "I had a horrible time with {subject}, deeply disappointing overall.",
    "Overpriced {subject}, not worth a penny of what I paid.",
]

NEUTRAL_TEMPLATES = [
    "{subject} was okay, nothing special but nothing terrible either.",
    "I have mixed feelings about {subject}, some parts were fine.",
    "{subject} did what it was supposed to do, an average experience.",
    "It's an average {subject}, met basic expectations and no more.",
    "{subject} was fine, I don't have strong feelings either way.",
    "The {subject} arrived on time, quality was as described.",
    "Decent {subject}, could be better but it's not bad.",
    "{subject} works as expected, nothing more to add.",
    "I neither liked nor disliked {subject}, it was just fine.",
    "The {subject} was fairly standard compared to similar options.",
    "{subject} is acceptable for the price, nothing memorable.",
    "It's a middle of the road {subject}, some pros and some cons.",
    "The {subject} was moderate in quality, could go either way.",
    "{subject} matched the description, no surprises good or bad.",
    "An unremarkable {subject}, does the job without any flair.",
]

SUBJECTS = [
    "the restaurant", "this laptop", "the movie", "the hotel room", "this book",
    "the customer service", "the phone", "this software update", "the concert",
    "the flight experience", "this course", "the delivery service", "the app",
    "this coffee shop", "the vacation package", "this pair of shoes",
    "the new headphones", "the online store", "this streaming service",
    "the gym membership", "this video game", "the car rental", "the bakery",
    "this website", "the airline", "the doctor's visit", "this backpack",
    "the tutoring session", "the smartwatch", "this camera",
]

EXTRA_POSITIVE = [
    "Loved every minute of it!", "Can't wait to come back.", "Simply the best.",
    "Highly recommend to everyone I know.", "Made my week so much better.",
]
EXTRA_NEGATIVE = [
    "Never again.", "Total letdown.", "Waste of time and money.",
    "Would not recommend to my worst enemy.", "So disappointed right now.",
]
EXTRA_NEUTRAL = [
    "Might try it again, might not.", "It is what it is.",
    "No strong opinion either way.", "Could go either way honestly.",
    "Not much else to say about it.",
]


def build_dataset(n_per_class=200):
    rows = []
    for _ in range(n_per_class):
        subj = random.choice(SUBJECTS)
        text = random.choice(POSITIVE_TEMPLATES).format(subject=subj)
        if random.random() < 0.5:
            text += " " + random.choice(EXTRA_POSITIVE)
        rows.append((text, "positive"))

        subj = random.choice(SUBJECTS)
        text = random.choice(NEGATIVE_TEMPLATES).format(subject=subj)
        if random.random() < 0.5:
            text += " " + random.choice(EXTRA_NEGATIVE)
        rows.append((text, "negative"))

        subj = random.choice(SUBJECTS)
        text = random.choice(NEUTRAL_TEMPLATES).format(subject=subj)
        if random.random() < 0.5:
            text += " " + random.choice(EXTRA_NEUTRAL)
        rows.append((text, "neutral"))

    random.shuffle(rows)
    return rows


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "sample_reviews.csv")
    rows = build_dataset(n_per_class=200)  # 600 total rows, balanced 3-class

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
