
import json

all_story_counts = []

total_counts = {
    "Beliefs/ Knowledge": 0,
    "Emotion": 0,
    "Desire/ Intention": 0,
    "Others": 0
}

def count_words_in_categories(text):
    beliefs_keywords = ["know", "think", "learn", "understand", "perceive", "believe", "forget", "guess", "notice", "recognize"]
    emotion_keywords = ["feel"]
    desire_keywords = ["want", "wish", "hope", "decide", "prefer"]
    others_keywords = ["imagine", "expect", "remember"]

    words = text.lower().split()
    category_counts = {
        "Beliefs/ Knowledge": 0,
        "Emotion": 0,
        "Desire/ Intention": 0,
        "Others": 0
    }

    for word in beliefs_keywords:
        count = words.count(word) + words.count(word + "s") + words.count(word + "es")
        category_counts["Beliefs/ Knowledge"] += count
        if count > 0: total_counts ["Beliefs/ Knowledge"] += 1
    for word in emotion_keywords:
        count = words.count(word) + words.count(word + "s") + words.count(word + "es")
        category_counts["Emotion"] += count
        if count > 0: total_counts ["Emotion"] += 1
    for word in desire_keywords:
        count = words.count(word) + words.count(word + "s") + words.count(word + "es")
        category_counts["Desire/ Intention"] += count
        if count > 0: total_counts ["Desire/ Intention"] += 1
    for word in others_keywords:
        count = words.count(word) + words.count(word + "s") + words.count(word + "es")
        category_counts["Others"] += count
        if count > 0: total_counts ["Others"] += 1

    return category_counts

def main():
    file_path = "../tinystories_words/tinystories_rows.txt"
    log_path = "../tinystories_words/mentalistic_verbs_all.json"

    # file_path = "../tinystories_words/tinystories_rows_gpt4.txt"
    # log_path = "../tinystories_words/mentalistic_verbs_gpt4.json"
    
    with open(file_path, "r") as file:
        stories = file.readlines() 

    for i, story in enumerate(stories, start=1):
        category_counts = count_words_in_categories(story)
        print(f"Story {i}:")
        for category, count in category_counts.items():
            print(f"{category}: {count} words")
        all_story_counts.append(category_counts)
        print("-" * 30)
    
    print(total_counts)
    entry = { "total_counts": total_counts, "catagory_counts": all_story_counts }

    with open(log_path, "w") as f:
        json.dump(entry, f)

if __name__ == "__main__":
    main()
