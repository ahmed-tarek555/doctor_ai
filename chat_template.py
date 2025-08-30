import json

chat_template = """
Human: {human}
Ai: {ai}
"""

if __name__ == '__main__':
    with open('conversations.txt', encoding='utf-8', mode='r') as f:
        text = f.read()

    conversations = text.split('"The conversation between human and AI assistant.')
    conversations = [c.strip(' "\n') for c in conversations if c.strip()]

    dataset = []
    for convo in conversations:
        if "[|Human|]" in convo and "[|AI|]" in convo:
            human = convo.split("[|Human|]")[1].split("[|AI|]")[0].strip()
            ai = convo.split("[|AI|]")[1].strip()
            dataset.append({
                "Human": human,
                "Ai": ai
            })


    with open("chat_templated_convo.jsonl", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    def format_example(example):
        return f"Human: {example['Human']}\nAi: {example['Ai']}\n\n"


    ready_text = ''.join([format_example(item) for item in dataset])

