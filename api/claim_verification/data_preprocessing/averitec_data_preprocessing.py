import csv, json

def preprocess_to_csv(input_path=None, output_path=None, sample_output_path=None, sample_size=50):
    with open(input_path, 'r') as f:
        data = json.load(f)

    rows = []
    for ex in data:
        claim = ex.get("claim", "").strip()
        label = ex.get("label", "").strip().upper()

        evidence_sents = []
        for q in ex.get("questions", []):
            for ans in q.get("answers", []):
                explanation = ans.get("boolean_explanation")
                if explanation:
                    evidence_sents.append(explanation.strip())

        # normalize label
        if label == "CONFLICTING EVIDENCE/CHERRYPICKING":
            label = "NOT ENOUGH INFO"
            
        if label == "NOT ENOUGH EVIDENCE":
            label = "NOT ENOUGH INFO"

        valid_labels = {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}

        if evidence_sents and label in valid_labels:
            # 🔥 convert list → JSON string
            evidence_json = json.dumps(evidence_sents)
            rows.append([claim, evidence_json, label])

    # write full output
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["claim", "evidence", "label"])
        writer.writerows(rows)

    # write sample
    with open(sample_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["claim", "evidence", "label"])
        writer.writerows(rows[:sample_size])

    print("Done. Saved:", output_path)

preprocess_to_csv(
    input_path='../data/raw/averitec.json',
    output_path='../data/processed/averitec.csv',
    sample_output_path='../data/processed/averitec_sample.csv',
    sample_size=50
)