import csv,json

def preprocess_to_csv(input_path=None, output_path=None, sample_output_path=None, sample_size=50):
    with open(input_path, 'r') as f:
        data = json.load(f)

    rows = []
    for ex in data:
        print("Processing example:", ex)
        claim = ex.get("claim", "").strip()
        label = ex.get("label", "").strip().upper()

        evidence_sents = []
        for q in ex.get("questions", []):
            for ans in q.get("answers", []):
                explanation = ans.get("boolean_explanation")
                if explanation:
                    evidence_sents.append(explanation.strip())

        if evidence_sents and label in {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO", "Conflicting Evidence/Cherry-picking"}:
            if label == "CONFLICTING EVIDENCE/CHERRY-PICKING":
                label = "NOT ENOUGH INFO"
            rows.append([claim, evidence_sents, label])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["claim", "evidence", "label"])
        writer.writerows(rows)
    
    with open(sample_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["claim", "evidence", "label"])
        writer.writerows(rows[:sample_size])
    

preprocess_to_csv('../data/raw/averitec.json', '../data/processed/averitec.csv', '../data/processed/averitec_sample.csv', sample_size=50)