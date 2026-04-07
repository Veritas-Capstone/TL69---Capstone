"""
tests for the bias detection server api
run the server first then run this file
"""

import requests
import sys
import time

URL = "http://localhost:8000"
passes = 0
fails = 0


def run_test(name, func):
    global passes, fails
    try:
        func()
        print(f"  PASS  {name}")
        passes += 1
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        fails += 1
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        fails += 1


# health endpoint
def test_health():
    r = requests.get(f"{URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["pipeline"] == "AA + LEACE"


# edge cases (empty, gibberish, no text, etc)
def test_empty():
    r = requests.post(f"{URL}/bias/analyze", json={"text": ""})
    assert r.status_code >= 400

def test_whitespace():
    r = requests.post(f"{URL}/bias/analyze", json={"text": "   \n\t  "})
    assert r.status_code >= 400

def test_no_text_field():
    r = requests.post(f"{URL}/bias/analyze", json={})
    assert r.status_code == 422

def test_gibberish():
    r = requests.post(f"{URL}/bias/analyze",
        json={"text": "asdfkjh qwerty zxcvbn mnbvcx lkjhgf poiuyt"})
    data = r.json()
    assert data["overall_bias"] in ("Not Political", "Uncertain")

def test_one_word():
    r = requests.post(f"{URL}/bias/analyze", json={"text": "Hello"})
    data = r.json()
    assert data["overall_bias"] in ("Not Political", "Uncertain")
    assert len(data["sentences"]) == 0

def test_not_political():
    text = ("The chocolate cake recipe requires two cups of flour, three eggs, "
            "and a cup of sugar. Preheat the oven to 350 degrees and bake for "
            "25 minutes until golden brown.")
    r = requests.post(f"{URL}/bias/analyze", json={"text": text})
    data = r.json()
    assert data["overall_bias"] == "Not Political"
    assert data["overall_confidence"] == 0.0
    assert len(data["sentences"]) == 0


# actual bias detection
def test_returns_bias():
    text = ("The Republican party has consistently pushed for tax cuts that benefit "
            "wealthy corporations while Democrats argue these policies increase "
            "income inequality and harm working families. Progressive lawmakers "
            "have introduced legislation to raise the minimum wage and expand "
            "healthcare access.")
    r = requests.post(f"{URL}/bias/analyze", json={"text": text})
    data = r.json()
    assert data["overall_bias"] in ("Left", "Center", "Right", "Uncertain")
    assert set(data["overall_probabilities"].keys()) == {"Left", "Center", "Right"}
    # probs should add up to 1
    total = sum(data["overall_probabilities"].values())
    assert 0.99 < total < 1.01
    assert data["checks"] > 0

def test_response_fields():
    text = ("Congress passed new immigration reform legislation today that "
            "strengthens border security while providing a pathway to citizenship "
            "for undocumented immigrants.")
    r = requests.post(f"{URL}/bias/analyze", json={"text": text})
    data = r.json()
    # top level
    for key in ["overall_bias", "overall_probabilities", "overall_confidence",
                "confidence_level", "sentences", "checks", "issues"]:
        assert key in data, f"missing {key}"
    # sentence level
    if data["sentences"]:
        s = data["sentences"][0]
        for key in ["text", "category", "description", "bias", "confidence", "probabilities"]:
            assert key in s, f"missing {key} in sentence"
        assert s["bias"] in ("Left", "Center", "Right", "Uncertain")
        assert 0 <= s["confidence"] <= 1

def test_confidence():
    text = ("The president signed an executive order rolling back environmental "
            "regulations that climate scientists say are critical to reducing "
            "carbon emissions and combating global warming.")
    r = requests.post(f"{URL}/bias/analyze", json={"text": text})
    data = r.json()
    assert 0 <= data["overall_confidence"] <= 1
    assert data["confidence_level"] in ("high", "moderate", "low", "none", "insufficient")
    for s in data["sentences"]:
        assert 0 <= s["confidence"] <= 1

def test_categories():
    text = ("Progressive Democrats unveiled their Green New Deal proposal calling "
            "for massive government spending on renewable energy. Conservative "
            "Republicans condemned the plan as socialist overreach that would "
            "destroy the American economy and eliminate jobs.")
    r = requests.post(f"{URL}/bias/analyze", json={"text": text})
    data = r.json()
    ok = {"Left-leaning", "Possibly Left-leaning", "Right-leaning",
            "Possibly Right-leaning", "Centrist", "Uncertain"}
    for s in data["sentences"]:
        assert s["category"] in ok, f"bad category: {s['category']}"

def test_issues_leq_checks():
    text = ("The liberal media continues to spread fake news about conservative "
            "policies while ignoring the successes of the Republican administration.")
    r = requests.post(f"{URL}/bias/analyze", json={"text": text})
    data = r.json()
    assert data["issues"] <= data["checks"]


# explainability 
def test_explain():
    r = requests.post(f"{URL}/bias/explain",
        json={"text": "The conservative government slashed welfare spending."})
    data = r.json()
    assert "top_tokens" in data and "text" in data
    assert len(data["top_tokens"]) <= 5
    for t in data["top_tokens"]:
        assert 0 <= t["score"] <= 1

def test_explain_empty():
    r = requests.post(f"{URL}/bias/explain", json={"text": ""})
    assert r.status_code >= 400

def test_explain_no_stopwords():
    r = requests.post(f"{URL}/bias/explain",
        json={"text": "The government implemented new progressive tax policies."})
    data = r.json()
    bad = {"the", "a", "an", "and", "or", "is", "was", "to", "for", "of"}
    for t in data["top_tokens"]:
        assert t["token"].lower() not in bad, f"stopword leaked: {t['token']}"


# latency on real articles 

# short, about 150 words
ART1 = (
    "Mayor Zohran Mamdani is facing criticism from advocates for homeless New Yorkers "
    "after abruptly reversing his policy pledge to end homeless encampment sweeps. "
    "City Hall officials said outreach workers with the Department of Homeless Services "
    "would begin notifying street homeless New Yorkers this week of plans to clear them "
    "out of public spaces. During a sweep, city sanitation workers often trash tents, "
    "makeshift encampments and other belongings if people refuse to pack up and go to a "
    "shelter or another location. Mamdani had called the encampment sweeps done by his "
    "predecessors Eric Adams and Bill de Blasio a failure because they rarely led to people "
    "being placed in permanent housing. But advocates for homeless New Yorkers say "
    "Mamdani's plan is more of the same, and will displace people while moving only a "
    "fraction of them into shelters or permanent housing."
)

# medium, about 450 words
ART2 = (
    "The Supreme Court on Friday ruled that President Donald Trump violated federal law "
    "when he unilaterally imposed sweeping tariffs across the globe, a striking loss for "
    "the White House on an issue that has been central to the president's foreign policy "
    "and economic agenda. Chief Justice John Roberts wrote the majority opinion and the "
    "court agreed 6-3 that the tariffs exceeded the law. The court, however, did not say "
    "what should happen to the more than $130 billion in tariffs that has already been "
    "collected. The president asserts the extraordinary power to unilaterally impose tariffs "
    "of unlimited amount, duration, and scope. In light of the breadth, history, and "
    "constitutional context of that asserted authority, he must identify clear congressional "
    "authorization to exercise it. The emergency authority Trump attempted to rely on, the "
    "International Emergency Economic Powers Act, falls short. Justices Amy Coney Barrett "
    "and Neil Gorsuch, both Trump appointees, joined with Roberts and the three liberal "
    "justices in the majority. Justices Clarence Thomas, Samuel Alito and Brett Kavanaugh "
    "dissented. The president and Justice Department officials had framed the dispute in "
    "existential terms for the country. A group of small businesses who challenged the "
    "duties similarly warned that Trump's position represented a breathtaking assertion of "
    "power to effectively levy a tax without oversight from Congress. When Congress grants "
    "the power to impose tariffs, it does so clearly and with careful constraints. It did "
    "neither here. The 6-3 majority offered no clarity on the specific practical question "
    "of what to do with the money the administration has already collected. That question "
    "will likely need to be sorted out by lower courts. The case was the most significant "
    "involving the American economy to reach the Supreme Court in years, challenging the "
    "legality of Trump's Liberation Day tariffs, as well as duties he imposed on imports "
    "from China, Mexico and Canada. The so-called reciprocal tariffs raised duties as high "
    "as 50% on key trading partners, including India and Brazil, and as high as 145% on "
    "China in 2025. Trump has relied on a 1970s-era emergency law, IEEPA, to levy the "
    "import duties at issue in the case. That law allows a president to regulate importation "
    "during emergencies. The administration argued the word plainly includes the power to "
    "impose tariffs, but the businesses noted that the words tariff or duty never appear "
    "in the law. In 2023, the conservative majority relied on the major questions doctrine "
    "to block Biden's student loan forgiveness plan. A year earlier, the court stopped "
    "Biden's vaccine and testing requirement for 84 million Americans. Even some "
    "conservatives said the same logic should apply to Trump's tariffs."
)

# long, about 750 words
ART3 = (
    "There's been no slowdown to the U.S. military buildup around Iran despite both sides "
    "engaging in nuclear talks this week. President Donald Trump has threatened the Islamic "
    "Republic with direct action if diplomacy doesn't produce the desired results. And with "
    "a second aircraft carrier group headed to the Gulf, Trump's warnings that an attack on "
    "Iran would go beyond last summer's bombings on nuclear facilities leave a range of "
    "options open to the president. There could be new rounds of targeted strikes, "
    "assassinations of top leaders or a more sustained military campaign that could resemble "
    "something closer to a Third Gulf War. Trump has throughout the escalation voiced his "
    "preference for a diplomatic solution. He wants a nuclear deal that would go beyond the "
    "2015 agreement that limited Iran's nuclear program in exchange for sanctions relief. "
    "But distance persists in the negotiating positions between Washington and Tehran. "
    "Despite some progress in Tuesday's talks in Geneva, the White House said they were "
    "still very far apart on key issues. If they reject these terms, the U.S. is not only "
    "ready to take significant military action against nuclear facilities and ballistic "
    "missile production and launch sites. The administration may have decided to continuously "
    "degrade Iran's nuclear and ballistic missile program militarily until there is nothing "
    "to preserve by not signing on to a new agreement. While Iran is not currently assessed "
    "to hold any nuclear weapons and regularly denies any plan to conduct such an endeavor, "
    "the Islamic Republic possesses the largest missile and drone arsenal in the Middle East. "
    "These capabilities continue to pose a major threat in the event of a conflict. The "
    "farthest reach of these weapons is believed to extend at least 2,000 kilometers, "
    "putting the entire Middle East and parts of Southeastern Europe well within range. "
    "Much of Trump's talk surrounding a deal with Iran has dealt specifically with its "
    "nuclear program, though the U.S. leader and other top officials have said that Iran's "
    "missile capabilities would also need to be subject to limits under a new agreement. "
    "This position is also being pressed by Israeli Prime Minister Benjamin Netanyahu, who "
    "visited Trump last week at the White House. Iranian diplomats have resisted any talk of "
    "including the nation's missile arsenal in an initial deal and have also argued it was "
    "their nation's right to enrich uranium at lower levels. Iranian officials have "
    "repeatedly warned even a limited U.S. strike would spark an all-out war, with Supreme "
    "Leader Ayatollah Ali Khamenei threatening directly to sink U.S. warships. Iran's air "
    "defenses have proved largely ineffective against state-of-the-art U.S. and Israeli "
    "aircraft, though the Islamic Republic has long invested in heavily fortified "
    "subterranean complexes sometimes referred to as missile cities to house its missile "
    "and drone stockpiles. The U.S. has demonstrated the ability to overcome the challenge "
    "by penetrating deep underground with bunker buster munitions like the 30,000-pound "
    "GBU-57 Massive Ordnance Penetrators used to hit the nuclear sites at Fordow, Isfahan "
    "and Natanz. While Israel has long targeted top commanders among Iran's allies and is "
    "widely believed to be behind the assassination of nuclear scientists in Iran, Trump "
    "fired the first open shot against the Islamic Republic back in January 2020, ordering "
    "the strike that killed IRGC Quds Force commander Major General Qassem Soleimani at "
    "Baghdad International Airport in Iraq. Then, in June 2025, as the U.S. and Iran "
    "engaged in previous rounds of nuclear diplomacy, Israel launched an unprecedented "
    "series of strikes across Iran, killing scores of senior military officials and nuclear "
    "scientists, and devastating air defenses and other military sites over 12 days. Trump "
    "ultimately opted for a more limited bombing of Iran's most heavily shielded nuclear "
    "sites. The U.S. leader has more recently demonstrated his willingness to target "
    "leadership in the U.S. Delta Force raid that seized Venezuelan President Nicolas "
    "Maduro and his wife from their home in Caracas. At the moment, none of the many signs "
    "indicating looming U.S. military action against Iran suggest any form of large-scale "
    "invasion. While nothing near the Iraq War figure has been mobilized in the vicinity "
    "of Iran today, the presence of two aircraft carrier strike groups, combat and refueling "
    "aircraft and other equipment is likely sufficient to support a prolonged period of "
    "aerial operations. It seems clear that Gulf Arab partners do not want to see a military "
    "confrontation between the U.S. and Iran. It could significantly destabilize the region "
    "and impact the free flow of energy as Iran has threatened and does have the ability to "
    "sea mine the Strait of Hormuz."
)

def test_latency():
    articles = [
        ("short (~150w)", ART1),
        ("medium (~450w)", ART2),
        ("long (~750w)", ART3),
    ]
    times = []
    for label, text in articles:
        t0 = time.time()
        r = requests.post(f"{URL}/bias/analyze", json={"text": text})
        dt = time.time() - t0
        times.append(dt)
        data = r.json()
        assert r.status_code == 200
        assert data["checks"] > 0
        print(f"          {label}: {dt:.1f}s, {data['checks']} sentences, bias={data['overall_bias']}")

    avg = sum(times) / len(times)
    print(f"          avg: {avg:.1f}s")
    assert avg < 15


if __name__ == "__main__":
    print(f"\nBias detection api tests ({URL})")

    # make sure server is up
    try:
        requests.get(f"{URL}/health", timeout=5)
    except requests.ConnectionError:
        print("Server not running! Start it first")
        sys.exit(1)

    print("\nEdge cases:")
    run_test("health", test_health)
    run_test("empty input", test_empty)
    run_test("whitespace only", test_whitespace)
    run_test("no text field", test_no_text_field)
    run_test("gibberish", test_gibberish)
    run_test("one word", test_one_word)
    run_test("non-political", test_not_political)

    print("\nBias detection:")
    run_test("returns bias prediction", test_returns_bias)
    run_test("response has all fields", test_response_fields)
    run_test("confidence in range", test_confidence)
    run_test("valid categories", test_categories)
    run_test("issues <= checks", test_issues_leq_checks)

    print("\nExplainability:")
    run_test("returns top tokens", test_explain)
    run_test("empty input rejected", test_explain_empty)
    run_test("no stopwords in output", test_explain_no_stopwords)

    print("\nLatency (real articles):")
    run_test("article benchmarks", test_latency)

    print(f"{passes} passed, {fails} failed")
    sys.exit(0 if fails == 0 else 1)