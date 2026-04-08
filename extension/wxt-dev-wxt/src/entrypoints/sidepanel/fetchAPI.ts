import { AnalysisResult, TokenAttribution } from '@/types';
import { MODEL_BACKEND } from '@/config';
import { fetchWithTimeout } from '@/utils/fetchWithTimeout';

export interface SentenceBias {
	text: string;
	category: string;
	description: string;
	bias: string;
	confidence: number;
	probabilities: {
		Left: number;
		Center: number;
		Right: number;
	};
}

// export default async function fetchAPI(text: string) {
// 	console.log(text);
// 	const response = await fetch(`https://dummyjson.com/http/200?delay=5000`, {
// 		headers: { 'Content-Type': 'application/json' },
// 		method: 'POST',
// 		body: JSON.stringify({
// 			text: text,
// 		}),
// 	});

// 	if (response.status === 200) {
// 		const data = await response.json();
// 		return {
// 			checks: 3,
// 			issues: 2,
// 			claims: [
// 				{
// 					text: 'Freedom House published a report Wednesday downgrading the United States from a democracy to whatever political system lobsters have.',
// 					category: 'Unverified Source',
// 					description: 'Quotes or attributions lack confirmation from credible or official sources.',
// 					valid: false,
// 				},
// 				{
// 					text: 'Persistent executive overreach and erosion of civil liberties mean that America now looks less like a traditional federal republic',
// 					category: 'Accurate Contextual Information',
// 					description: 'Statement is accurate and supported by credible, verifiable information.',
// 					valid: true,
// 				},
// 				{
// 					text: 'Our nation already passed the tipping point where we might hope to match the deliberative bicameral legislative process of, say, shore crabs. At this juncture, there’s just too much scuttling in American politics to call it anything other than a flawed lobster republic.',
// 					category: 'False or Misleading',
// 					description: 'Claims are not supported by verifiable sources or evidence.',
// 					valid: false,
// 				},
// 			],
// 		};
// 	} else {
// 		// await throwAPIError(response)
// 	}
// }

function generateMockFactChecks(text: string): Array<{
	text: string;
	category: string;
	description: string;
	valid: boolean;
}> {
	// TODO: Replace this with real fact-checking API call
	// This is just mock data to show the UI framework
	const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);

	// Generate 2-3 mock fact-check claims
	const mockClaims = [];
	const categories = [
		{
			name: 'Unverified Source',
			desc: 'Quotes or attributions lack confirmation from credible or official sources.',
			valid: false,
		},
		{
			name: 'Accurate Contextual Information',
			desc: 'Statement is accurate and supported by credible, verifiable information.',
			valid: true,
		},
		{
			name: 'False or Misleading',
			desc: 'Claims are not supported by verifiable sources or evidence.',
			valid: false,
		},
		{
			name: 'Partially True',
			desc: 'Statement contains some truth but missing important context.',
			valid: true,
		},
		{
			name: 'Needs Verification',
			desc: 'Claim requires further fact-checking from authoritative sources.',
			valid: false,
		},
	];

	// Pick a few random sentences and assign mock fact-check results
	const numClaims = Math.min(3, sentences.length);
	for (let i = 0; i < numClaims; i++) {
		const sentence = sentences[i].trim();
		if (sentence) {
			const randomCategory = categories[Math.floor(Math.random() * categories.length)];
			mockClaims.push({
				text: sentence + '.',
				category: randomCategory.name,
				description: randomCategory.desc,
				valid: randomCategory.valid,
			});
		}
	}

	return mockClaims;
}

export default async function fetchAPI(text: string): Promise<AnalysisResult> {
	try {
		const response = await fetchWithTimeout(`${MODEL_BACKEND}/bias/analyze`, {
			headers: { 'Content-Type': 'application/json' },
			method: 'POST',
			body: JSON.stringify({ text }),
			timeoutMs: 30_000,
			retries: 1,
		});
		let fact_checks_response = []; // TODO: probably remove the try catch around this since this is only currently done because this API is not ready
		try {
			const response2 = await fetchWithTimeout(`${MODEL_BACKEND}/claim/verify-claims-from-passage`, {
				headers: { 'Content-Type': 'application/json' },
				method: 'POST',
				body: JSON.stringify({ text }),
				timeoutMs: 45_000,
				retries: 1,
			});
			if (response2.ok) {
				fact_checks_response = await response2.json();
			}
		} catch (e) {
			console.warn('Claim verification unavailable:', e);
		}

		if (!response.ok) {
			throw new Error(`API request failed: ${response.status}`);
		}

		const data = await response.json();

		// Map sentence bias results with explainability tokens
		const bias_claims = data.sentences.map((sentence: SentenceBias) => ({
			text: normalizeSpaces(sentence.text),
			category: sentence.category,
			description: sentence.description,
			valid: sentence.bias === 'Center' || sentence.confidence < 0.5,
			confidence: sentence.confidence,
		}));

		return {
			checks: data.checks,
			issues: data.issues,
			overall_bias: data.overall_bias,
			overall_probabilities: data.overall_probabilities,
			bias_claims,
			fact_check_claims: fact_checks_response,
		};
	} catch (error) {
		console.error('Error calling bias detection API:', error);
		throw error;
	}
}

export async function fetchExplain(text: string): Promise<TokenAttribution[]> {
	try {
		const response = await fetchWithTimeout(`${MODEL_BACKEND}/bias/explain`, {
			headers: { 'Content-Type': 'application/json' },
			method: 'POST',
			body: JSON.stringify({ text }),
			timeoutMs: 20_000,
			retries: 1,
		});

		if (!response.ok) {
			throw new Error(`Explain request failed: ${response.status}`);
		}

		const data = await response.json();
		return data.top_tokens || [];
	} catch (error) {
		console.error('Error calling explain API:', error);
		return [];
	}
}

function normalizeSpaces(str: string) {
	return str.replace(/\u00A0/g, ' ');
}
