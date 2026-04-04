export interface AnalysisResult {
	checks: number;
	issues: number;
	overall_bias: string;
	overall_probabilities: {
		Left: number;
		Center: number;
		Right: number;
	};
	bias_claims: {
		text: string;
		category: 'Left-leaning' | 'Possibly Left-leaning' | 'Right-leaning' | 'Possibly Right-leaning' | 'Centrist' | 'Uncertain';
		description: string;
		valid: boolean;
		confidence: number;
	}[];
	fact_check_claims: {
		claim: string;
		evidence: string[];
		label: 'SUPPORTED' | 'REFUTED' | 'NOT ENOUGH INFO';
		valid: boolean;
	}[];
}

export interface TokenAttribution {
	token: string;
	score: number;
}

export type Stats = {
	scansNum: number;
	leftBiasNum: number;
	rightBiasNum: number;
	centerBiasNum: number;
	supportedClaimNum: number;
	refutedClaimNum: number;
	noInfoClaimNum: number;
};
