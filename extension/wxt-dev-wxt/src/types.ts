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
		category: 'Left-leaning' | 'Right-leaning' | 'Centrist';
		description: string;
		valid: boolean;
	}[];
	fact_check_claims: {
		claim: string;
		evidence: string[];
		label: 'SUPPORTED' | 'REFUTED' | 'NOT ENOUGH INFO';
		valid: boolean;
	}[];
}

export type Stats = {
	leftBiasNum: number;
	rightBiasNum: number;
	centerBiasNum: number;
	supportedClaimNum: number;
	refutedClaimNum: number;
	noInfoClaimNum: number;
};
