import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { ShieldIcon, CheckCircleIcon, TriangleAlertIcon } from 'lucide-react';

interface AnalysisResult {
	checks: number;
	issues: number;
	overall_bias: string;
	overall_probabilities: {
		Left: number;
		Center: number;
		Right: number;
	};
	bias_claims: { text: string; category: string; description: string; valid: boolean }[];
	fact_check_claims: { text: string; category: string; description: string; valid: boolean }[];
}

export default function ClaimTab({
	result,
	currentHovered,
	handleHighlight,
}: {
	result: AnalysisResult | undefined;
	currentHovered: string | undefined;
	handleHighlight: Function;
}) {
	console.log(result);
	return (
		<>
			<Card>
				<CardHeader className="flex gap-2 items-center">
					<ShieldIcon size={20} />
					<p className="font-semibold text-base">Claim Verification</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">
					{result?.fact_check_claims.map((claim, idx) => (
						<div
							key={`fact-${idx}`}
							className={`bg-gray-50 flex rounded-b-sm gap-4 items-center py-2 pl-4 hover:cursor-pointer border-2 border-white ${
								currentHovered && claim.claim.includes(currentHovered) && 'border-yellow-200'
							}`}
							claim-text={claim.claim}
							onMouseEnter={() => handleHighlight(claim.claim)}
							onMouseLeave={() => handleHighlight('')}
						>
							{claim.valid ? (
								<CheckCircleIcon className="min-w-6 h-6 text-green-400" />
							) : (
								<TriangleAlertIcon className="min-w-6 h-6 text-red-400" />
							)}
							<div className="flex flex-col">
								<h3 className="text-sm">{claim.label}</h3>
								{claim.evidence.map((ev) => (
									<p className="text-xs text-gray-400">- {ev}</p>
								))}
							</div>
						</div>
					))}
				</CardContent>
			</Card>
		</>
	);
}
